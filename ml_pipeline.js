const fs = require('fs');
const path = require('path');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;
const csv = require('csv-parser');

// KMeans handles unsupervised physical clustering on pet data
const { kmeans } = require('ml-kmeans');
// Agnes handles hierarchical personality grouping (agglomerative nesting)
const hclust = require('ml-hclust');
// RandomForestClassifier evaluates multi-feature pet compatibility for late-stage adjustments
const { RandomForestClassifier } = require('ml-random-forest');
// Apriori analyzes session-based liked attributes to build association rules
const { Apriori } = require('node-apriori');

// Setup absolute path variables for database storage and cached models
const DB_DIR = path.join(__dirname, 'DB');
const MODELS_DIR = path.join(__dirname, 'models');
const STATE_FILE = path.join(MODELS_DIR, 'ml_pipeline_state.json');
const STATE_FILE_TEMP = path.join(MODELS_DIR, 'ml_pipeline_state.tmp.json');

// Paths to specific database CSV files
const PETS_CSV = path.join(DB_DIR, 'individual_pets.csv');
const INTERACTIONS_CSV = path.join(DB_DIR, 'interactions.csv');

// Verify that the necessary directory exists to store persistent files, and create it recursively if missing
if (!fs.existsSync(MODELS_DIR)) {
    fs.mkdirSync(MODELS_DIR, { recursive: true });
}

// Global configuration object storing hyperparameters and tuning variables for all ML modules
const ML_CONFIG = {
    // Multipliers for scaling each dimensions when performing physical distance assessments
    featureMultipliers: [1.0, 1.0, 1.0, 3.0, 1.5, 1.2, 1.2, 1.2, 1.2, 1.2],
    // Fusion weights determining the contribution of physical, behavioral, collaborative, and trend scoring layers
    fusionWeights: { physical: 0.35, behavioral: 0.20, cf: 0.30, apriori: 0.15 },
    // Minimum Jaccard similarity coefficient required to recognize user overlaps
    jaccardMinThreshold: 0.1,
    // Support threshold for Apriori transactions to filter out highly infrequent attribute patterns
    aprioriMinSupport: 0.05,
    // Lift threshold for Apriori rules to filter out trivial correlations
    aprioriMinLift: 1.2,
    // One hour window in milliseconds to group user swipe events into single sessions for association mining
    sessionWindowMs: 3600000,
    // Maximum number of features randomly split at each decision node in the Random Forest
    rfMaxFeatures: 3,
    // Total number of decision trees in the Random Forest ensemble
    rfEstimators: 50,
    // Cap on the maximum number of clusters considered when detecting the optimal K
    maxK: 10,
    // Strict boundary from Chi-Square distribution (3 degrees of freedom at 99% confidence) for outlier filtering
    chiSquareThreshold: 11.34
};

// Global in-memory variables to cache trained states and accelerate runtime matching queries
let globalRfModel = null; // Holds the active Random Forest classifier
let globalCentroids = null; // Stores K-Means physical centroids
let globalScalingParams = { means: [0, 0, 0], stdDevs: [1, 1, 1] }; // Physical Z-Score normalisation variables
let globalOptimalK = 3; // Dynamic count of active physical clusters
let globalClusterStats = {}; // Stores means and inverse covariance matrices for Mahalanobis distances
let globalAgnesMap = {}; // Maps pet usernames to behavioral archetype IDs
let globalElbowWcss = []; // Stores Within-Cluster Sum of Squares for Elbow Chart
let globalAgnesTreeCache = null; // Caches the full complete-linkage dendrogram tree
let globalUserSimMatrix = {}; // Stores Jaccard similarity coordinates between active users
let globalItemLikers = {}; // Inverted index mapping pets to users who liked them
let globalItemSkippers = {}; // Inverted index mapping pets to users who skipped them
let globalAprioriRules = {}; // Stores active item association rules map

// Cache mechanism tracking dataset size to prevent redundant retraining on identical datasets
let lastTrainingState = { petsCount: 0, interactionsCount: 0 };

// Restore all model states from the local models folder on server boot to prevent cold starts
async function loadStateFromLocal() {
    try {
        if (fs.existsSync(STATE_FILE)) {
            // Read and parse the persistent JSON state file synchronously
            const data = fs.readFileSync(STATE_FILE, 'utf8');
            const state = JSON.parse(data);

            // Reconstruct in-memory states from the cached values
            globalCentroids = state.centroids || null;
            globalScalingParams = state.scalingParams || { means: [0, 0, 0], stdDevs: [1, 1, 1] };
            globalOptimalK = state.optimalK || 3;
            globalClusterStats = state.clusterStats || {};
            globalAgnesMap = state.agnesMap || {};
            globalElbowWcss = state.elbowWcss || [];

            // Deserialise the Apriori Map array back into active Map structures
            if (state.aprioriRules) {
                globalAprioriRules = {};
                state.aprioriRules.forEach(([key, rulesArr]) => {
                    globalAprioriRules[key] = new Map(rulesArr);
                });
            }
            // If a Random Forest model exists in the JSON, load its tree structures
            if (state.rfModelJSON) {
                globalRfModel = RandomForestClassifier.load(state.rfModelJSON);
            }
            console.log("ML State fully restored from local models folder.");
        }
    } catch (err) {
        // Fallback gracefully on first boot when no JSON state is present
        console.log("Local restore skipped/failed (normal for first boot):", err.message);
    }
}

// Synchronise the current model weights and clusters to the persistent filesystem
async function syncToLocal() {
    try {
        // Serialise the Map structures into plain nested arrays for JSON compatibility
        const aprioriSerialized = Object.entries(globalAprioriRules).map(([key, map]) => [key, Array.from(map.entries())]);
        const state = {
            centroids: globalCentroids,
            scalingParams: globalScalingParams,
            optimalK: globalOptimalK,
            clusterStats: globalClusterStats,
            agnesMap: globalAgnesMap,
            aprioriRules: aprioriSerialized,
            rfModelJSON: globalRfModel ? globalRfModel.toJSON() : null,
            elbowWcss: globalElbowWcss
        };

        // Write atomically: save to a temporary file first, then perform an atomic rename
        // This ensures the live state file is never corrupted if the server crashes mid-write
        fs.writeFileSync(STATE_FILE_TEMP, JSON.stringify(state), 'utf8');
        fs.renameSync(STATE_FILE_TEMP, STATE_FILE);
    } catch (err) {
        console.error("Local sync failed:", err.message);
    }
}

// Helper parsing CSV data streams into structured arrays using Promises
const readCsv = (filePath) => {
    return new Promise((resolve) => {
        const results = [];
        // Resolve early with an empty list if the target CSV doesn't exist yet
        if (!fs.existsSync(filePath)) return resolve(results);
        const stream = fs.createReadStream(filePath);
        // Pipe the file stream to csv-parser and push rows to results on data event
        stream.pipe(csv()).on('data', d => results.push(d)).on('end', () => resolve(results));
    });
};

// Helper capturing structured arrays and writing them back to a clean CSV on disk
const writeCsv = (filePath, headers, records) => createCsvWriter({ path: filePath, header: headers }).writeRecords(records);

// Fisher-Yates shuffle algorithm for in-place array randomisation in O(N) time complexity
function fisherYatesShuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        // Swap elements using destructuring assignment syntax
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

/*
==================================================================================
ALGORITHM 1: Z-SCORE PHYSICAL NORMALISATION (Standardizing Pet Sizes)
==================================================================================
* INTUITION: Comparing a 40kg dog to a 4kg cat is numerically massive, while comparing their ages (e.g., 2 years vs 3 years) is tiny. Z-Score normalisation rescales these metrics (weight, length, age) so that a large difference in weight doesn't completely overwhelm a critical age difference when finding matches.
* HOW IT WORKS: Subtracts the average pet weight/length/age from a pet's raw features and divides by the global standard deviation (using Bessel's N-1 correction for accurate small-database scaling).
* WHY IT IS THERE: Ensures our physical similarity search treats weight, length, and age with equal scientific importance rather than letting weight dominate the matches.
==================================================================================
*/
function preprocess(petData) {
    // Clone the petData object using the spread operator to prevent mutating variables out of scope
    const processed = { ...petData };
    // Enforce birthYear to be an integer, defaulting to 2020 on failure
    processed.birthYear = parseInt(processed.birthYear) || 2020;
    // Enforce weight to be a float, defaulting to 10.0 kg on failure
    processed.weight = parseFloat(processed.weight) || 10.0;
    // Enforce length to be a float, defaulting to 20.0 cm on failure
    processed.length = parseFloat(processed.length) || 20.0;
    // Lowercase vaccination status string to standardise comparisons
    processed.vaccination = (processed.vaccination || '').toLowerCase();
    return processed;
}

// Extract the weight, length, and age vector [w, l, a] from raw pet attributes
function extractPhysical(p) {
    return [parseFloat(p.weight) || 10, parseFloat(p.length) || 20, new Date().getFullYear() - (parseInt(p.birthYear) || 2020)];
}

// Extract a 5-D binary trait representation vector for AGNES clustering from pet personalities list
function extractBehavioral(p) {
    const traits = (p.personality || '').toLowerCase().split(',').map(t => t.trim());
    // Map each trait to 1 if present, 0 if absent, yielding [active, friendly, calm, touchy, sleepy]
    return ['active', 'friendly', 'calm', 'touchy', 'sleepy'].map(t => traits.includes(t) ? 1 : 0);
}

// Standardise all elements of physical vectors using the pre-computed global mean and standard deviation
function standardizePhysical(features) {
    // Apply standard Z-Score normalization: (Value - Mean) / StdDev
    return features.map(row => row.map((val, idx) => (val - globalScalingParams.means[idx]) / globalScalingParams.stdDevs[idx]));
}


/*
==================================================================================
ALGORITHM 2: MAHALANOBIS DISTANCE & CHI-SQUARE OUTLIER DETECTION (Spam Profile Gatekeeper)
==================================================================================
* INTUITION: Standard distance checks only look at individual limits (e.g., age or weight). Mahalanobis distance looks at how features correlate—like how a pet's length and weight should scale together. It easily catches anomalies like a tiny 20cm Chihuahua profile that claims to weigh 80kg.
* HOW IT WORKS: Scales physical offsets using the inverse covariance matrix of the pet's size cluster, then flags profiles as anomalous if they exceed the Chi-Square 99% confidence threshold (11.34).
* WHY IT IS THERE: Acts as our automated database moderator, flagging suspicious or joke pet profiles before they can dilute standard centroid dimensions.
==================================================================================
*/
function getMahalanobisDistanceSq(scaledPoint, stats) {
    // Return Infinity if the cluster has no valid covariance inversion matrix (unable to calculate)
    if (!stats || !stats.invCovMatrix) return Infinity;
    // Calculate the difference vector: (x - meanVector)
    const diff = scaledPoint.map((val, i) => val - stats.meanVector[i]);
    let mahalanobisSq = 0;
    // Perform standard matrix multiplication: diff^T * invCov * diff
    for (let i = 0; i < 3; i++) {
        let temp = 0;
        for (let j = 0; j < 3; j++) temp += diff[j] * stats.invCovMatrix[j][i];
        mahalanobisSq += temp * diff[i];
    }
    return mahalanobisSq;
}

// Assign a standardized point to the nearest physical cluster using covariance structures
function findNearestCluster(scaledPhys) {
    // Return default cluster 0 if K-Means has not been trained yet
    if (!globalCentroids) return { clusterIdx: 0, distanceSq: 0 };
    let minDistance = Infinity, closestCluster = 0;

    // Iterate through all centroids and find the one minimizing the distance metric
    globalCentroids.forEach((centroid, index) => {
        const stats = globalClusterStats[index];
        let dist;
        // Use Mahalanobis distance if covariance stats exist, otherwise fallback to standard Euclidean
        if (stats && stats.invCovMatrix) {
            dist = getMahalanobisDistanceSq(scaledPhys, stats);
        } else {
            const coords = Array.isArray(centroid) ? centroid : centroid.centroid;
            dist = coords.reduce((sum, val, d) => sum + Math.pow(val - scaledPhys[d], 2), 0);
        }
        // Track the minimum distance seen
        if (dist < minDistance) { minDistance = dist; closestCluster = index; }
    });
    return { clusterIdx: closestCluster, distanceSq: minDistance };
}

// Gatekeeper security checkpoint flagging anomalous inputs using cluster Mahalanobis limits
function gatekeeper(rawPetData) {
    // Immediately block the profile if critical physical statistics are non-numeric
    if (isNaN(parseFloat(rawPetData.weight)) || isNaN(parseInt(rawPetData.birthYear)) || isNaN(parseFloat(rawPetData.length))) return true;

    const age = new Date().getFullYear() - parseInt(rawPetData.birthYear);
    // Hard check: flag ages greater than 25, or cats weighing more than 40kg (unrealistic)
    if (age > 25 || ((rawPetData.type || '').toLowerCase() === 'cat' && parseFloat(rawPetData.weight) > 40)) return true;
    // Flag descriptions containing fake or spam parameters in their vaccination details
    if (['spam', 'fake', 'test'].includes((rawPetData.vaccination || '').toLowerCase())) return true;

    // If clusters are available, check Mahalanobis distance against the Chi-Square limit
    if (globalCentroids && Object.keys(globalClusterStats).length > 0) {
        const scaledPhys = standardizePhysical([extractPhysical(rawPetData)])[0];
        const { clusterIdx, distanceSq } = findNearestCluster(scaledPhys);

        const stats = globalClusterStats[clusterIdx];
        // If distance squared exceeds the Chi-Square 99% boundary of 11.34, flag as outlier
        if (stats && stats.invCovMatrix && distanceSq > ML_CONFIG.chiSquareThreshold) return true;
    } else if (age > 25 || parseFloat(rawPetData.weight) > 150) {
        // Fallback outlier check if models have not yet completed their initial training pass
        return true;
    }
    return false;
}

// Invert the 3x3 covariance matrix analytically using determinants
function invert3x3Covariance(matrix, epsilon = 1e-4) {
    // Add Ridge penalty epsilon down the diagonal to guarantee non-singularity and mathematical stability
    let m = matrix.map((row, i) => row.map((val, j) => val + (i === j ? epsilon : 0)));
    // Compute the matrix determinant using Sarrus' rule expansion
    const det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    // Fallback to identity matrix if the determinant is zero (singular covariance)
    if (Math.abs(det) < 1e-8) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const invDet = 1.0 / det;
    // Map the adjugate matrix scaled by 1/determinant to return the inverse covariance
    return [
        [(m[1][1] * m[2][2] - m[2][1] * m[1][2]) * invDet, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invDet, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet],
        [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invDet, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet, (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * invDet],
        [(m[1][0] * m[2][1] - m[2][0] * m[1][1]) * invDet, (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * invDet, (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * invDet]
    ];
}

// Compute the Within-Cluster Sum of Squares (WCSS) to evaluate KMeansclustering density
function computeWCSS(data, clusters, centroids) {
    let wcss = 0;
    for (let i = 0; i < data.length; i++) {
        const clusterIdx = clusters[i];
        const coords = Array.isArray(centroids[clusterIdx]) ? centroids[clusterIdx] : centroids[clusterIdx].centroid;
        // Sum the squared Euclidean distances from each data point to its assigned centroid
        wcss += data[i].reduce((sum, val, d) => sum + Math.pow(val - coords[d], 2), 0);
    }
    return wcss;
}


/*
==================================================================================
ALGORITHM 3: KNEEDLE ELBOW DETECTION (Finding Optimal Pet Cohort Counts)
==================================================================================
* INTUITION: We want to sort our pet database into physical size cohorts. Having only 1 size cohort is too broad, but having 100 cohorts means almost every pet is in its own group. Elbow detection plots the clustering "messiness" drop-off and picks the sweet spot where adding more groups ceases to improve size classification.
* HOW IT WORKS: Normalizes the WCSS (Within-Cluster Sum of Squares) for size cluster counts K (1 to 10), projects them on a 0-to-1 grid, and finds the K value with the furthest perpendicular distance to the start-to-end chord line.
* WHY IT IS THERE: Dynamically calculates the perfect number of size cohorts (K) to use as new pet registrations scale up the database over time.
==================================================================================
*/
function findOptimalK(data, maxK = 10) {
    const limit = Math.min(maxK, data.length - 1);
    // Return basic fallback cluster range if the dataset is too small
    if (limit <= 2) {
        globalElbowWcss = [0, 0, 0, 0, 0, 0, 0, 0];
        return Math.max(2, limit);
    }

    const wcssValues = [];
    // Compute WCSS for K values ranging from 1 up to limit
    for (let k = 1; k <= limit; k++) {
        const result = kmeans(data, k, { initialization: 'kmeans++' });
        wcssValues.push(computeWCSS(data, result.clusters, result.centroids));
    }
    globalElbowWcss = [...wcssValues];

    const minWcss = Math.min(...wcssValues);
    const maxWcss = Math.max(...wcssValues);
    // Normalise WCSS values to fit within a 0-1 unit square to keep distance calculations unbiased
    const normWcss = wcssValues.map(w => (maxWcss - minWcss === 0) ? 0 : (w - minWcss) / (maxWcss - minWcss));
    const normK = Array.from({ length: limit }, (_, i) => i / (limit - 1));

    // Formulate straight line equation coefficients (Ax + By + C = 0) connecting first and last points
    const A = normWcss[limit - 1] - normWcss[0];
    const B = normK[0] - normK[limit - 1];
    const C = normK[limit - 1] * normWcss[0] - normK[0] * normWcss[limit - 1];
    const denominator = Math.sqrt(A * A + B * B);

    let maxDistance = -1, optimalK = 2;
    // Calculate perpendicular distance to the line for each point to locate the elbow/knee
    for (let i = 0; i < limit; i++) {
        const perpDistance = Math.abs(A * normK[i] + B * normWcss[i] + C) / denominator;
        if (perpDistance > maxDistance) { maxDistance = perpDistance; optimalK = i + 1; }
    }
    return Math.max(2, optimalK);
}

// This background loop trains the physical clusters, behavior trees, collaborative filtering index,
// and random forest models asynchronously every few minutes.
async function trainBackgroundModels() {
    let pets = await readCsv(PETS_CSV);
    // Ignore all suspicious/flagged profiles during standard training loops to keep models unpolluted
    const approvedPets = pets.filter(p => p.isFlagged !== 'true');
    const interactions = await readCsv(INTERACTIONS_CSV);

    // Skip training execution entirely if dataset sizes have not changed since the last training cycle
    if (approvedPets.length === lastTrainingState.petsCount && interactions.length === lastTrainingState.interactionsCount) {
        return;
    }

    // Refresh training state markers
    lastTrainingState.petsCount = approvedPets.length;
    lastTrainingState.interactionsCount = interactions.length;

    // Minimum requirement of 3 pets to perform spatial clustering operations
    if (approvedPets.length >= 3) {
        /*
        ==================================================================================
        ALGORITHM 4: K-MEANS PHYSICAL CLUSTERING (Sizing Cohort Partitioning)
        ==================================================================================
        * INTUITION: Automatically partitions our registered pets into size groups (like Tiny Puppies, Medium Adults, or Giant Seniors) purely based on their standardized physical traits.
        * HOW IT WORKS: Distributes initial centroid seeds using kmeans++ to guarantee balanced clustering, assigns each pet to the closest size centroid, and updates centroid centers until assignments lock.
        * WHY IT IS THERE: Grouping pets by physical similarity establishes the first layer of search compatibility, ensuring small lap dogs aren't immediately matched with giant mastiffs.
        ==================================================================================
        */
        const rawPhysFeatures = approvedPets.map(extractPhysical);
        const means = [0, 0, 0], stdDevs = [1, 1, 1];
        // Calculate the mean and sample standard deviation (using Bessel's correction N-1) for each physical dimension
        for (let idx = 0; idx < 3; idx++) {
            const vals = rawPhysFeatures.map(f => f[idx]);
            const m = vals.reduce((a, b) => a + b, 0) / vals.length;
            const v = vals.reduce((sum, val) => sum + Math.pow(val - m, 2), 0) / (vals.length > 1 ? vals.length - 1 : 1);
            means[idx] = m; stdDevs[idx] = Math.sqrt(v) || 0.001; // Avoid divide-by-zero standard deviation using small fallback
        }
        globalScalingParams = { means, stdDevs };
        // Normalise vectors to standardised physical points
        const scaledPhysFeatures = standardizePhysical(rawPhysFeatures);

        // Dynamically locate elbow and execute KMeans clustering
        globalOptimalK = findOptimalK(scaledPhysFeatures, ML_CONFIG.maxK);
        const kmeansResult = kmeans(scaledPhysFeatures, globalOptimalK, { initialization: 'kmeans++' });
        globalCentroids = kmeansResult.centroids;

        // Assign each approved pet to their clusterGroup
        approvedPets.forEach((p, i) => {
            p.clusterGroup = kmeansResult.clusters[i];
        });

        // Write the updated pets list back to individual_pets.csv
        const petHeaders = [
            { id: 'username', title: 'username' }, { id: 'petName', title: 'petName' },
            { id: 'type', title: 'type' }, { id: 'gender', title: 'gender' },
            { id: 'birthYear', title: 'birthYear' }, { id: 'vaccination', title: 'vaccination' },
            { id: 'breed', title: 'breed' }, { id: 'length', title: 'length' },
            { id: 'weight', title: 'weight' }, { id: 'color', title: 'color' },
            { id: 'personality', title: 'personality' }, { id: 'photoPath', title: 'photoPath' },
            { id: 'isFlagged', title: 'isFlagged' }, { id: 'clusterGroup', title: 'clusterGroup' }
        ];
        await writeCsv(PETS_CSV, petHeaders, pets);

        // Reset the cluster stats dictionary and compute covariance properties for each active cluster
        globalClusterStats = {};
        for (let k = 0; k < globalOptimalK; k++) {
            const pts = scaledPhysFeatures.filter((_, i) => kmeansResult.clusters[i] === k);
            // Require at least 4 coordinates within a cluster to avoid mathematically singular covariance profiles
            if (pts.length > 3) {
                const meanVector = [0, 0, 0];
                pts.forEach(p => { for (let d = 0; d < 3; d++) meanVector[d] += p[d]; });
                meanVector.forEach((_, d) => meanVector[d] /= pts.length);

                const covMatrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
                // Sum the outer products of differences to assemble the covariance matrix
                pts.forEach(p => {
                    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) covMatrix[i][j] += (p[i] - meanVector[i]) * (p[j] - meanVector[j]);
                });
                // Unbiased sample covariance factor division by pts.length - 1
                for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) covMatrix[i][j] /= (pts.length - 1);
                globalClusterStats[k] = { meanVector, invCovMatrix: invert3x3Covariance(covMatrix) };
            }
        }

        /*
        ==================================================================================
        ALGORITHM 5: AGNES HIERARCHICAL BEHAVIORAL ARCHETYPES (Personality Segmentation)
        ==================================================================================
        * INTUITION: Instead of drawing rigid circles around behaviors, we build a behavioral family tree connecting pets by their personality overlaps (active, friendly, touchy, calm, sleepy). We cut this tree at the largest vertical gap to cluster pets into highly distinct playdate personality styles.
        * HOW IT WORKS: Computes Hamming distances across personality trait profiles, merges similar profiles iteratively using AGNES complete-linkage, and cuts the resulting tree at the height maximizing cluster separation.
        * WHY IT IS THERE: Groups pets into behavioral play archetypes (e.g., "Hyper Playmate" vs "Chill Couch Potato") to recommend matches with compatible social energy.
        ==================================================================================
        */
        const behavioralFeatures = approvedPets.map(extractBehavioral);
        const distMatrix = [];
        // Construct the distance matrix using Hamming distance metrics across all behavioral trait pairs
        for (let i = 0; i < behavioralFeatures.length; i++) {
            distMatrix[i] = [];
            for (let j = 0; j < behavioralFeatures.length; j++) {
                let diff = 0;
                for (let d = 0; d < 5; d++) if (behavioralFeatures[i][d] !== behavioralFeatures[j][d]) diff++;
                distMatrix[i][j] = diff;
            }
        }

        // Run AGNES complete linkage hierarchical clustering
        const tree = hclust.agnes(distMatrix, { method: 'complete', isDistanceMatrix: true });
        globalAgnesTreeCache = tree;

        let maxGap = 0, bestCutHeight = tree.height;
        // Traverse the tree recursively to find the split level with the largest gap difference
        function findLargestGap(node) {
            if (node.isLeaf) return;
            const childMaxHeight = Math.max(...node.children.map(c => c.height));
            const gap = node.height - childMaxHeight;
            if (gap > maxGap) { maxGap = gap; bestCutHeight = childMaxHeight + (gap / 2); }
            node.children.forEach(findLargestGap);
        }
        findLargestGap(tree);

        // Slice the hierarchy tree at the computed bestCutHeight level
        let nodes = [tree];
        while (nodes.some(n => n.height > bestCutHeight && !n.isLeaf)) {
            let splitIndex = nodes.findIndex(n => n.height > bestCutHeight && !n.isLeaf);
            let target = nodes.splice(splitIndex, 1)[0];
            nodes.push(...target.children);
        }

        // Helper aggregating all leaf index values beneath a target cluster node
        const extractLeaves = (node, arr) => {
            if (node.isLeaf) arr.push(node.index);
            else node.children.forEach(c => extractLeaves(c, arr));
            return arr;
        };

        const petUsernames = approvedPets.map(p => p.username);
        globalAgnesMap = {};
        // Map each pet username to its corresponding behavioral archetype index
        nodes.forEach((clusterNode, archetypeId) => {
            extractLeaves(clusterNode, []).forEach(idx => {
                const uname = petUsernames[idx];
                if (uname) globalAgnesMap[uname] = archetypeId;
            });
        });
    }

    /*
    ==================================================================================
    ALGORITHM 6: JACCARD COLLABORATIVE FILTERING (Finding Matchmaker Swipe Soulmates)
    ==================================================================================
    * INTUITION: If two pet owners swipe 'like' and 'skip' on the exact same pets, they share highly similar match preferences. If they like a new pet you haven't seen yet, it is highly recommended to you.
    * HOW IT WORKS: Uses inverted lookup indices (pet -> liking/skipping owners) to calculate the Jaccard similarity (Intersection / Union) only between owners who have swiped on at least one overlapping pet.
    * WHY IT IS THERE: Powers our social-proof playdate engine by leveraging shared community preferences, suggesting matches that pure physical sizes would miss.
    ==================================================================================
    */
    const userLikes = {}, userSkips = {};
    const itemLikers = {}, itemSkippers = {};

    interactions.forEach(i => {
        // Initialize structures if undefined
        if (!userLikes[i.username]) userLikes[i.username] = new Set();
        if (!userSkips[i.username]) userSkips[i.username] = new Set();
        // Parse the interaction action and update inverted indices mapping targets to users
        if (i.action === 'like') {
            userLikes[i.username].add(i.targetUsername);
            if (!itemLikers[i.targetUsername]) itemLikers[i.targetUsername] = new Set();
            itemLikers[i.targetUsername].add(i.username);
        } else if (i.action === 'skip') {
            userSkips[i.username].add(i.targetUsername);
            if (!itemSkippers[i.targetUsername]) itemSkippers[i.targetUsername] = new Set();
            itemSkippers[i.targetUsername].add(i.username);
        }
    });

    globalItemLikers = itemLikers;
    globalItemSkippers = itemSkippers;

    // Collect list of all unique user names present in interactions history
    const users = Array.from(new Set([...Object.keys(userLikes), ...Object.keys(userSkips)]));
    const newSimMatrix = {};
    for (const u of users) newSimMatrix[u] = {};

    // Compute pairwise Jaccard similarities using the optimized inverted mapping structures
    for (let i = 0; i < users.length; i++) {
        const u1 = users[i];
        const u1Items = new Set([...(userLikes[u1] || []), ...(userSkips[u1] || [])]);

        // Construct a list of candidates who swiped on at least one item shared with u1
        const candidateUsers = new Set();
        u1Items.forEach(item => {
            if (itemLikers[item]) itemLikers[item].forEach(u => candidateUsers.add(u));
            if (itemSkippers[item]) itemSkippers[item].forEach(u => candidateUsers.add(u));
        });
        candidateUsers.delete(u1); // Avoid calculating self-similarity

        // Calculate overlap coefficients for each intersecting candidate
        candidateUsers.forEach(u2 => {
            if (u1 < u2) { // Enforce undirected edge index constraint to compute pairs only once
                const u2Items = new Set([...(userLikes[u2] || []), ...(userSkips[u2] || [])]);

                let intersect = 0;
                for (let item of u1Items) if (u2Items.has(item)) intersect++;

                // Formula: Jaccard = Intersection / Union
                const union = u1Items.size + u2Items.size - intersect;
                const sim = union === 0 ? 0 : intersect / union;

                // Cache Jaccard similarity coordinates if they exceed configuration thresholds
                if (sim > ML_CONFIG.jaccardMinThreshold) {
                    newSimMatrix[u1][u2] = sim;
                    newSimMatrix[u2][u1] = sim;
                }
            }
        });
    }
    globalUserSimMatrix = newSimMatrix;

    /*
    ==================================================================================
    ALGORITHM 7: RANDOM FOREST LATE-STAGE RATING ADJUSTER (Playdate Success Predictor)
    ==================================================================================
    * INTUITION: Learns from historical playdate feedback. It trains a forest of 50 decision trees to look at a pair of pets (size difference, breed match, behavioral overlaps) and votes on the probability of a swipe match.
    * HOW IT WORKS: Balance-samples historically liked and skipped playdates, extracts physical and behavioral diffs, trains 50 decision trees, and applies consensus votes to adjust recommendation ranks.
    * WHY IT IS THERE: Acts as the final quality checkpoint, predicting non-linear relationship compatibility before displaying recommendations in the user feed.
    ==================================================================================
    */
    if (interactions.length >= 20 && approvedPets.length > 0) {
        let allData = [], allLabels = [];
        interactions.forEach(interaction => {
            const userPet = pets.find(p => p.username === interaction.username);
            const targetPet = pets.find(p => p.username === interaction.targetUsername);
            // Construct differences feature vector for each completed interaction
            if (userPet && targetPet) {
                const f1 = standardizePhysical([extractPhysical(userPet)])[0];
                const f2 = standardizePhysical([extractPhysical(targetPet)])[0];
                // Difference vector: Absolute physical parameter differences
                const diffVector = f1.map((val, i) => Math.abs(val - f2[i]));

                const isSameBreed = (userPet.breed && targetPet.breed && userPet.breed === targetPet.breed) ? 1 : 0;
                const beh1 = extractBehavioral(userPet);
                const beh2 = extractBehavioral(targetPet);

                let behOverlap = 0;
                for (let i = 0; i < 5; i++) if (beh1[i] === 1 && beh2[i] === 1) behOverlap++;

                // Append categorical sameBreed and normalized behavior overlap to the features
                diffVector.push(isSameBreed, behOverlap / 5.0);
                allData.push(diffVector);
                allLabels.push(interaction.action === 'like' ? 1 : 0);
            }
        });

        // Split target sets into positive (likes) and negative (skips) lists
        const lIdx = allLabels.map((l, i) => l === 1 ? i : -1).filter(i => i !== -1);
        const sIdx = allLabels.map((l, i) => l === 0 ? i : -1).filter(i => i !== -1);

        // We balance our training set by downsampling the majority class (likes vs skips).
        // If we train on highly skewed classes, the forest will just learn to predict the majority class.
        if (lIdx.length > 0 && sIdx.length > 0) {
            const minSize = Math.min(lIdx.length, sIdx.length);
            // Downsample via shuffling and slicing the indices to balance class frequencies 1:1
            let bIdx = [...fisherYatesShuffle(lIdx).slice(0, minSize), ...fisherYatesShuffle(sIdx).slice(0, minSize)];
            bIdx = fisherYatesShuffle(bIdx);

            const bData = bIdx.map(i => allData[i]);
            const bLabels = bIdx.map(i => allLabels[i]);

            // Execute an 80/20 train/validation split
            const split = Math.floor(bData.length * 0.8);
            if (split > 0) {
                const tempModel = new RandomForestClassifier({ maxFeatures: ML_CONFIG.rfMaxFeatures, replacement: true, nEstimators: ML_CONFIG.rfEstimators });
                // Train the random forest using the 80% split
                tempModel.train(bData.slice(0, split), bLabels.slice(0, split));

                let correct = 0;
                const testLabels = bLabels.slice(split);
                // Predict labels on the 20% validation split
                const preds = tempModel.predict(bData.slice(split));
                for (let i = 0; i < testLabels.length; i++) if (preds[i] === testLabels[i]) correct++;

                // Keep model only if validation accuracy meets or exceeds 70%
                if ((correct / testLabels.length) >= 0.70) globalRfModel = tempModel;
            }
        }
    }
    await syncToLocal();
}

/*
==================================================================================
ALGORITHM 8: APRIORI ASSOCIATION TREND MINING (Session-Based Playdate Preferences)
==================================================================================
* INTUITION: If an owner likes a "golden retriever" and a "young active dog" in their current swiping session, they are in the mood for high-energy retriever playmates. Apriori captures these temporary session trends.
* HOW IT WORKS: Pools likes into 1-hour swiping session baskets, counts occurrences of breeds/ages/traits to extract frequent itemsets, and maps rules with a Lift > 1.2 to boost current feed recommendations.
* WHY IT IS THERE: Adapts recommendations in real-time to the owner's immediate swiping mood, prioritizing active trends as they browse the app.
==================================================================================
*/
async function runApriori() {
    const interactions = await readCsv(INTERACTIONS_CSV);
    const pets = await readCsv(PETS_CSV);

    const sessionLikes = {};
    const itemTxFrequencies = {};

    // Iterate through interactions and group positive likes by user session windows
    interactions.filter(i => i.action === 'like').forEach(i => {
        const targetPet = pets.find(p => p.username === i.targetUsername);
        if (targetPet) {
            // Group session by dividing the timestamp by the window multiplier (1 hour)
            const sessionId = `${i.username}_${Math.floor(i.timestamp / ML_CONFIG.sessionWindowMs)}`;
            if (!sessionLikes[sessionId]) sessionLikes[sessionId] = new Set();

            // We map breed, type, gender, age category, and behavioral traits into distinct items for association analysis.
            if (targetPet.breed) sessionLikes[sessionId].add(`breed_${targetPet.breed}`);
            if (targetPet.type) sessionLikes[sessionId].add(`type_${targetPet.type.toLowerCase()}`);
            if (targetPet.gender) sessionLikes[sessionId].add(`gender_${targetPet.gender.toLowerCase()}`);

            const age = new Date().getFullYear() - (parseInt(targetPet.birthYear) || 2020);
            const ageGroup = age < 2 ? 'young' : age < 8 ? 'adult' : 'senior';
            sessionLikes[sessionId].add(`age_${ageGroup}`);

            const traits = (targetPet.personality || '').toLowerCase().split(',').map(t => t.trim());
            traits.forEach(t => {
                if (['active', 'friendly', 'calm', 'touchy', 'sleepy'].includes(t)) {
                    sessionLikes[sessionId].add(`trait_${t}`);
                }
            });
        }
    });

    const transactions = Object.values(sessionLikes).map(set => Array.from(set));
    const totalTx = transactions.length;
    // Require a minimum threshold of 5 completed user sessions to build viable statistics
    if (totalTx >= 5) {
        // Count transaction occurrence frequencies across all unique items
        transactions.forEach(tx => {
            const uniqueItems = new Set(tx);
            uniqueItems.forEach(item => { itemTxFrequencies[item] = (itemTxFrequencies[item] || 0) + 1; });
        });

        // Initialize node-apriori algorithm using minimum support constraints
        return new Promise((resolve) => {
            const apriori = new Apriori(ML_CONFIG.aprioriMinSupport);
            apriori.exec(transactions).then(result => {
                const newRules = {};

                result.itemsets.forEach(itemset => {
                    // We only look at pairs (size 2) and compute their support and lift.
                    // Lift tells us how much more likely they are to be liked together compared to random chance.
                    if (itemset.items.length === 2) {
                        const itemA = itemset.items[0];
                        const itemB = itemset.items[1];

                        const supportA = itemTxFrequencies[itemA] / totalTx;
                        const supportB = itemTxFrequencies[itemB] / totalTx;
                        // Lift formula: Support(A & B) / (Support(A) * Support(B))
                        const lift = itemset.support / (supportA * supportB);

                        // If rule lift is significant, map the association rules symmetrically
                        if (lift > ML_CONFIG.aprioriMinLift) {
                            if (!newRules[itemA]) newRules[itemA] = new Map();
                            if (!newRules[itemB]) newRules[itemB] = new Map();
                            newRules[itemA].set(itemB, lift);
                            newRules[itemB].set(itemA, lift);
                        }
                    }
                });
                globalAprioriRules = newRules;
                syncToLocal(); // Cache trained apriori coefficients locally
                resolve(result.itemsets);
            });
        });
    } else {
        globalAprioriRules = {};
        await syncToLocal();
    }
    return [];
}

// Assigns a newly registered pet profile to its closest K-Means cluster index
function assignToCluster(newPetData) {
    const scaledPhys = standardizePhysical([extractPhysical(newPetData)])[0];
    return findNearestCluster(scaledPhys).clusterIdx;
}

// --- THE MATCHMAKING AND RECOMMENDATION CORRIDOR ---
// This is the hybrid recommendation engine that fuses:
// 1. Physical features (Exponential decay of Euclidean distance between standardized weights/lengths/ages).
// 2. Collaborative Filtering (Jaccard overlaps of likes/skips).
// 3. AGNES Archetype matching (Behavioral profiles).
// 4. Apriori Association rules (Session-based preference mapping).
// It then passes the top candidates to a Random Forest multiplier to refine predictions.
async function getPlaydatesFeed(username) {
    let pets = await readCsv(PETS_CSV);
    const currentUser = pets.find(p => p.username === username);
    // Terminate query if the current user profile is invalid or flagged suspicious
    if (!currentUser || currentUser.isFlagged === 'true') return [];

    // Filter candidates list: exclude self and already flagged spam profiles
    let candidates = pets.filter(p => p.username !== username && p.isFlagged !== 'true');
    const interactions = await readCsv(INTERACTIONS_CSV);
    // Ignore profiles where the user has already liked or skipped them in the past
    const pastInteractions = new Set(interactions.filter(i => i.username === username).map(i => i.targetUsername));
    candidates = candidates.filter(c => !pastInteractions.has(c.username));

    // Resolve immediately if no eligible candidates are left
    if (candidates.length === 0) return [];

    // Map counts of user liked archetypes to identify behavioral class preference
    const archetypeCounts = {};
    const userLikes = interactions.filter(i => i.username === username && i.action === 'like');
    userLikes.forEach(like => {
        const archId = globalAgnesMap[like.targetUsername];
        if (archId !== undefined) archetypeCounts[archId] = (archetypeCounts[archId] || 0) + 1;
    });
    // Sort and retrieve the most frequently liked behavioral archetype index
    const preferredArchetype = Object.keys(archetypeCounts).sort((a, b) => archetypeCounts[b] - archetypeCounts[a])[0];

    // Compute active associations from the user's liked history using the Apriori rules index
    const activeAssociations = new Map();
    userLikes.forEach(like => {
        const targetPet = pets.find(p => p.username === like.targetUsername);
        if (targetPet) {
            const petProps = [];
            // Parse physical/behavioral properties of the liked pet
            if (targetPet.breed) petProps.push(`breed_${targetPet.breed}`);
            if (targetPet.type) petProps.push(`type_${targetPet.type.toLowerCase()}`);
            if (targetPet.gender) petProps.push(`gender_${targetPet.gender.toLowerCase()}`);

            const targetAge = new Date().getFullYear() - (parseInt(targetPet.birthYear) || 2020);
            petProps.push(`age_${targetAge < 2 ? 'young' : targetAge < 8 ? 'adult' : 'senior'}`);

            (targetPet.personality || '').toLowerCase().split(',').forEach(t => {
                const trait = t.trim();
                if (['active', 'friendly', 'calm', 'touchy', 'sleepy'].includes(trait)) petProps.push(`trait_${trait}`);
            });

            // Map and keep the maximum lift value for related attribute patterns
            petProps.forEach(prop => {
                if (globalAprioriRules[prop]) {
                    globalAprioriRules[prop].forEach((liftVal, assocItem) => {
                        const currentVal = activeAssociations.get(assocItem) || 0;
                        if (liftVal > currentVal) activeAssociations.set(assocItem, liftVal);
                    });
                }
            });
        }
    });

    const targetPhys = standardizePhysical([extractPhysical(currentUser)])[0];
    const targetBeh = extractBehavioral(currentUser);

    // Score each candidate against the hybrid weighting matrix
    candidates.forEach(c => {
        c._tempPhys = standardizePhysical([extractPhysical(c)])[0];
        // 1. Physical Layer: Euclidean distance with exponential decay mapping
        let physDistance = Math.sqrt(c._tempPhys.reduce((sum, val, idx) => sum + Math.pow(val - targetPhys[idx], 2), 0));
        let physScore = Math.exp(-physDistance * 0.5);

        // 2. Collaborative Layer: Aggregate similarity scores of users with similar swipe overlap
        let cfRaw = 0, cfInteractions = 0;
        const likers = globalItemLikers[c.username] || new Set();
        likers.forEach(liker => {
            if (globalUserSimMatrix[username] && globalUserSimMatrix[username][liker]) {
                cfRaw += globalUserSimMatrix[username][liker]; cfInteractions++;
            }
        });
        const skippers = globalItemSkippers[c.username] || new Set();
        skippers.forEach(skipper => {
            if (globalUserSimMatrix[username] && globalUserSimMatrix[username][skipper]) {
                cfRaw -= globalUserSimMatrix[username][skipper]; cfInteractions++;
            }
        });
        // Normalize collaborative filtering score between 0 and 1
        let cfScore = Math.max(0, Math.min(1, cfInteractions > 0 ? (cfRaw / cfInteractions + 1) / 2 : 0.5));

        // 3. Behavioral Layer: Archetype compliance (1.0 check), falling back to Hamming trait overlap
        let behScore = 0;
        if (preferredArchetype && globalAgnesMap[c.username] === parseInt(preferredArchetype)) {
            behScore = 1.0;
        } else {
            let overlaps = 0;
            const cBeh = extractBehavioral(c);
            for (let i = 0; i < 5; i++) if (cBeh[i] === 1 && targetBeh[i] === 1) overlaps++;
            behScore = overlaps / 5.0;
        }

        // 4. Trend Layer: Find and extract maximum Apriori association lift values
        let aprioriScore = 0.0;
        const cProps = [];
        if (c.breed) cProps.push(`breed_${c.breed}`);
        if (c.type) cProps.push(`type_${c.type.toLowerCase()}`);
        if (c.gender) cProps.push(`gender_${c.gender.toLowerCase()}`);

        const cAge = new Date().getFullYear() - (parseInt(c.birthYear) || 2020);
        cProps.push(`age_${cAge < 2 ? 'young' : cAge < 8 ? 'adult' : 'senior'}`);

        (c.personality || '').toLowerCase().split(',').forEach(t => {
            const trait = t.trim();
            if (['active', 'friendly', 'calm', 'touchy', 'sleepy'].includes(trait)) cProps.push(`trait_${trait}`);
        });

        cProps.forEach(prop => {
            if (activeAssociations.has(prop)) {
                const lift = activeAssociations.get(prop);
                // Map the rule lift value onto a normalized probability score bounds [0.0, 1.0]
                const score = Math.min((lift - 1.0) / 2.0, 1.0);
                if (score > aprioriScore) aprioriScore = score;
            }
        });

        // Compute the final unified linear combination score using configured multipliers
        c.fusionScore = (physScore * ML_CONFIG.fusionWeights.physical) +
            (cfScore * ML_CONFIG.fusionWeights.cf) +
            (behScore * ML_CONFIG.fusionWeights.behavioral) +
            (aprioriScore * ML_CONFIG.fusionWeights.apriori);
    });

    // Sort descending by initial linear scores and slice top 50 candidates
    candidates.sort((a, b) => b.fusionScore - a.fusionScore);
    candidates = candidates.slice(0, 50);

    // Apply late-stage Random Forest correction predictions if a model is successfully trained
    if (globalRfModel) {
        const inferenceData = candidates.map(c => {
            const diffVector = targetPhys.map((val, idx) => Math.abs(val - c._tempPhys[idx]));
            const isSameBreed = (currentUser.breed && c.breed && currentUser.breed === c.breed) ? 1 : 0;

            const cBeh = extractBehavioral(c);
            let behOverlap = 0;
            for (let i = 0; i < 5; i++) if (cBeh[i] === 1 && targetBeh[i] === 1) behOverlap++;

            diffVector.push(isSameBreed, behOverlap / 5.0);
            return diffVector;
        });

        // Predict labels across the entire top 50 cohort
        const basePredictions = globalRfModel.predict(inferenceData);

        candidates.forEach((c, i) => {
            let prob;
            // Aggregate precise probability distributions by calculating positive vote frequencies across all forest trees
            if (globalRfModel.estimators && globalRfModel.estimators.length > 0) {
                let positiveVotes = 0;
                globalRfModel.estimators.forEach(tree => { if (tree.predict([inferenceData[i]])[0] === 1) positiveVotes++; });
                prob = positiveVotes / globalRfModel.estimators.length;
            } else {
                prob = basePredictions[i];
            }

            // Calculate multiplier constraint ranging from 0.8 up to 1.2
            const multiplier = 0.8 + (prob * 0.4);
            c.fusionScore = Math.min(1.0, c.fusionScore * multiplier);
            c.matchScore = prob > 0.5 ? 'High' : 'Low';
        });
        // Resort the candidates by their adjusted prediction fusion scores
        candidates.sort((a, b) => b.fusionScore - a.fusionScore);
    } else {
        // Flag matching indicator as pending if the random forest has not completed its initial 20 swipe training
        candidates.forEach(c => c.matchScore = 'Pending Model');
    }

    // Clean up in-memory temporary variables to free up resources
    candidates.forEach(c => delete c._tempPhys);

    // Hard requirements gatekeeper:
    // 1. Same species only (dogs match dogs, cats match cats).
    // 2. Opposite gender only (for matchmaking/playdates safety).
    const myType   = (currentUser.type   || '').toLowerCase().trim();
    const myGender = (currentUser.gender || '').toLowerCase().trim();

    candidates = candidates.filter(c => {
        const cType   = (c.type   || '').toLowerCase().trim();
        const cGender = (c.gender || '').toLowerCase().trim();

        if (myType !== cType)     return false;   // different species → reject
        if (myGender === cGender) return false;   // same gender      → reject
        return true;
    });

    return candidates;
}

// Exposes current Agnes dendrogram statistics
function getAgnesTree() { return { tree: globalAgnesTreeCache, optimalK: globalOptimalK }; }
// Exposes active Apriori association rules mapping plain serialization formatting
function getAprioriRules() {
    return Object.fromEntries(Object.entries(globalAprioriRules).map(([k, v]) => [k, Array.from(v.entries())]));
}
// Exposes current K-Means elbow WCSS curve data
function getElbowData() { return { wcss: globalElbowWcss, optimalK: globalOptimalK }; }

// Module exports exposed to controllers and routing scripts
module.exports = {
    loadStateFromLocal,
    preprocess,
    gatekeeper,
    trainBackgroundModels,
    assignToCluster,
    runApriori,
    getPlaydatesFeed,
    getAgnesTree,
    getAprioriRules,
    getElbowData,
    readCsv,
    writeCsv
};