const fs = require('fs');
const path = require('path');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;
const csv = require('csv-parser');

const { kmeans } = require('ml-kmeans');
const hclust = require('ml-hclust');
const { RandomForestClassifier } = require('ml-random-forest');
const { Apriori } = require('node-apriori');

// Google Drive API Integration for Cloud-Native State Persistence
const { google } = require('googleapis');
const auth = new google.auth.GoogleAuth({
    scopes: ['https://www.googleapis.com/auth/drive.file']
});
const drive = google.drive({ version: 'v3', auth });

const DB_DIR = path.join(__dirname, 'DB');
const PETS_CSV = path.join(DB_DIR, 'individual_pets.csv');
const INTERACTIONS_CSV = path.join(DB_DIR, 'interactions.csv');
const MODEL_STATE_FILE_ID = process.env.DRIVE_STATE_FILE_ID || null; // Google Drive File ID

// --- HYPERPARAMETERS & CONFIGURATION ---
const ML_CONFIG = {
    featureMultipliers: [1.0, 1.0, 1.0, 3.0, 1.5, 1.2, 1.2, 1.2, 1.2, 1.2],
    fusionWeights: { physical: 0.35, behavioral: 0.20, cf: 0.30, apriori: 0.15 },
    jaccardMinThreshold: 0.1,
    aprioriMinSupport: 0.05,
    aprioriMinLift: 1.2, 
    sessionWindowMs: 3600000, 
    rfMaxFeatures: 3,
    rfEstimators: 50, 
    maxK: 10,
    chiSquareThreshold: 11.34 
};

// --- GLOBAL STATE ---
let globalRfModel = null;
let globalCentroids = null;
let globalScalingParams = { means: [0,0,0], stdDevs: [1,1,1] }; 
let globalOptimalK = 3; 

// Mahalanobis Distribution Stats { meanVector, invCovMatrix }
let globalClusterStats = {}; 

let globalAgnesMap = {}; 
let globalUserSimMatrix = {}; 
let globalItemLikers = {}; 
let globalItemSkippers = {}; 
let globalAprioriRules = {}; 

// --- CLOUD PERSISTENCE ---
async function syncToGoogleDrive() {
    try {
        const state = {
            centroids: globalCentroids,
            scalingParams: globalScalingParams,
            optimalK: globalOptimalK,
            clusterStats: globalClusterStats,
            agnesMap: globalAgnesMap,
            aprioriRules: Array.from(Object.entries(globalAprioriRules)) // Map to Array for JSON
        };
        
        const fileMetadata = { name: 'ml_pipeline_state.json' };
        const media = {
            mimeType: 'application/json',
            body: JSON.stringify(state)
        };

        if (MODEL_STATE_FILE_ID) {
            await drive.files.update({ fileId: MODEL_STATE_FILE_ID, media: media });
        } else {
            await drive.files.create({ requestBody: fileMetadata, media: media });
        }
        console.log("ML State synced to Google Drive.");
    } catch (err) {
        console.error("Cloud sync failed, falling back to next cycle:", err.message);
    }
}

// --- IO HELPERS ---
const readCsv = (filePath) => {
    return new Promise((resolve, reject) => {
        const results = [];
        if (!fs.existsSync(filePath)) return resolve(results);
        const readStream = fs.createReadStream(filePath);
        readStream.pipe(csv())
            .on('data', (data) => results.push(data))
            .on('error', (err) => { readStream.destroy(); reject(err); })
            .on('end', () => resolve(results));
    });
};

const writeCsv = (filePath, headers, records) => {
    const csvWriter = createCsvWriter({ path: filePath, header: headers });
    return csvWriter.writeRecords(records);
};

// --- PREPROCESSING & MAHALANOBIS GATEKEEPER ---
function preprocess(petData) {
    const processed = { ...petData };
    processed.birthYear = parseInt(processed.birthYear) || 2020;
    processed.weight = parseFloat(processed.weight) || 10.0;
    processed.length = parseFloat(processed.length) || 20.0;
    processed.vaccination = (processed.vaccination || '').toLowerCase();
    return processed;
}

function gatekeeper(rawPetData) {
    if (isNaN(parseFloat(rawPetData.weight))) return true;
    if (isNaN(parseInt(rawPetData.birthYear))) return true;
    if (isNaN(parseFloat(rawPetData.length))) return true;

    const age = new Date().getFullYear() - parseInt(rawPetData.birthYear);
    const type = (rawPetData.type || '').toLowerCase();
    const weight = parseFloat(rawPetData.weight);
    
    if (age > 25 || (type === 'cat' && weight > 40)) return true; 

    // TRUE MAHALANOBIS DISTANCE 
    if (globalCentroids && Object.keys(globalClusterStats).length > 0) {
        const physFeatures = extractPhysical(rawPetData);
        const scaledPhys = standardizePhysical([physFeatures])[0];

        let minDistance = Infinity;
        let closestCluster = -1;

        globalCentroids.forEach((centroid, index) => {
            const coords = Array.isArray(centroid) ? centroid : centroid.centroid;
            const dist = Math.sqrt(coords.reduce((sum, val, d) => sum + Math.pow(val - scaledPhys[d], 2), 0));
            if (dist < minDistance) { minDistance = dist; closestCluster = index; }
        });

        const stats = globalClusterStats[closestCluster];
        if (stats && stats.invCovMatrix) {
            const diff = scaledPhys.map((val, i) => val - stats.meanVector[i]);
            let mahalanobisSq = 0;
            
            // D^2 = (x - μ)ᵀ Σ⁻¹ (x - μ)
            for (let i = 0; i < 3; i++) {
                let temp = 0;
                for (let j = 0; j < 3; j++) {
                    temp += diff[j] * stats.invCovMatrix[j][i];
                }
                mahalanobisSq += temp * diff[i];
            }
            
            // Follows Chi-Square distribution for 3 degrees of freedom
            if (mahalanobisSq > ML_CONFIG.chiSquareThreshold) return true; 
        }
    } else {
        if (age > 25 || weight > 150) return true; // Cold start bounds
    }

    return false;
}

// --- FEATURE ENGINEERING ---
function extractPhysical(p) {
    const currentYear = new Date().getFullYear();
    return [parseFloat(p.weight) || 10, parseFloat(p.length) || 20, currentYear - (parseInt(p.birthYear) || 2020)]; 
}

function extractBehavioral(p) {
    const coreTraits = ['active', 'friendly', 'calm', 'touchy', 'sleepy'];
    const petTraits = (p.personality || '').toLowerCase().split(',').map(t => t.trim());
    return coreTraits.map(trait => petTraits.includes(trait) ? 1 : 0); 
}

function fitStandardizationParams(features) {
    if (features.length === 0) return;
    const means = [0, 0, 0];
    const stdDevs = [1, 1, 1];

    for(let idx = 0; idx < 3; idx++) {
        const vals = features.map(f => f[idx]);
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const variance = vals.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (vals.length > 1 ? vals.length - 1 : 1);
        means[idx] = mean;
        stdDevs[idx] = Math.sqrt(variance) || 0.001; 
    }
    globalScalingParams = { means, stdDevs };
}

function standardizePhysical(features) {
    return features.map(row => row.map((val, idx) => (val - globalScalingParams.means[idx]) / globalScalingParams.stdDevs[idx]));
}

// --- TRUE NORMALIZED KNEEDLE ---
function computeWCSS(data, clusters, centroids) {
    let wcss = 0;
    for (let i = 0; i < data.length; i++) {
        const clusterIdx = clusters[i];
        const coords = Array.isArray(centroids[clusterIdx]) ? centroids[clusterIdx] : centroids[clusterIdx].centroid;
        wcss += data[i].reduce((sum, val, d) => sum + Math.pow(val - coords[d], 2), 0);
    }
    return wcss;
}

function findOptimalK(data, maxK = 10) {
    const limit = Math.min(maxK, data.length - 1);
    if (limit <= 2) return Math.max(2, limit); 

    const wcssValues = [];
    for (let k = 1; k <= limit; k++) {
        const result = kmeans(data, k, { initialization: 'kmeans++' }); 
        wcssValues.push(computeWCSS(data, result.clusters, result.centroids));
    }

    // Normalize both axes to [0, 1] for accurate geometric distance
    const minWcss = Math.min(...wcssValues);
    const maxWcss = Math.max(...wcssValues);
    const normWcss = wcssValues.map(w => (maxWcss - minWcss === 0) ? 0 : (w - minWcss) / (maxWcss - minWcss));
    const normK = Array.from({length: limit}, (_, i) => i / (limit - 1));

    const A = normWcss[limit - 1] - normWcss[0];
    const B = normK[0] - normK[limit - 1];
    const C = normK[limit - 1] * normWcss[0] - normK[0] * normWcss[limit - 1];
    const denominator = Math.sqrt(A * A + B * B);

    let maxDistance = -1;
    let optimalK = 2; 

    for (let i = 0; i < limit; i++) {
        const perpDistance = Math.abs(A * normK[i] + B * normWcss[i] + C) / denominator;
        if (perpDistance > maxDistance) {
            maxDistance = perpDistance;
            optimalK = i + 1;
        }
    }
    return Math.max(2, optimalK); 
}

// 3x3 Matrix Inverse (Adjugate Method) with Ridge Regularization
function invert3x3Covariance(matrix, epsilon = 1e-4) {
    let m = matrix.map((row, i) => row.map((val, j) => val + (i === j ? epsilon : 0))); // Ridge
    const det = m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
    if (Math.abs(det) < 1e-8) return [[1,0,0],[0,1,0],[0,0,1]]; // Fallback to Euclidean if singular

    const invDet = 1.0 / det;
    return [
        [(m[1][1]*m[2][2] - m[2][1]*m[1][2]) * invDet, (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * invDet, (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * invDet],
        [(m[1][2]*m[2][0] - m[1][0]*m[2][2]) * invDet, (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * invDet, (m[1][0]*m[0][2] - m[0][0]*m[1][2]) * invDet],
        [(m[1][0]*m[2][1] - m[2][0]*m[1][1]) * invDet, (m[2][0]*m[0][1] - m[0][0]*m[2][1]) * invDet, (m[0][0]*m[1][1] - m[1][0]*m[0][1]) * invDet]
    ];
}

// --- BACKGROUND TRAINING TASKS ---
async function trainBackgroundModels() {
    let pets = await readCsv(PETS_CSV);
    const approvedPets = pets.filter(p => p.isFlagged !== 'true');
    const interactions = await readCsv(INTERACTIONS_CSV);

    if (approvedPets.length >= 3) {
        // 1. K-Means
        const rawPhysFeatures = approvedPets.map(extractPhysical);
        fitStandardizationParams(rawPhysFeatures); 
        const scaledPhysFeatures = standardizePhysical(rawPhysFeatures);
        
        globalOptimalK = findOptimalK(scaledPhysFeatures, ML_CONFIG.maxK);
        const kmeansResult = kmeans(scaledPhysFeatures, globalOptimalK, { initialization: 'kmeans++' });
        globalCentroids = kmeansResult.centroids;

        // Covariance Matrix Calculation for Mahalanobis
        globalClusterStats = {};
        for(let k = 0; k < globalOptimalK; k++) {
            const pts = scaledPhysFeatures.filter((_, i) => kmeansResult.clusters[i] === k);
            if(pts.length > 3) {
                const meanVector = [0,0,0];
                pts.forEach(p => { for(let d=0; d<3; d++) meanVector[d] += p[d]; });
                meanVector.forEach((_, d) => meanVector[d] /= pts.length);

                const covMatrix = [[0,0,0],[0,0,0],[0,0,0]];
                pts.forEach(p => {
                    for(let i=0; i<3; i++) {
                        for(let j=0; j<3; j++) {
                            covMatrix[i][j] += (p[i] - meanVector[i]) * (p[j] - meanVector[j]);
                        }
                    }
                });
                for(let i=0; i<3; i++) for(let j=0; j<3; j++) covMatrix[i][j] /= (pts.length - 1);
                
                globalClusterStats[k] = { meanVector, invCovMatrix: invert3x3Covariance(covMatrix) };
            }
        }

        // 2. AGNES (Complete Linkage on Binary Space with True Gap Detection)
        const behavioralFeatures = approvedPets.map(extractBehavioral);
        const tree = hclust.agnes(behavioralFeatures, { method: 'complete' }); // Superior for binary
        
        let maxGap = 0;
        let bestCutHeight = tree.height;
        
        function findLargestGap(node) {
            if (node.isLeaf) return;
            const childMaxHeight = Math.max(...node.children.map(c => c.height));
            const gap = node.height - childMaxHeight;
            if (gap > maxGap) { maxGap = gap; bestCutHeight = childMaxHeight + (gap / 2); }
            node.children.forEach(findLargestGap);
        }
        findLargestGap(tree);

        let nodes = [tree];
        while (nodes.some(n => n.height > bestCutHeight && !n.isLeaf)) {
            let splitIndex = nodes.findIndex(n => n.height > bestCutHeight && !n.isLeaf);
            let target = nodes.splice(splitIndex, 1)[0];
            nodes.push(...target.children);
        }
        
        const extractLeaves = (node, arr) => {
            if (node.isLeaf) arr.push(node.index);
            else node.children.forEach(c => extractLeaves(c, arr));
            return arr;
        };
        
        globalAgnesMap = {};
        nodes.forEach((clusterNode, archetypeId) => {
            extractLeaves(clusterNode, []).forEach(idx => {
                if (approvedPets[idx]) globalAgnesMap[approvedPets[idx].username] = archetypeId;
            });
        });
    }

    // 3. Jaccard CF
    const userLikes = {};
    const userSkips = {};
    const itemLikers = {}; 
    const itemSkippers = {}; 
    
    interactions.forEach(i => {
        if (!userLikes[i.username]) userLikes[i.username] = new Set();
        if (!userSkips[i.username]) userSkips[i.username] = new Set();

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

    // Build matrix for ALL active users, not just likers
    const users = Array.from(new Set([...Object.keys(userLikes), ...Object.keys(userSkips)]));
    const newSimMatrix = {};
    
    for (let i = 0; i < users.length; i++) {
        newSimMatrix[users[i]] = {};
        for (let j = i + 1; j < users.length; j++) {
            const u1 = users[i], u2 = users[j];
            const setA = userLikes[u1] || new Set(), setB = userLikes[u2] || new Set();
            
            let intersection = 0;
            for (let item of setA) if (setB.has(item)) intersection++;
            const union = setA.size + setB.size - intersection;
            const sim = union === 0 ? 0 : intersection / union;
            
            if (sim > ML_CONFIG.jaccardMinThreshold) { 
                newSimMatrix[u1][u2] = sim;
                if (!newSimMatrix[u2]) newSimMatrix[u2] = {};
                newSimMatrix[u2][u1] = sim;
            }
        }
    }
    globalUserSimMatrix = newSimMatrix;

    // 4. Random Forest (Survivorship Bias Fixed)
    if (interactions.length >= 20 && approvedPets.length > 0) {
        let allData = [];
        let allLabels = []; 

        interactions.forEach(interaction => {
            // Map against the raw CSV to prevent survivorship bias from purged pets
            const userPet = pets.find(p => p.username === interaction.username);
            const targetPet = pets.find(p => p.username === interaction.targetUsername);
            
            if (userPet && targetPet) {
                const f1 = standardizePhysical([extractPhysical(userPet)])[0];
                const f2 = standardizePhysical([extractPhysical(targetPet)])[0];
                const diffVector = f1.map((val, i) => Math.abs(val - f2[i]));
                const isSameBreed = (userPet.breed && targetPet.breed && userPet.breed === targetPet.breed) ? 1 : 0;
                diffVector.push(isSameBreed); 
                
                allData.push(diffVector);
                allLabels.push(interaction.action === 'like' ? 1 : 0);
            }
        });

        const likeIndices = allLabels.map((l, i) => l === 1 ? i : -1).filter(i => i !== -1);
        const skipIndices = allLabels.map((l, i) => l === 0 ? i : -1).filter(i => i !== -1);
        
        if (likeIndices.length > 0 && skipIndices.length > 0) {
            const minSize = Math.min(likeIndices.length, skipIndices.length);
            const balancedIndices = [
                ...likeIndices.sort(() => 0.5 - Math.random()).slice(0, minSize),
                ...skipIndices.sort(() => 0.5 - Math.random()).slice(0, minSize)
            ].sort(() => 0.5 - Math.random()); 

            const bData = balancedIndices.map(i => allData[i]);
            const bLabels = balancedIndices.map(i => allLabels[i]);

            const splitPoint = Math.floor(bData.length * 0.8);
            const trainData = bData.slice(0, splitPoint);
            const trainLabels = bLabels.slice(0, splitPoint);
            const testData = bData.slice(splitPoint);
            const testLabels = bLabels.slice(splitPoint);

            if (trainData.length > 0 && testData.length > 0) {
                const tempModel = new RandomForestClassifier({ 
                    seed: Math.floor(Math.random() * 1000), 
                    maxFeatures: ML_CONFIG.rfMaxFeatures, 
                    replacement: true, nEstimators: ML_CONFIG.rfEstimators 
                });
                tempModel.train(trainData, trainLabels);

                let correct = 0;
                const preds = tempModel.predict(testData);
                for(let i = 0; i < testLabels.length; i++) if (preds[i] === testLabels[i]) correct++;

                if ((correct / testLabels.length) > 0.55) {
                    globalRfModel = tempModel;
                }
            }
        }
    }

    syncToGoogleDrive(); // Push updated ML state to cloud
}

// 5. APRIORI
async function runApriori() {
    const interactions = await readCsv(INTERACTIONS_CSV);
    const pets = await readCsv(PETS_CSV);
    
    const sessionLikes = {};
    const itemFrequencies = {}; 
    
    interactions.filter(i => i.action === 'like').forEach(i => {
        const targetPet = pets.find(p => p.username === i.targetUsername);
        if (targetPet && targetPet.breed) {
            const sessionId = `${i.username}_${Math.floor(i.timestamp / ML_CONFIG.sessionWindowMs)}`;
            const breedKey = `breed_${targetPet.breed}`;
            if (!sessionLikes[sessionId]) sessionLikes[sessionId] = new Set();
            sessionLikes[sessionId].add(breedKey);
        }
    });

    const transactions = Object.values(sessionLikes).map(set => Array.from(set));
    const totalTx = transactions.length;
    if (totalTx < 5) return [];

    transactions.forEach(tx => {
        tx.forEach(item => { itemFrequencies[item] = (itemFrequencies[item] || 0) + 1; });
    });

    return new Promise((resolve) => {
        const apriori = new Apriori(ML_CONFIG.aprioriMinSupport); 
        apriori.exec(transactions).then(result => {
            const newRules = {};
            result.itemsets.forEach(itemset => {
                if (itemset.items.length === 2) {
                    const breedA = itemset.items[0], breedB = itemset.items[1];
                    const supportAB = itemset.support; 
                    const supportA = itemFrequencies[breedA] / totalTx;
                    const supportB = itemFrequencies[breedB] / totalTx;
                    const lift = supportAB / (supportA * supportB);
                    
                    if (lift > ML_CONFIG.aprioriMinLift) {
                        if (!newRules[breedA]) newRules[breedA] = new Map();
                        if (!newRules[breedB]) newRules[breedB] = new Map();
                        newRules[breedA].set(breedB, lift);
                        newRules[breedB].set(breedA, lift);
                    }
                }
            });
            globalAprioriRules = newRules;
            resolve(result.itemsets);
        });
    });
}

function assignToCluster(newPetData) {
    if (!globalCentroids) return 0; 
    const scaledPhys = standardizePhysical([extractPhysical(newPetData)])[0];
    let minDistance = Infinity;
    let closestCluster = 0;

    globalCentroids.forEach((centroid, index) => {
        const coords = Array.isArray(centroid) ? centroid : centroid.centroid;
        const dist = Math.sqrt(coords.reduce((sum, val, i) => sum + Math.pow(val - scaledPhys[i], 2), 0));
        if (dist < minDistance) { minDistance = dist; closestCluster = index; }
    });
    return closestCluster;
}

// --- RECOMMENDATION ENGINE ---
async function getPlaydatesFeed(username) {
    let pets = await readCsv(PETS_CSV);
    const currentUser = pets.find(p => p.username === username);
    if (!currentUser || currentUser.isFlagged === 'true') return [];

    let candidates = pets.filter(p => p.username !== username && p.isFlagged !== 'true');
    if (candidates.length === 0) return [];

    const interactions = await readCsv(INTERACTIONS_CSV);
    const userLikes = interactions.filter(i => i.username === username && i.action === 'like');
    
    const pastInteractions = new Set(interactions.filter(i => i.username === username).map(i => i.targetUsername));
    candidates = candidates.filter(c => !pastInteractions.has(c.username));
    if (candidates.length === 0) return [];

    const archetypeCounts = {};
    userLikes.forEach(like => {
        const archId = globalAgnesMap[like.targetUsername];
        if (archId !== undefined) archetypeCounts[archId] = (archetypeCounts[archId] || 0) + 1;
    });
    const preferredArchetype = Object.keys(archetypeCounts).sort((a, b) => archetypeCounts[b] - archetypeCounts[a])[0];

    const userBreedAssociations = new Map(); 
    userLikes.forEach(like => {
        const targetPet = pets.find(p => p.username === like.targetUsername);
        if (targetPet && targetPet.breed && globalAprioriRules[`breed_${targetPet.breed}`]) {
            globalAprioriRules[`breed_${targetPet.breed}`].forEach((liftVal, associatedBreed) => {
                const currentVal = userBreedAssociations.get(associatedBreed) || 0;
                if (liftVal > currentVal) userBreedAssociations.set(associatedBreed, liftVal);
            });
        }
    });

    const targetPhys = standardizePhysical([extractPhysical(currentUser)])[0];
    const targetBeh = extractBehavioral(currentUser);

    candidates.forEach(c => {
        const cPhys = standardizePhysical([extractPhysical(c)])[0];
        const cBeh = extractBehavioral(c);
        c._tempPhys = cPhys; 
        
        let physDistance = Math.sqrt(cPhys.reduce((sum, val, idx) => sum + Math.pow(val - targetPhys[idx], 2), 0));
        let physScore = Math.exp(-physDistance * 0.5); 

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
        
        let rawAvg = cfInteractions > 0 ? (cfRaw / cfInteractions) : 0;
        let cfScore = Math.max(0, Math.min(1, (rawAvg + 1) / 2)); // Strictly Clamped

        let behScore = 0;
        if (preferredArchetype && globalAgnesMap[c.username] === parseInt(preferredArchetype)) {
            behScore = 1.0;
        } else {
            let overlaps = 0;
            for(let i=0; i<5; i++) if (cBeh[i] === 1 && targetBeh[i] === 1) overlaps++;
            behScore = overlaps / 5;
        }

        let aprioriScore = 0.0;
        if (c.breed && userBreedAssociations.has(`breed_${c.breed}`)) {
            const lift = userBreedAssociations.get(`breed_${c.breed}`);
            aprioriScore = Math.min((lift - 1.0) / 2.0, 1.0); 
        }

        c.fusionScore = (physScore * ML_CONFIG.fusionWeights.physical) + 
                        (cfScore * ML_CONFIG.fusionWeights.cf) + 
                        (behScore * ML_CONFIG.fusionWeights.behavioral) + 
                        (aprioriScore * ML_CONFIG.fusionWeights.apriori);
    });
    
    candidates.sort((a, b) => b.fusionScore - a.fusionScore); 
    candidates = candidates.slice(0, 50); 

    if (globalRfModel) {
        const inferenceData = candidates.map(c => {
            const diffVector = targetPhys.map((val, idx) => Math.abs(val - c._tempPhys[idx]));
            const isSameBreed = (currentUser.breed && c.breed && currentUser.breed === c.breed) ? 1 : 0;
            diffVector.push(isSameBreed);
            return diffVector;
        });
        
        const predictions = globalRfModel.predict(inferenceData);
        
        candidates.forEach((c, i) => {
            c.fusionScore *= predictions[i] === 1 ? 1.2 : 0.8;
            c.matchScore = predictions[i] === 1 ? 'High' : 'Low'; 
        });
        candidates.sort((a, b) => b.fusionScore - a.fusionScore);
    } else {
        candidates.forEach(c => c.matchScore = 'Pending Model');
    }

    candidates.forEach(c => delete c._tempPhys);
    return candidates;
}

module.exports = {
    preprocess,
    gatekeeper,
    trainBackgroundModels,
    assignToCluster,
    runApriori,
    getPlaydatesFeed,
    readCsv,
    writeCsv
};