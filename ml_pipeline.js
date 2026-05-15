const fs = require('fs');
const path = require('path');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;
const csv = require('csv-parser');

const { kmeans } = require('ml-kmeans');
const hclust = require('ml-hclust'); 
const { RandomForestClassifier } = require('ml-random-forest');
const { Apriori } = require('node-apriori');

const DB_DIR = path.join(__dirname, 'DB');
const PETS_CSV = path.join(DB_DIR, 'individual_pets.csv');
const INTERACTIONS_CSV = path.join(DB_DIR, 'interactions.csv');

// --- GLOBAL MODEL STATE ---
let globalRfModel = null;
let globalCentroids = null;
let globalScalingParams = { min: [0, 0, 0], max: [100, 100, 25] }; // [weight, length, age]

// --- IO HELPERS ---

const readCsv = (filePath) => {
    return new Promise((resolve, reject) => {
        const results = [];
        if (!fs.existsSync(filePath)) {
            return resolve(results);
        }
        const readStream = fs.createReadStream(filePath);
        readStream
            .pipe(csv())
            .on('data', (data) => results.push(data))
            .on('error', (err) => {
                readStream.destroy();
                reject(err);
            })
            .on('end', () => resolve(results));
    });
};

const writeCsv = (filePath, headers, records) => {
    const csvWriter = createCsvWriter({ path: filePath, header: headers });
    return csvWriter.writeRecords(records);
};

// --- PREPROCESSING & GATEKEEPER ---

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

    const currentYear = new Date().getFullYear();
    const age = currentYear - parseInt(rawPetData.birthYear);
    
    // Heuristic Anomaly Detection
    if (age <= 1 && parseFloat(rawPetData.weight) > 80) return true;
    if (age > 25) return true;

    const spamKeywords = ['spam', 'fake', 'none', 'no', 'test'];
    if (spamKeywords.includes((rawPetData.vaccination || '').toLowerCase())) return true;

    return false;
}

// --- FEATURE ENGINEERING ---

function extractFeatures(pets) {
    const currentYear = new Date().getFullYear();
    return pets.map(p => [
        parseFloat(p.weight) || 10,
        parseFloat(p.length) || 20,
        currentYear - (parseInt(p.birthYear) || 2020)
    ]);
}

function updateScalingParams(features) {
    if (features.length === 0) return;
    const numFeatures = features[0].length;
    globalScalingParams.min = Array(numFeatures).fill(Infinity);
    globalScalingParams.max = Array(numFeatures).fill(-Infinity);

    features.forEach(row => {
        row.forEach((val, i) => {
            if (val < globalScalingParams.min[i]) globalScalingParams.min[i] = val;
            if (val > globalScalingParams.max[i]) globalScalingParams.max[i] = val;
        });
    });
}

function normalizeFeatures(features) {
    return features.map(row => 
        row.map((val, i) => {
            const min = globalScalingParams.min[i];
            const max = globalScalingParams.max[i];
            return max - min === 0 ? 0 : (val - min) / (max - min);
        })
    );
}

// --- BACKGROUND TRAINING ---

async function trainBackgroundModels() {
    let pets = await readCsv(PETS_CSV);
    const approvedPets = pets.filter(p => p.isFlagged !== 'true');
    const interactions = await readCsv(INTERACTIONS_CSV);

    if (approvedPets.length >= 3) {
        // 1. Train K-Means
        const rawFeatures = extractFeatures(approvedPets);
        updateScalingParams(rawFeatures); 
        const scaledFeatures = normalizeFeatures(rawFeatures);
        
        const kmeansResult = kmeans(scaledFeatures, 3);
        globalCentroids = kmeansResult.centroids;
    }

    if (interactions.length >= 10 && approvedPets.length > 0) {
        // 2. Train Random Forest (Target: 1 for Like, 0 for Skip)
        const trainingData = [];
        const trainingLabels = []; 

        interactions.forEach(interaction => {
            const userPet = approvedPets.find(p => p.username === interaction.username);
            const targetPet = approvedPets.find(p => p.username === interaction.targetUsername);
            
            if (userPet && targetPet) {
                const f1 = normalizeFeatures(extractFeatures([userPet]))[0];
                const f2 = normalizeFeatures(extractFeatures([targetPet]))[0];
                const diffVector = f1.map((val, i) => Math.abs(val - f2[i]));
                
                trainingData.push(diffVector);
                trainingLabels.push(interaction.action === 'like' ? 1 : 0);
            }
        });

        const uniqueLabels = new Set(trainingLabels);
        if (trainingData.length > 0 && uniqueLabels.size > 1) {
            const options = { seed: 3, maxFeatures: 2, replacement: true, nEstimators: 25 };
            globalRfModel = new RandomForestClassifier(options);
            globalRfModel.train(trainingData, trainingLabels);
        }
    }
}

// O(k) Non-blocking cluster assignment for new/approved profiles
function assignToCluster(newPetData) {
    if (!globalCentroids) return 0; 
    
    const rawFeat = extractFeatures([newPetData])[0];
    const scaledFeat = normalizeFeatures([rawFeat])[0];

    let minDistance = Infinity;
    let closestCluster = 0;

    globalCentroids.forEach((centroid, index) => {
        // Handle varying returns from different versions of ml-kmeans
        const coords = Array.isArray(centroid) ? centroid : centroid.centroid;
        
        const dist = Math.sqrt(coords.reduce((sum, val, i) => sum + Math.pow(val - scaledFeat[i], 2), 0));
        if (dist < minDistance) {
            minDistance = dist;
            closestCluster = index;
        }
    });

    return closestCluster;
}

// --- APRIORI ---

async function runApriori() {
    const interactions = await readCsv(INTERACTIONS_CSV);
    const pets = await readCsv(PETS_CSV);
    
    // Group likes by user to find breed associations
    const userLikes = {};
    interactions.filter(i => i.action === 'like').forEach(i => {
        const targetPet = pets.find(p => p.username === i.targetUsername);
        if (targetPet && targetPet.breed) {
            if (!userLikes[i.username]) userLikes[i.username] = new Set();
            userLikes[i.username].add(`breed_${targetPet.breed}`);
        }
    });

    const transactions = Object.values(userLikes).map(set => Array.from(set));
    if (transactions.length < 5) return [];

    return new Promise((resolve) => {
        const apriori = new Apriori(0.15); // 15% support threshold
        apriori.exec(transactions).then(result => resolve(result.itemsets));
    });
}

// --- RECOMMENDATION ENGINE ---

async function getPlaydatesFeed(username) {
    let pets = await readCsv(PETS_CSV);
    const currentUser = pets.find(p => p.username === username);
    if (!currentUser || currentUser.isFlagged === 'true') return [];

    let candidates = pets.filter(p => p.username !== username && p.isFlagged !== 'true');
    if (candidates.length === 0) return [];

    const interactions = await readCsv(INTERACTIONS_CSV);

    // 1. STATE MANAGEMENT: Filter out previously interacted pets
    const pastInteractions = new Set(
        interactions.filter(i => i.username === username).map(i => i.targetUsername)
    );
    candidates = candidates.filter(c => !pastInteractions.has(c.username));
    if (candidates.length === 0) return [];

    // 2. COLLABORATIVE FILTERING: Jaccard Similarity
    const userLikes = {};
    interactions.filter(i => i.action === 'like').forEach(i => {
        if (!userLikes[i.username]) userLikes[i.username] = new Set();
        userLikes[i.username].add(i.targetUsername);
    });

    const targetUserLikes = userLikes[username] || new Set();

    const calculateJaccard = (setA, setB) => {
        if (setA.size === 0 && setB.size === 0) return 0;
        const intersection = new Set([...setA].filter(x => setB.has(x)));
        const union = new Set([...setA, ...setB]);
        return intersection.size / union.size;
    };

    candidates.forEach(candidate => {
        let cfBoost = 0;
        Object.keys(userLikes).forEach(otherUser => {
            if (otherUser !== username && userLikes[otherUser].has(candidate.username)) {
                const similarity = calculateJaccard(targetUserLikes, userLikes[otherUser]);
                cfBoost += similarity; 
            }
        });
        candidate.cfScore = cfBoost; 
    });

    // 3. CONTENT-BASED FILTERING: Normalized KNN
    const rawData = extractFeatures(candidates);
    const scaledData = normalizeFeatures(rawData);
    const targetScaled = normalizeFeatures(extractFeatures([currentUser]))[0];

    candidates.forEach((c, i) => {
        const f = scaledData[i];
        let distance = Math.sqrt(
            Math.pow(f[0] - targetScaled[0], 2) +
            Math.pow(f[1] - targetScaled[1], 2) +
            Math.pow(f[2] - targetScaled[2], 2)
        );
        
        // Hybrid Score: Shorter spatial distance is better, high CF score brings it closer
        c.hybridDistance = distance - (c.cfScore * 0.5); 
    });
    
    candidates.sort((a, b) => a.hybridDistance - b.hybridDistance);
    candidates = candidates.slice(0, 50); 

    // 4. PREDICTIVE INFERENCE: Random Forest
    if (globalRfModel) {
        const inferenceData = candidates.map((c, i) => {
            const f1 = targetScaled;
            const f2 = scaledData[i];
            return f1.map((val, idx) => Math.abs(val - f2[idx]));
        });
        
        const predictions = globalRfModel.predict(inferenceData);
        
        candidates.forEach((c, i) => {
            c.matchScore = predictions[i] === 1 ? 'High' : 'Medium'; 
        });
    } else {
        candidates.forEach(c => c.matchScore = 'Pending Model');
    }

    return candidates;
}

// Generate an AGNES tree purely for the admin dashboard visualization
async function getAgnesTree() {
    let pets = await readCsv(PETS_CSV);
    const approvedPets = pets.filter(p => p.isFlagged !== 'true');
    if (approvedPets.length < 3) return null;
    
    const scaledData = normalizeFeatures(extractFeatures(approvedPets));
    return hclust.agnes(scaledData, { method: 'ward' });
}

module.exports = {
    preprocess,
    gatekeeper,
    trainBackgroundModels,
    assignToCluster,
    runApriori,
    getPlaydatesFeed,
    getAgnesTree,
    readCsv,
    writeCsv
};