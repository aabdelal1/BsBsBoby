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

// --- GLOBAL STATE & MODELS ---
let globalRfModel = null;
let globalCentroids = null;
let globalScalingParams = { min: [0,0,0,0,0], max: [100,100,25,1,1] }; // [weight, length, age, isDog, isMale]

// Background Caches to ensure O(1) API lookups
let globalAgnesMap = {}; // { username: clusterId }
let globalAgnesTreeCache = null; 
let globalUserSimMatrix = {}; // { userA: { userB: 0.85 } }
let globalAprioriAssociations = {}; // { 'breed_A': Set(['breed_B']) }

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

    const age = new Date().getFullYear() - parseInt(rawPetData.birthYear);
    const type = (rawPetData.type || '').toLowerCase();
    
    // Improved Species-Aware Anomaly Detection
    if (age > 25) return true;
    if (type === 'cat' && parseFloat(rawPetData.weight) > 40) return true; 
    if (age <= 1 && parseFloat(rawPetData.weight) > 100) return true;

    const spamKeywords = ['spam', 'fake', 'none', 'test'];
    if (spamKeywords.includes((rawPetData.vaccination || '').toLowerCase())) return true;

    return false;
}

// --- 5D FEATURE ENGINEERING ---
function extractFeatures(pets) {
    const currentYear = new Date().getFullYear();
    return pets.map(p => {
        const isDog = (p.type || '').toLowerCase() === 'dog' ? 1 : 0;
        const isMale = (p.gender || '').toLowerCase() === 'male' ? 1 : 0;
        return [
            parseFloat(p.weight) || 10,
            parseFloat(p.length) || 20,
            currentYear - (parseInt(p.birthYear) || 2020),
            isDog,
            isMale
        ];
    });
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

// --- BACKGROUND TRAINING TASKS ---
async function trainBackgroundModels() {
    let pets = await readCsv(PETS_CSV);
    const approvedPets = pets.filter(p => p.isFlagged !== 'true');
    const interactions = await readCsv(INTERACTIONS_CSV);

    if (approvedPets.length >= 3) {
        const rawFeatures = extractFeatures(approvedPets);
        updateScalingParams(rawFeatures); 
        const scaledFeatures = normalizeFeatures(rawFeatures);
        
        // 1. K-Means
        const k = Math.min(5, Math.floor(approvedPets.length / 2)); 
        const kmeansResult = kmeans(scaledFeatures, Math.max(2, k));
        globalCentroids = kmeansResult.centroids;

        // 2. AGNES (Dendrogram Cutting for Archetypes)
        const tree = hclust.agnes(scaledFeatures, { method: 'ward' });
        globalAgnesTreeCache = tree; // Cache for admin dashboard
        
        // Cut tree into k macro-clusters
        let nodes = [tree];
        while (nodes.length < k && !nodes[0].isLeaf) {
            nodes.sort((a, b) => b.height - a.height);
            let highest = nodes.shift();
            nodes.push(...highest.children);
        }
        
        const extractLeaves = (node, arr) => {
            if (node.isLeaf) arr.push(node.index);
            else node.children.forEach(c => extractLeaves(c, arr));
            return arr;
        };
        
        globalAgnesMap = {};
        nodes.forEach((clusterNode, archetypeId) => {
            let originalIndices = extractLeaves(clusterNode, []);
            originalIndices.forEach(idx => {
                globalAgnesMap[approvedPets[idx].username] = archetypeId;
            });
        });
    }

    // 3. Jaccard User-to-User Similarity Matrix
    const userLikes = {};
    interactions.filter(i => i.action === 'like').forEach(i => {
        if (!userLikes[i.username]) userLikes[i.username] = new Set();
        userLikes[i.username].add(i.targetUsername);
    });

    const users = Object.keys(userLikes);
    const newSimMatrix = {};
    
    for (let i = 0; i < users.length; i++) {
        newSimMatrix[users[i]] = {};
        for (let j = i + 1; j < users.length; j++) {
            const u1 = users[i], u2 = users[j];
            const setA = userLikes[u1], setB = userLikes[u2];
            
            let intersection = 0;
            for (let item of setA) if (setB.has(item)) intersection++;
            
            const union = setA.size + setB.size - intersection;
            const sim = union === 0 ? 0 : intersection / union;
            
            if (sim > 0.1) { // Threshold to prevent noise
                newSimMatrix[u1][u2] = sim;
                if (!newSimMatrix[u2]) newSimMatrix[u2] = {};
                newSimMatrix[u2][u1] = sim;
            }
        }
    }
    globalUserSimMatrix = newSimMatrix;

    // 4. Random Forest Inference
    if (interactions.length >= 10 && approvedPets.length > 0) {
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
            const options = { seed: 3, maxFeatures: 3, replacement: true, nEstimators: 25 };
            globalRfModel = new RandomForestClassifier(options);
            globalRfModel.train(trainingData, trainingLabels);
        }
    }
}

// 5. APRIORI (Extracting applied associations)
async function runApriori() {
    const interactions = await readCsv(INTERACTIONS_CSV);
    const pets = await readCsv(PETS_CSV);
    
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
        const apriori = new Apriori(0.10); // Lowered threshold to 10% to catch more niche breeds
        apriori.exec(transactions).then(result => {
            const newAssociations = {};
            result.itemsets.forEach(itemset => {
                if (itemset.items.length > 1) {
                    itemset.items.forEach(breed1 => {
                        if (!newAssociations[breed1]) newAssociations[breed1] = new Set();
                        itemset.items.forEach(breed2 => {
                            if (breed1 !== breed2) newAssociations[breed1].add(breed2);
                        });
                    });
                }
            });
            globalAprioriAssociations = newAssociations;
            resolve(result.itemsets);
        });
    });
}

function assignToCluster(newPetData) {
    if (!globalCentroids) return 0; 
    const scaledFeat = normalizeFeatures(extractFeatures([newPetData]))[0];
    let minDistance = Infinity;
    let closestCluster = 0;

    globalCentroids.forEach((centroid, index) => {
        const coords = Array.isArray(centroid) ? centroid : centroid.centroid;
        const dist = Math.sqrt(coords.reduce((sum, val, i) => sum + Math.pow(val - scaledFeat[i], 2), 0));
        if (dist < minDistance) { minDistance = dist; closestCluster = index; }
    });
    return closestCluster;
}

// --- RECOMMENDATION ENGINE (O(1) Matrix Synergy + 5D Hybrid) ---
async function getPlaydatesFeed(username) {
    let pets = await readCsv(PETS_CSV);
    const currentUser = pets.find(p => p.username === username);
    if (!currentUser || currentUser.isFlagged === 'true') return [];

    let candidates = pets.filter(p => p.username !== username && p.isFlagged !== 'true');
    if (candidates.length === 0) return [];

    const interactions = await readCsv(INTERACTIONS_CSV);
    const userLikes = interactions.filter(i => i.username === username && i.action === 'like');
    
    // 1. Filter out previously seen pets
    const pastInteractions = new Set(interactions.filter(i => i.username === username).map(i => i.targetUsername));
    candidates = candidates.filter(c => !pastInteractions.has(c.username));
    if (candidates.length === 0) return [];

    // Establish User's "Archetype" from AGNES history
    const archetypeCounts = {};
    userLikes.forEach(like => {
        const archId = globalAgnesMap[like.targetUsername];
        if (archId !== undefined) archetypeCounts[archId] = (archetypeCounts[archId] || 0) + 1;
    });
    const preferredArchetype = Object.keys(archetypeCounts).sort((a, b) => archetypeCounts[b] - archetypeCounts[a])[0];

    // Establish User's "Apriori" active associations
    const userBreedAssociations = new Set();
    userLikes.forEach(like => {
        const targetPet = pets.find(p => p.username === like.targetUsername);
        if (targetPet && targetPet.breed && globalAprioriAssociations[`breed_${targetPet.breed}`]) {
            globalAprioriAssociations[`breed_${targetPet.breed}`].forEach(b => userBreedAssociations.add(b));
        }
    });

    const scaledData = normalizeFeatures(extractFeatures(candidates));
    const targetScaled = normalizeFeatures(extractFeatures([currentUser]))[0];

    candidates.forEach((c, i) => {
        const f = scaledData[i];
        
        // Base 5D KNN Distance
        let distance = Math.sqrt(f.reduce((sum, val, idx) => sum + Math.pow(val - targetScaled[idx], 2), 0));

        // CF Boost (Lookup Pre-computed Matrix)
        let cfBoost = 0;
        const likersOfCandidate = interactions.filter(int => int.targetUsername === c.username && int.action === 'like').map(int => int.username);
        likersOfCandidate.forEach(liker => {
            if (globalUserSimMatrix[username] && globalUserSimMatrix[username][liker]) {
                cfBoost += globalUserSimMatrix[username][liker]; // Add established synergy
            }
        });

        // Archetype Boost (AGNES)
        let agnesBoost = 0;
        if (preferredArchetype && globalAgnesMap[c.username] === parseInt(preferredArchetype)) {
            agnesBoost = 0.3; // High affinity for this demographic
        }

        // Association Boost (Apriori)
        let aprioriBoost = 0;
        if (c.breed && userBreedAssociations.has(`breed_${c.breed}`)) {
            aprioriBoost = 0.25; // Data mining dictates they usually like this pairing
        }

        // Apply Hybrid Reductions (Lower distance is better)
        c.hybridDistance = distance - (cfBoost * 0.4) - agnesBoost - aprioriBoost; 
    });
    
    candidates.sort((a, b) => a.hybridDistance - b.hybridDistance);
    candidates = candidates.slice(0, 50); 

    // RF Predictive Filter
    if (globalRfModel) {
        const inferenceData = candidates.map((c, i) => {
            const f1 = targetScaled;
            const f2 = scaledData[i]; // Needs to map back to the slice, but scaledData is from original array
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

// O(1) Dashboard retrieval
function getAgnesTree() { return globalAgnesTreeCache; }

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