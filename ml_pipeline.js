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

// --- HYPERPARAMETERS & CONFIGURATION ---
const ML_CONFIG = {
    // Distance Weights: [Weight, Length, Age, isDog, isMale]
    featureWeights: [0.5, 0.5, 0.5, 4.0, 1.5],
    
    // Engine Boost Multipliers
    cfBoostWeight: 0.4,
    agnesArchetypeBoost: 0.3,
    aprioriConfidenceMultiplier: 1.5,
    
    // Background Mining Thresholds
    jaccardMinThreshold: 0.1,
    aprioriSupportThreshold: 0.10,

    // Random Forest Settings
    rfMaxFeatures: 3,
    rfEstimators: 25
};

// --- GLOBAL STATE & MODELS ---
let globalRfModel = null;
let globalCentroids = null;
let globalScalingParams = { min: [0,0,0,0,0], max: [100,100,25,1,1] }; 

// Background Caches (O(1) Lookups)
let globalAgnesMap = {}; 
let globalAgnesTreeCache = null; 
let globalUserSimMatrix = {}; 
let globalItemLikers = {}; 
let globalAprioriAssociations = {}; 

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
    const weight = parseFloat(rawPetData.weight);
    
    if (age > 25) return true;
    if (type === 'cat' && weight > 40) return true; 
    if (age <= 1 && weight > 100 && type !== 'dog') return true; 

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

        // 2. AGNES
        const tree = hclust.agnes(scaledFeatures, { method: 'ward' });
        globalAgnesTreeCache = tree; 
        
        let nodes = [tree];
        while (nodes.length < k && nodes.some(n => !n.isLeaf)) {
            nodes.sort((a, b) => b.height - a.height);
            let splitIndex = nodes.findIndex(n => !n.isLeaf);
            let highest = nodes.splice(splitIndex, 1)[0];
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

    // 3. Jaccard & Reverse Liker Cache (O(1) Lookups)
    const userLikes = {};
    const itemLikers = {}; 
    
    interactions.filter(i => i.action === 'like').forEach(i => {
        if (!userLikes[i.username]) userLikes[i.username] = new Set();
        userLikes[i.username].add(i.targetUsername);
        
        if (!itemLikers[i.targetUsername]) itemLikers[i.targetUsername] = new Set();
        itemLikers[i.targetUsername].add(i.username);
    });
    
    globalItemLikers = itemLikers;

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
            
            if (sim > ML_CONFIG.jaccardMinThreshold) { 
                newSimMatrix[u1][u2] = sim;
                if (!newSimMatrix[u2]) newSimMatrix[u2] = {};
                newSimMatrix[u2][u1] = sim;
            }
        }
    }
    globalUserSimMatrix = newSimMatrix;

    // 4. Random Forest (Now utilizing Categorical Breed & explicit Age Rules)
    if (interactions.length >= 10 && approvedPets.length > 0) {
        const trainingData = [];
        const trainingLabels = []; 

        interactions.forEach(interaction => {
            const userPet = approvedPets.find(p => p.username === interaction.username);
            const targetPet = approvedPets.find(p => p.username === interaction.targetUsername);
            
            if (userPet && targetPet) {
                const f1 = normalizeFeatures(extractFeatures([userPet]))[0];
                const f2 = normalizeFeatures(extractFeatures([targetPet]))[0];
                
                // Base 5D Euclidean Difference
                const diffVector = f1.map((val, i) => Math.abs(val - f2[i]));
                
                // Append explicit categorical Rules for the Decision Trees
                const isSameBreed = (userPet.breed && targetPet.breed && userPet.breed === targetPet.breed) ? 1 : 0;
                const ageDiff = Math.abs((parseInt(userPet.birthYear) || 2020) - (parseInt(targetPet.birthYear) || 2020));
                
                diffVector.push(isSameBreed, ageDiff); // Vector size is now 7
                
                trainingData.push(diffVector);
                trainingLabels.push(interaction.action === 'like' ? 1 : 0);
            }
        });

        const uniqueLabels = new Set(trainingLabels);
        if (trainingData.length > 0 && uniqueLabels.size > 1) {
            const options = { 
                seed: 3, 
                maxFeatures: ML_CONFIG.rfMaxFeatures, 
                replacement: true, 
                nEstimators: ML_CONFIG.rfEstimators 
            };
            globalRfModel = new RandomForestClassifier(options);
            globalRfModel.train(trainingData, trainingLabels);
        }
    }
}

// 5. APRIORI
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
        const apriori = new Apriori(ML_CONFIG.aprioriSupportThreshold); 
        apriori.exec(transactions).then(result => {
            const newAssociations = {};
            result.itemsets.forEach(itemset => {
                if (itemset.items.length > 1) {
                    itemset.items.forEach(breed1 => {
                        if (!newAssociations[breed1]) newAssociations[breed1] = new Map();
                        itemset.items.forEach(breed2 => {
                            if (breed1 !== breed2) newAssociations[breed1].set(breed2, itemset.support);
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

// --- RECOMMENDATION ENGINE ---
async function getPlaydatesFeed(username) {
    let pets = await readCsv(PETS_CSV);
    const currentUser = pets.find(p => p.username === username);
    if (!currentUser || currentUser.isFlagged === 'true') return [];

    let candidates = pets.filter(p => p.username !== username && p.isFlagged !== 'true');
    if (candidates.length === 0) return [];

    const interactions = await readCsv(INTERACTIONS_CSV);
    const userLikes = interactions.filter(i => i.username === username && i.action === 'like');
    
    // 1. Seen Filter
    const pastInteractions = new Set(interactions.filter(i => i.username === username).map(i => i.targetUsername));
    candidates = candidates.filter(c => !pastInteractions.has(c.username));
    
    // 2. K-Means Cluster Partitioning (Vector Search Optimization)
    // Limits the O(N) KNN search space to the user's localized node.
    const userCluster = currentUser.clusterGroup;
    if (userCluster !== undefined && userCluster !== '') {
        const inClusterCandidates = candidates.filter(c => String(c.clusterGroup) === String(userCluster));
        // Fallback: Only restrict if the localized cluster actually contains enough potential matches
        if (inClusterCandidates.length >= 5) {
            candidates = inClusterCandidates;
        }
    }

    if (candidates.length === 0) return [];

    // AGNES Archetype Processing
    const archetypeCounts = {};
    userLikes.forEach(like => {
        const archId = globalAgnesMap[like.targetUsername];
        if (archId !== undefined) archetypeCounts[archId] = (archetypeCounts[archId] || 0) + 1;
    });
    const preferredArchetype = Object.keys(archetypeCounts).sort((a, b) => archetypeCounts[b] - archetypeCounts[a])[0];

    // Apriori Active Associations 
    const userBreedAssociations = new Map(); 
    userLikes.forEach(like => {
        const targetPet = pets.find(p => p.username === like.targetUsername);
        if (targetPet && targetPet.breed && globalAprioriAssociations[`breed_${targetPet.breed}`]) {
            const associationsMap = globalAprioriAssociations[`breed_${targetPet.breed}`];
            associationsMap.forEach((supportVal, associatedBreed) => {
                const currentVal = userBreedAssociations.get(associatedBreed) || 0;
                if (supportVal > currentVal) userBreedAssociations.set(associatedBreed, supportVal);
            });
        }
    });

    const scaledData = normalizeFeatures(extractFeatures(candidates));
    const targetScaled = normalizeFeatures(extractFeatures([currentUser]))[0];

    candidates.forEach((c, i) => {
        c.scaledFeatures = scaledData[i]; 
        
        // Base 5D KNN Distance
        let distance = Math.sqrt(c.scaledFeatures.reduce((sum, val, idx) => {
            return sum + (ML_CONFIG.featureWeights[idx] * Math.pow(val - targetScaled[idx], 2));
        }, 0));

        // CF Boost (O(1) Lookup)
        let cfBoost = 0;
        const likersOfCandidate = globalItemLikers[c.username] || new Set();
        likersOfCandidate.forEach(liker => {
            if (globalUserSimMatrix[username] && globalUserSimMatrix[username][liker]) {
                cfBoost += globalUserSimMatrix[username][liker]; 
            }
        });

        // AGNES Boost
        let agnesBoost = 0;
        if (preferredArchetype && globalAgnesMap[c.username] === parseInt(preferredArchetype)) {
            agnesBoost = ML_CONFIG.agnesArchetypeBoost; 
        }

        // Apriori Boost
        let aprioriBoost = 0;
        if (c.breed && userBreedAssociations.has(`breed_${c.breed}`)) {
            const supportScore = userBreedAssociations.get(`breed_${c.breed}`);
            aprioriBoost = supportScore * ML_CONFIG.aprioriConfidenceMultiplier; 
        }

        c.hybridDistance = distance - (cfBoost * ML_CONFIG.cfBoostWeight) - agnesBoost - aprioriBoost; 
    });
    
    candidates.sort((a, b) => a.hybridDistance - b.hybridDistance);
    candidates = candidates.slice(0, 50); 

    // Random Forest Inference (Now mapping 7 Features including rules)
    if (globalRfModel) {
        const inferenceData = candidates.map(c => {
            const f1 = targetScaled;
            const f2 = c.scaledFeatures; 
            const diffVector = f1.map((val, idx) => Math.abs(val - f2[idx]));
            
            // Re-apply the explicit categorical rules for the prediction vector
            const isSameBreed = (currentUser.breed && c.breed && currentUser.breed === c.breed) ? 1 : 0;
            const ageDiff = Math.abs((parseInt(currentUser.birthYear) || 2020) - (parseInt(c.birthYear) || 2020));
            
            diffVector.push(isSameBreed, ageDiff);
            return diffVector;
        });
        
        const predictions = globalRfModel.predict(inferenceData);
        
        candidates.forEach((c, i) => {
            c.matchScore = predictions[i] === 1 ? 'High' : 'Low'; 
        });
    } else {
        candidates.forEach(c => c.matchScore = 'Pending Model');
    }

    candidates.forEach(c => delete c.scaledFeatures);
    return candidates;
}

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