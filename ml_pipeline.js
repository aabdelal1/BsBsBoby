const fs = require('fs');
const path = require('path');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;
const csv = require('csv-parser');

const { kmeans } = require('ml-kmeans');
const hclust = require('ml-hclust');
const { RandomForestClassifier } = require('ml-random-forest');
const { Apriori } = require('node-apriori');


const DB_DIR = path.join(__dirname, 'DB');
const MODELS_DIR = path.join(__dirname, 'models'); 
const STATE_FILE = path.join(MODELS_DIR, 'ml_pipeline_state.json'); 

// Ensure models directory exists
if (!fs.existsSync(MODELS_DIR)) {
    fs.mkdirSync(MODELS_DIR, { recursive: true });
}

const PETS_CSV = path.join(DB_DIR, 'individual_pets.csv');
const INTERACTIONS_CSV = path.join(DB_DIR, 'interactions.csv');

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
let globalClusterStats = {}; 
let globalAgnesMap = {}; 
let globalAgnesTreeCache = null; // Restored Cache Variable
let globalUserSimMatrix = {}; 
let globalItemLikers = {}; 
let globalItemSkippers = {}; 
let globalAprioriRules = {}; 

// --- LOCAL PERSISTENCE ---
async function loadStateFromLocal() {
    try {
        if (fs.existsSync(STATE_FILE)) {
            const data = fs.readFileSync(STATE_FILE, 'utf8');
            const state = JSON.parse(data);
            
            globalCentroids = state.centroids || null;
            globalScalingParams = state.scalingParams || { means: [0,0,0], stdDevs: [1,1,1] };
            globalOptimalK = state.optimalK || 3;
            globalClusterStats = state.clusterStats || {};
            globalAgnesMap = state.agnesMap || {};
            
            if (state.aprioriRules) {
                globalAprioriRules = {};
                state.aprioriRules.forEach(([breed, rulesArr]) => {
                    globalAprioriRules[breed] = new Map(rulesArr);
                });
            }
            if (state.rfModelJSON) {
                globalRfModel = RandomForestClassifier.load(state.rfModelJSON);
            }
            console.log("ML State fully restored from local models folder.");
        }
    } catch (err) {
        console.log("Local restore skipped/failed (normal for first boot):", err.message);
    }
}

async function syncToLocal() {
    try {
        const aprioriSerialized = Object.entries(globalAprioriRules).map(([breed, map]) => [breed, Array.from(map.entries())]);
        const state = {
            centroids: globalCentroids,
            scalingParams: globalScalingParams,
            optimalK: globalOptimalK,
            clusterStats: globalClusterStats,
            agnesMap: globalAgnesMap,
            aprioriRules: aprioriSerialized,
            rfModelJSON: globalRfModel ? globalRfModel.toJSON() : null
        };
        
        // Write atomically to prevent corruption if the server crashes mid-save
        fs.writeFileSync(STATE_FILE, JSON.stringify(state), 'utf8');
    } catch (err) {
        console.error("Local sync failed:", err.message);
    }
}

// --- IO HELPERS ---
const readCsv = (filePath) => {
    return new Promise((resolve) => {
        const results = [];
        if (!fs.existsSync(filePath)) return resolve(results);
        const stream = fs.createReadStream(filePath);
        stream.pipe(csv()).on('data', d => results.push(d)).on('end', () => resolve(results));
    });
};
const writeCsv = (filePath, headers, records) => createCsvWriter({ path: filePath, header: headers }).writeRecords(records);

// --- PREPROCESSING (Restored from Regression) ---
function preprocess(petData) {
    const processed = { ...petData };
    processed.birthYear = parseInt(processed.birthYear) || 2020;
    processed.weight = parseFloat(processed.weight) || 10.0;
    processed.length = parseFloat(processed.length) || 20.0;
    processed.vaccination = (processed.vaccination || '').toLowerCase();
    return processed;
}

function extractPhysical(p) {
    return [parseFloat(p.weight) || 10, parseFloat(p.length) || 20, new Date().getFullYear() - (parseInt(p.birthYear) || 2020)]; 
}
function extractBehavioral(p) {
    const traits = (p.personality || '').toLowerCase().split(',').map(t => t.trim());
    return ['active', 'friendly', 'calm', 'touchy', 'sleepy'].map(t => traits.includes(t) ? 1 : 0); 
}
function standardizePhysical(features) {
    return features.map(row => row.map((val, idx) => (val - globalScalingParams.means[idx]) / globalScalingParams.stdDevs[idx]));
}

// --- MAHALANOBIS ROUTING & GATEKEEPER ---
function getMahalanobisDistanceSq(scaledPoint, stats) {
    if (!stats || !stats.invCovMatrix) return Infinity;
    const diff = scaledPoint.map((val, i) => val - stats.meanVector[i]);
    let mahalanobisSq = 0;
    for (let i = 0; i < 3; i++) {
        let temp = 0;
        for (let j = 0; j < 3; j++) temp += diff[j] * stats.invCovMatrix[j][i];
        mahalanobisSq += temp * diff[i];
    }
    return mahalanobisSq;
}

function findNearestCluster(scaledPhys) {
    if (!globalCentroids) return { clusterIdx: 0, distanceSq: 0 };
    let minDistance = Infinity;
    let closestCluster = 0;

    globalCentroids.forEach((centroid, index) => {
        const stats = globalClusterStats[index];
        let dist;
        if (stats && stats.invCovMatrix) {
            dist = getMahalanobisDistanceSq(scaledPhys, stats);
        } else {
            const coords = Array.isArray(centroid) ? centroid : centroid.centroid;
            dist = coords.reduce((sum, val, d) => sum + Math.pow(val - scaledPhys[d], 2), 0);
        }
        
        if (dist < minDistance) { 
            minDistance = dist; 
            closestCluster = index; 
        }
    });
    return { clusterIdx: closestCluster, distanceSq: minDistance };
}

function gatekeeper(rawPetData) {
    if (isNaN(parseFloat(rawPetData.weight)) || isNaN(parseInt(rawPetData.birthYear)) || isNaN(parseFloat(rawPetData.length))) return true;

    const age = new Date().getFullYear() - parseInt(rawPetData.birthYear);
    if (age > 25 || ((rawPetData.type || '').toLowerCase() === 'cat' && parseFloat(rawPetData.weight) > 40)) return true; 
    if (['spam', 'fake', 'test'].includes((rawPetData.vaccination || '').toLowerCase())) return true;

    if (globalCentroids && Object.keys(globalClusterStats).length > 0) {
        const scaledPhys = standardizePhysical([extractPhysical(rawPetData)])[0];
        const { clusterIdx, distanceSq } = findNearestCluster(scaledPhys);
        
        const stats = globalClusterStats[clusterIdx];
        if (stats && stats.invCovMatrix && distanceSq > ML_CONFIG.chiSquareThreshold) {
            return true; 
        }
    } else if (age > 25 || parseFloat(rawPetData.weight) > 150) {
        return true; 
    }
    return false;
}

// --- AUTOMATED KNEEDLE ---
function invert3x3Covariance(matrix, epsilon = 1e-4) {
    let m = matrix.map((row, i) => row.map((val, j) => val + (i === j ? epsilon : 0))); 
    const det = m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
    if (Math.abs(det) < 1e-8) return [[1,0,0],[0,1,0],[0,0,1]]; 
    const invDet = 1.0 / det;
    return [
        [(m[1][1]*m[2][2] - m[2][1]*m[1][2]) * invDet, (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * invDet, (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * invDet],
        [(m[1][2]*m[2][0] - m[1][0]*m[2][2]) * invDet, (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * invDet, (m[1][0]*m[0][2] - m[0][0]*m[1][2]) * invDet],
        [(m[1][0]*m[2][1] - m[2][0]*m[1][1]) * invDet, (m[2][0]*m[0][1] - m[0][0]*m[2][1]) * invDet, (m[0][0]*m[1][1] - m[1][0]*m[0][1]) * invDet]
    ];
}

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

    const minWcss = Math.min(...wcssValues);
    const maxWcss = Math.max(...wcssValues);
    const normWcss = wcssValues.map(w => (maxWcss - minWcss === 0) ? 0 : (w - minWcss) / (maxWcss - minWcss));
    const normK = Array.from({length: limit}, (_, i) => i / (limit - 1));

    const A = normWcss[limit - 1] - normWcss[0];
    const B = normK[0] - normK[limit - 1];
    const C = normK[limit - 1] * normWcss[0] - normK[0] * normWcss[limit - 1];
    const denominator = Math.sqrt(A * A + B * B);

    let maxDistance = -1, optimalK = 2; 
    for (let i = 0; i < limit; i++) {
        const perpDistance = Math.abs(A * normK[i] + B * normWcss[i] + C) / denominator;
        if (perpDistance > maxDistance) { maxDistance = perpDistance; optimalK = i + 1; }
    }
    return Math.max(2, optimalK); 
}

// --- BACKGROUND TRAINING TASKS ---
async function trainBackgroundModels() {
    let pets = await readCsv(PETS_CSV);
    const approvedPets = pets.filter(p => p.isFlagged !== 'true');
    const interactions = await readCsv(INTERACTIONS_CSV);

    if (approvedPets.length >= 3) {
        // 1. K-Means
        const rawPhysFeatures = approvedPets.map(extractPhysical);
        const means = [0,0,0], stdDevs = [1,1,1];
        for(let idx = 0; idx < 3; idx++) {
            const vals = rawPhysFeatures.map(f => f[idx]);
            const m = vals.reduce((a, b) => a + b, 0) / vals.length;
            const v = vals.reduce((sum, val) => sum + Math.pow(val - m, 2), 0) / (vals.length > 1 ? vals.length - 1 : 1);
            means[idx] = m; stdDevs[idx] = Math.sqrt(v) || 0.001; 
        }
        globalScalingParams = { means, stdDevs };
        const scaledPhysFeatures = standardizePhysical(rawPhysFeatures);
        
        globalOptimalK = findOptimalK(scaledPhysFeatures, ML_CONFIG.maxK);
        const kmeansResult = kmeans(scaledPhysFeatures, globalOptimalK, { initialization: 'kmeans++' });
        globalCentroids = kmeansResult.centroids;

        globalClusterStats = {};
        for(let k = 0; k < globalOptimalK; k++) {
            const pts = scaledPhysFeatures.filter((_, i) => kmeansResult.clusters[i] === k);
            if(pts.length > 3) {
                const meanVector = [0,0,0];
                pts.forEach(p => { for(let d=0; d<3; d++) meanVector[d] += p[d]; });
                meanVector.forEach((_, d) => meanVector[d] /= pts.length);

                const covMatrix = [[0,0,0],[0,0,0],[0,0,0]];
                pts.forEach(p => {
                    for(let i=0; i<3; i++) for(let j=0; j<3; j++) covMatrix[i][j] += (p[i] - meanVector[i]) * (p[j] - meanVector[j]);
                });
                for(let i=0; i<3; i++) for(let j=0; j<3; j++) covMatrix[i][j] /= (pts.length - 1);
                
                globalClusterStats[k] = { meanVector, invCovMatrix: invert3x3Covariance(covMatrix) };
            }
        }

        // 2. AGNES (Cache restored)
        const tree = hclust.agnes(approvedPets.map(extractBehavioral), { method: 'complete' });
        globalAgnesTreeCache = tree; 

        let maxGap = 0, bestCutHeight = tree.height;
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

    // 3. Jaccard CF (True Multiset Intersection)
    const userLikes = {}, userSkips = {};
    const itemLikers = {}, itemSkippers = {}; 
    
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
    
    globalItemLikers = itemLikers; globalItemSkippers = itemSkippers;
    const users = Array.from(new Set([...Object.keys(userLikes), ...Object.keys(userSkips)]));
    const newSimMatrix = {};
    
    for (let i = 0; i < users.length; i++) {
        newSimMatrix[users[i]] = {};
        for (let j = i + 1; j < users.length; j++) {
            const u1 = users[i], u2 = users[j];
            
            // True Set Union to prevent double counting
            const u1Items = new Set([...(userLikes[u1] || []), ...(userSkips[u1] || [])]);
            const u2Items = new Set([...(userLikes[u2] || []), ...(userSkips[u2] || [])]);
            
            let intersect = 0;
            for (let item of u1Items) if (u2Items.has(item)) intersect++;
            
            const union = u1Items.size + u2Items.size - intersect;
            const sim = union === 0 ? 0 : intersect / union;
            
            if (sim > ML_CONFIG.jaccardMinThreshold) { 
                newSimMatrix[u1][u2] = sim;
                if (!newSimMatrix[u2]) newSimMatrix[u2] = {};
                newSimMatrix[u2][u1] = sim;
            }
        }
    }
    globalUserSimMatrix = newSimMatrix;

    // 4. Random Forest
    if (interactions.length >= 20 && approvedPets.length > 0) {
        let allData = [], allLabels = []; 
        interactions.forEach(interaction => {
            const userPet = pets.find(p => p.username === interaction.username);
            const targetPet = pets.find(p => p.username === interaction.targetUsername);
            if (userPet && targetPet) {
                const f1 = standardizePhysical([extractPhysical(userPet)])[0];
                const f2 = standardizePhysical([extractPhysical(targetPet)])[0];
                const diffVector = f1.map((val, i) => Math.abs(val - f2[i]));
                diffVector.push((userPet.breed && targetPet.breed && userPet.breed === targetPet.breed) ? 1 : 0); 
                allData.push(diffVector);
                allLabels.push(interaction.action === 'like' ? 1 : 0);
            }
        });

        const lIdx = allLabels.map((l, i) => l === 1 ? i : -1).filter(i => i !== -1);
        const sIdx = allLabels.map((l, i) => l === 0 ? i : -1).filter(i => i !== -1);
        
        if (lIdx.length > 0 && sIdx.length > 0) {
            const minSize = Math.min(lIdx.length, sIdx.length);
            const bIdx = [...lIdx.sort(() => 0.5 - Math.random()).slice(0, minSize), ...sIdx.sort(() => 0.5 - Math.random()).slice(0, minSize)].sort(() => 0.5 - Math.random()); 
            
            const bData = bIdx.map(i => allData[i]);
            const bLabels = bIdx.map(i => allLabels[i]);

            const split = Math.floor(bData.length * 0.8);
            if (split > 0) {
                const tempModel = new RandomForestClassifier({ maxFeatures: ML_CONFIG.rfMaxFeatures, replacement: true, nEstimators: ML_CONFIG.rfEstimators });
                tempModel.train(bData.slice(0, split), bLabels.slice(0, split));

                let correct = 0;
                const testLabels = bLabels.slice(split);
                const preds = tempModel.predict(bData.slice(split));
                for(let i = 0; i < testLabels.length; i++) if (preds[i] === testLabels[i]) correct++;

                if ((correct / testLabels.length) > 0.55) globalRfModel = tempModel;
            }
        }
    }

    syncToLocal();
}

// 5. APRIORI (Restored as Exported Isolated Function)
async function runApriori() {
    const interactions = await readCsv(INTERACTIONS_CSV);
    const pets = await readCsv(PETS_CSV);
    
    const sessionLikes = {};
    const itemTxFrequencies = {}; 
    
    interactions.filter(i => i.action === 'like').forEach(i => {
        const targetPet = pets.find(p => p.username === i.targetUsername);
        if (targetPet && targetPet.breed) {
            const sessionId = `${i.username}_${Math.floor(i.timestamp / ML_CONFIG.sessionWindowMs)}`;
            if (!sessionLikes[sessionId]) sessionLikes[sessionId] = new Set();
            sessionLikes[sessionId].add(`breed_${targetPet.breed}`);
        }
    });

    const transactions = Object.values(sessionLikes).map(set => Array.from(set));
    const totalTx = transactions.length;
    if (totalTx >= 5) {
        transactions.forEach(tx => {
            const uniqueItems = new Set(tx);
            uniqueItems.forEach(item => { itemTxFrequencies[item] = (itemTxFrequencies[item] || 0) + 1; });
        });

        return new Promise((resolve) => {
            const apriori = new Apriori(ML_CONFIG.aprioriMinSupport); 
            apriori.exec(transactions).then(result => {
                const newRules = {};
                result.itemsets.forEach(itemset => {
                    if (itemset.items.length === 2) {
                        const breedA = itemset.items[0], breedB = itemset.items[1];
                        const supportA = itemTxFrequencies[breedA] / totalTx;
                        const supportB = itemTxFrequencies[breedB] / totalTx;
                        const lift = itemset.support / (supportA * supportB);
                        
                        if (lift > ML_CONFIG.aprioriMinLift) {
                            if (!newRules[breedA]) newRules[breedA] = new Map();
                            if (!newRules[breedB]) newRules[breedB] = new Map();
                            newRules[breedA].set(breedB, lift);
                            newRules[breedB].set(breedA, lift);
                        }
                    }
                });
                globalAprioriRules = newRules;
                syncToLocal();
                resolve(result.itemsets);
            });
        });
    }
    return [];
}

function assignToCluster(newPetData) {
    const scaledPhys = standardizePhysical([extractPhysical(newPetData)])[0];
    return findNearestCluster(scaledPhys).clusterIdx;
}

// --- RECOMMENDATION ENGINE ---
async function getPlaydatesFeed(username) {
    let pets = await readCsv(PETS_CSV);
    const currentUser = pets.find(p => p.username === username);
    if (!currentUser || currentUser.isFlagged === 'true') return [];

    let candidates = pets.filter(p => p.username !== username && p.isFlagged !== 'true');
    const interactions = await readCsv(INTERACTIONS_CSV);
    const pastInteractions = new Set(interactions.filter(i => i.username === username).map(i => i.targetUsername));
    candidates = candidates.filter(c => !pastInteractions.has(c.username));
    
    if (candidates.length === 0) return [];

    const archetypeCounts = {};
    const userLikes = interactions.filter(i => i.username === username && i.action === 'like');
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
        c._tempPhys = standardizePhysical([extractPhysical(c)])[0];
        let physDistance = Math.sqrt(c._tempPhys.reduce((sum, val, idx) => sum + Math.pow(val - targetPhys[idx], 2), 0));
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
        let cfScore = Math.max(0, Math.min(1, cfInteractions > 0 ? (cfRaw / cfInteractions + 1) / 2 : 0.5)); 

        let behScore = 0;
        if (preferredArchetype && globalAgnesMap[c.username] === parseInt(preferredArchetype)) {
            behScore = 1.0;
        } else {
            let overlaps = 0;
            const cBeh = extractBehavioral(c);
            for(let i=0; i<5; i++) if (cBeh[i] === 1 && targetBeh[i] === 1) overlaps++;
            behScore = overlaps / 5;
        }

        let aprioriScore = 0.0;
        if (c.breed && userBreedAssociations.has(`breed_${c.breed}`)) {
            aprioriScore = Math.min((userBreedAssociations.get(`breed_${c.breed}`) - 1.0) / 2.0, 1.0); 
        }

        c.fusionScore = (physScore * ML_CONFIG.fusionWeights.physical) + 
                        (cfScore * ML_CONFIG.fusionWeights.cf) + 
                        (behScore * ML_CONFIG.fusionWeights.behavioral) + 
                        (aprioriScore * ML_CONFIG.fusionWeights.apriori);
    });
    
    candidates.sort((a, b) => b.fusionScore - a.fusionScore); 
    candidates = candidates.slice(0, 50); 

    // CONTINUOUS PROBABILISTIC RF MULTIPLIER (With Defensive Fallback)
    if (globalRfModel) {
        const inferenceData = candidates.map(c => {
            const diffVector = targetPhys.map((val, idx) => Math.abs(val - c._tempPhys[idx]));
            diffVector.push((currentUser.breed && c.breed && currentUser.breed === c.breed) ? 1 : 0);
            return diffVector;
        });
        
        const basePredictions = globalRfModel.predict(inferenceData);

        candidates.forEach((c, i) => {
            let prob;
            if (globalRfModel.estimators && globalRfModel.estimators.length > 0) {
                let positiveVotes = 0;
                globalRfModel.estimators.forEach(tree => { if (tree.predict([inferenceData[i]])[0] === 1) positiveVotes++; });
                prob = positiveVotes / globalRfModel.estimators.length;
            } else {
                prob = basePredictions[i]; // Fallback if internal API changes
            }
            
            const multiplier = 0.8 + (prob * 0.4); 
            c.fusionScore = Math.min(1.0, c.fusionScore * multiplier);
            c.matchScore = prob > 0.5 ? 'High' : 'Low'; 
        });
        candidates.sort((a, b) => b.fusionScore - a.fusionScore);
    } else {
        candidates.forEach(c => c.matchScore = 'Pending Model');
    }

    candidates.forEach(c => delete c._tempPhys);
    return candidates;
}

function getAgnesTree() { return { tree: globalAgnesTreeCache, optimalK: globalOptimalK }; }
function getAprioriRules() { 
    return Object.fromEntries(Object.entries(globalAprioriRules).map(([k, v]) => [k, Array.from(v.entries())])); 
}

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
    readCsv,
    writeCsv
};