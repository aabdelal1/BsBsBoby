const express = require('express');

function sanitizeAgnes(node) {
    if (!node) return null;
    return {
        height: node.height, size: node.size, isLeaf: node.isLeaf,
        children: node.children ? node.children.map(sanitizeAgnes) : []
    };
}

module.exports = (mlPipeline, PETS_CSV, petHeaders, INTERACTIONS_CSV, breedMap, USERS_CSV, userHeaders) => {
    const router = express.Router();

    router.get('/dashboard', async (req, res) => {
        try {
            const path = require('path');
            const SUSPICIOUS_CSV = path.join(path.dirname(PETS_CSV), 'suspicious_profiles.csv');
            const pets = await mlPipeline.readCsv(PETS_CSV);
            let suspiciousPets = [];
            try {
                suspiciousPets = await mlPipeline.readCsv(SUSPICIOUS_CSV);
            } catch (e) {}

            const flaggedUsers = pets.filter(p => p.isFlagged === 'true').map(p => {
                const susMatch = suspiciousPets.find(sp => sp.username === p.username);
                return {
                    ...p,
                    breed: breedMap[p.breed] || p.breed,
                    reason: susMatch ? susMatch.reason : 'Anomalous physical dimension ratio flagged by Mahalanobis-Chi2 distance check.'
                };
            });
            const approvedUsers = pets.filter(p => p.isFlagged !== 'true').map(p => ({ ...p, breed: breedMap[p.breed] || p.breed }));
            const interactions = await mlPipeline.readCsv(INTERACTIONS_CSV);

            const agnesTree = mlPipeline.getAgnesTree();
            const rawApriori = mlPipeline.getAprioriRules();

            // Format apriori data for the frontend chart
            const formatItem = (item) => {
                if (item.startsWith('breed_')) {
                    const breedId = item.substring(6);
                    const breedName = breedMap[breedId] || breedId;
                    return `breed_${breedName}`;
                }
                return item;
            };

            const aprioriList = [];
            for (const [item, associations] of Object.entries(rawApriori)) {
                for (const [assocItem, support] of associations) {
                    aprioriList.push({ items: [formatItem(item), formatItem(assocItem)], support: support });
                }
            }

            // Compute KMeans data from the pets list
            const kmeansData = { clusterCounts: {}, total: 0 };
            pets.forEach(p => {
                if (p.clusterGroup !== undefined && p.clusterGroup !== '' && p.isFlagged !== 'true') {
                    kmeansData.clusterCounts[p.clusterGroup] = (kmeansData.clusterCounts[p.clusterGroup] || 0) + 1;
                    kmeansData.total++;
                }
            });

            const elbowData = mlPipeline.getElbowData ? mlPipeline.getElbowData() : { wcss: [], optimalK: 3 };

            res.json({
                success: true, suspicious: flaggedUsers, users: approvedUsers,
                interactions: interactions, agnes: sanitizeAgnes(agnesTree.tree), optimalK: agnesTree.optimalK,
                apriori: aprioriList, kmeans: kmeansData, elbow: elbowData
            });
        } catch (err) { res.status(500).json({ error: 'Failed to load dashboard data' }); }
    });

    router.patch('/pets/:username/accept', async (req, res) => {
        try {
            const username = req.params.username;
            let pets = await mlPipeline.readCsv(PETS_CSV);
            const petIndex = pets.findIndex(p => p.username === username);
            if (petIndex > -1) {
                pets[petIndex].isFlagged = 'false';
                pets[petIndex].clusterGroup = mlPipeline.assignToCluster(pets[petIndex]);
                await mlPipeline.writeCsv(PETS_CSV, petHeaders, pets);
            }

            // Unblock user in users.csv if blocked
            try {
                let usersList = await mlPipeline.readCsv(USERS_CSV);
                const userIndex = usersList.findIndex(u => u.username === username);
                if (userIndex > -1) {
                    usersList[userIndex].isBlocked = 'false';
                    await mlPipeline.writeCsv(USERS_CSV, userHeaders, usersList);
                }
            } catch (e) {}

            // Also clean up from suspicious_profiles.csv
            const path = require('path');
            const SUSPICIOUS_CSV = path.join(path.dirname(PETS_CSV), 'suspicious_profiles.csv');
            try {
                let suspiciousList = await mlPipeline.readCsv(SUSPICIOUS_CSV);
                suspiciousList = suspiciousList.filter(p => p.username !== username);
                const suspiciousHeaders = [
                    { id: 'username', title: 'username' }, { id: 'petName', title: 'petName' },
                    { id: 'type', title: 'type' }, { id: 'gender', title: 'gender' },
                    { id: 'birthYear', title: 'birthYear' }, { id: 'vaccination', title: 'vaccination' },
                    { id: 'breed', title: 'breed' }, { id: 'length', title: 'length' },
                    { id: 'weight', title: 'weight' }, { id: 'color', title: 'color' },
                    { id: 'personality', title: 'personality' }, { id: 'photoPath', title: 'photoPath' },
                    { id: 'reason', title: 'reason' }
                ];
                await mlPipeline.writeCsv(SUSPICIOUS_CSV, suspiciousHeaders, suspiciousList);
            } catch (e) {}

            res.json({ success: true });
        } catch (err) { res.status(500).json({ error: 'Failed to accept user' }); }
    });

    router.patch('/pets/:username/refuse', async (req, res) => {
        try {
            const username = req.params.username;
            let pets = await mlPipeline.readCsv(PETS_CSV);
            pets = pets.filter(p => p.username !== username);
            await mlPipeline.writeCsv(PETS_CSV, petHeaders, pets);

            // Block user in users.csv
            try {
                let usersList = await mlPipeline.readCsv(USERS_CSV);
                const userIndex = usersList.findIndex(u => u.username === username);
                if (userIndex > -1) {
                    usersList[userIndex].isBlocked = 'true';
                    await mlPipeline.writeCsv(USERS_CSV, userHeaders, usersList);
                }
            } catch (e) {}

            // Also clean up from suspicious_profiles.csv
            const path = require('path');
            const SUSPICIOUS_CSV = path.join(path.dirname(PETS_CSV), 'suspicious_profiles.csv');
            try {
                let suspiciousList = await mlPipeline.readCsv(SUSPICIOUS_CSV);
                suspiciousList = suspiciousList.filter(p => p.username !== username);
                const suspiciousHeaders = [
                    { id: 'username', title: 'username' }, { id: 'petName', title: 'petName' },
                    { id: 'type', title: 'type' }, { id: 'gender', title: 'gender' },
                    { id: 'birthYear', title: 'birthYear' }, { id: 'vaccination', title: 'vaccination' },
                    { id: 'breed', title: 'breed' }, { id: 'length', title: 'length' },
                    { id: 'weight', title: 'weight' }, { id: 'color', title: 'color' },
                    { id: 'personality', title: 'personality' }, { id: 'photoPath', title: 'photoPath' },
                    { id: 'reason', title: 'reason' }
                ];
                await mlPipeline.writeCsv(SUSPICIOUS_CSV, suspiciousHeaders, suspiciousList);
            } catch (e) {}

            res.json({ success: true });
        } catch (err) { res.status(500).json({ error: 'Failed to refuse user' }); }
    });

    return router;
};
