const express = require('express');

function sanitizeAgnes(node) {
    if (!node) return null;
    return {
        height: node.height, size: node.size, isLeaf: node.isLeaf,
        children: node.children ? node.children.map(sanitizeAgnes) : []
    };
}

module.exports = (mlPipeline, PETS_CSV, petHeaders, INTERACTIONS_CSV, breedMap) => {
    const router = express.Router();

    router.get('/dashboard', async (req, res) => {
        try {
            const pets = await mlPipeline.readCsv(PETS_CSV);
            const flaggedUsers = pets.filter(p => p.isFlagged === 'true').map(p => ({ ...p, breed: breedMap[p.breed] || p.breed }));
            const approvedUsers = pets.filter(p => p.isFlagged !== 'true').map(p => ({ ...p, breed: breedMap[p.breed] || p.breed }));
            const interactions = await mlPipeline.readCsv(INTERACTIONS_CSV);

            const agnesTree = mlPipeline.getAgnesTree();
            const rawApriori = mlPipeline.getAprioriRules();

            // Format apriori data for the frontend chart
            const aprioriList = [];
            for (const [item, associations] of Object.entries(rawApriori)) {
                for (const [assocItem, support] of associations) {
                    aprioriList.push({ items: [item, assocItem], support: support });
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

            res.json({
                success: true, suspicious: flaggedUsers, users: approvedUsers,
                interactions: interactions, agnes: sanitizeAgnes(agnesTree.tree), optimalK: agnesTree.optimalK,
                apriori: aprioriList, kmeans: kmeansData
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
            res.json({ success: true });
        } catch (err) { res.status(500).json({ error: 'Failed to accept user' }); }
    });

    router.patch('/pets/:username/refuse', async (req, res) => {
        try {
            const username = req.params.username;
            let pets = await mlPipeline.readCsv(PETS_CSV);
            pets = pets.filter(p => p.username !== username);
            await mlPipeline.writeCsv(PETS_CSV, petHeaders, pets);
            res.json({ success: true });
        } catch (err) { res.status(500).json({ error: 'Failed to refuse user' }); }
    });

    return router;
};
