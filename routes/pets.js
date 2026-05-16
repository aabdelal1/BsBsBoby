const express = require('express');
const path = require('path');

module.exports = (mlPipeline, PETS_CSV, petHeaders, dbDir) => {
    const router = express.Router();

    router.post('/', async (req, res) => {
        try {
            const { username, petName, type, gender, birthYear, vaccination, breed, length, weight, color, personality, photoPath } = req.body;
            if (!username) return res.status(400).json({ error: 'Username is required' });

            let pets = await mlPipeline.readCsv(PETS_CSV);
            let petIndex = pets.findIndex(p => p.username === username);

            const newPetData = mlPipeline.preprocess({
                username, petName: petName || '', type: type || '', gender: gender || '',
                birthYear: birthYear || '', vaccination: vaccination || '', breed: breed || '',
                length: length || '', weight: weight || '', color: color || '',
                personality: personality || '', photoPath: photoPath || ''
            });

            const isAnomaly = mlPipeline.gatekeeper(newPetData);
            newPetData.isFlagged = isAnomaly ? 'true' : 'false';
            newPetData.clusterGroup = isAnomaly ? '' : mlPipeline.assignToCluster(newPetData);

            if (petIndex > -1) pets[petIndex] = { ...pets[petIndex], ...newPetData };
            else pets.push(newPetData);

            await mlPipeline.writeCsv(PETS_CSV, petHeaders, pets);
            
            if (isAnomaly) return res.json({ success: true, flagged: true, message: 'Account creation pending review' });
            res.json({ success: true, message: 'Pet profile saved' });

        } catch (err) { console.error("Pet Profile Error:", err); res.status(500).json({ error: 'Server error while saving pet profile' }); }
    });

    router.get('/search', async (req, res) => {
        try {
            const query = (req.query.q || '').toLowerCase();
            const pets = await mlPipeline.readCsv(PETS_CSV);
            
            const results = pets.filter(p => 
                (p.petName || '').toLowerCase().includes(query) ||
                (p.breed || '').toLowerCase().includes(query) ||
                (p.type || '').toLowerCase().includes(query)
            );
            
            res.json({ success: true, results });
        } catch (err) { res.status(500).json({ error: 'Search failed' }); }
    });

    router.get('/:username', async (req, res) => {
        try {
            const username = req.params.username;
            const pets = await mlPipeline.readCsv(PETS_CSV);
            const pet = pets.find(p => p.username === username);
            if (!pet) return res.status(404).json({ error: 'Pet not found' });
            res.json({ success: true, pet });
        } catch (err) { res.status(500).json({ error: 'Server error' }); }
    });

    return router;
};
