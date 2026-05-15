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

    return router;
};
