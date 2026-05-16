const express = require('express');
const path = require('path');

module.exports = (mlPipeline, dbDir) => {
    const router = express.Router();

    router.get('/', async (req, res) => {
        try {
            const type = req.query.type;
            const file = type === 'dog' 
                ? path.join(dbDir, 'updated_dog_breeds.csv') 
                : path.join(dbDir, 'updated_cat_breeds.csv');
                
            const breeds = await mlPipeline.readCsv(file);
            const breedData = breeds.map(b => ({ id: b.breed_id, name: b.Name })).filter(b => b.name && b.id);
            res.json({ success: true, breeds: breedData });
        } catch (err) { res.status(500).json({ error: 'Server error while fetching breeds' }); }
    });

    return router;
};
