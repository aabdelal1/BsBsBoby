const express = require('express');

module.exports = (mlPipeline, USERS_CSV, userHeaders, breedMap) => {
    const router = express.Router();

    router.put('/:username', async (req, res) => {
        try {
            const username = req.params.username;
            const { phone, location, fullName, photoPath } = req.body;
            let users = await mlPipeline.readCsv(USERS_CSV);
            let userIndex = users.findIndex(u => u.username === username);

            if (userIndex === -1) return res.status(404).json({ error: 'User not found' });

            if (phone !== undefined) users[userIndex].phone = phone;
            if (location !== undefined) users[userIndex].location = location;
            if (fullName !== undefined) users[userIndex].fullName = fullName;
            if (photoPath !== undefined) users[userIndex].photoPath = photoPath;

            await mlPipeline.writeCsv(USERS_CSV, userHeaders, users);
            res.json({ success: true, message: 'Profile updated' });
        } catch (err) { res.status(500).json({ error: 'Server error' }); }
    });

    router.get('/:username/playdates', async (req, res) => {
        try {
            const username = req.params.username;
            const candidatesRaw = await mlPipeline.getPlaydatesFeed(username);
            const candidates = candidatesRaw.map(c => ({...c, breed: breedMap[c.breed] || c.breed}));
            res.json({ success: true, candidates });
        } catch (err) { console.error(err); res.status(500).json({ error: 'Failed to fetch playdates' }); }
    });

    router.get('/:username', async (req, res) => {
        try {
            const username = req.params.username;
            const users = await mlPipeline.readCsv(USERS_CSV);
            const user = users.find(u => u.username === username);
            if (!user) return res.status(404).json({ error: 'User not found' });
            
            // Exclude password for security
            const { password, ...safeUser } = user;
            res.json({ success: true, user: safeUser });
        } catch (err) { res.status(500).json({ error: 'Server error' }); }
    });

    return router;
};
