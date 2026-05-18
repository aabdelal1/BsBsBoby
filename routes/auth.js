const express = require('express');
const bcrypt = require('bcrypt');

module.exports = (mlPipeline, USERS_CSV, PETS_CSV, userHeaders, breedMap) => {
    const router = express.Router();

    router.post('/register', async (req, res) => {
        try {
            const { username, email, phone, location, password } = req.body;
            if (!username || !email || !phone || !location || !password) return res.status(400).json({ error: 'All fields are mandatory' });

            const users = await mlPipeline.readCsv(USERS_CSV);
            if (users.find(u => u.username === username)) return res.status(400).json({ error: 'Username already exists' });

            const hashedPassword = await bcrypt.hash(password, 10);
            users.push({ username, email, phone, location, password: hashedPassword, fullName: '', photoPath: '' });
            await mlPipeline.writeCsv(USERS_CSV, userHeaders, users);
            
            res.status(201).json({ success: true, message: 'User created' });
        } catch (err) { res.status(500).json({ error: 'Server error' }); }
    });

    router.post('/login', async (req, res) => {
        try {
            const { username, password } = req.body;
            const users = await mlPipeline.readCsv(USERS_CSV);
            const user = users.find(u => u.username === username);

            if (!user || !(await bcrypt.compare(password, user.password))) return res.status(401).json({ error: 'Invalid username or password' });

            const isAdmin = username === 'admin';

            let pet = null;
            if (!isAdmin) {
                const pets = await mlPipeline.readCsv(PETS_CSV);
                const found = pets.find(p => p.username === username);
                if (found) pet = { ...found, breed: breedMap[found.breed] || found.breed };
            }

            res.json({ success: true, message: 'Login successful', isAdmin, user: { username: user.username, email: user.email, phone: user.phone, location: user.location, fullName: user.fullName, photoPath: user.photoPath, isBlocked: user.isBlocked === 'true' }, pet });
        } catch (err) { res.status(500).json({ error: 'Server error' }); }
    });

    return router;
};
