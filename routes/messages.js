const express = require('express');

module.exports = (mlPipeline, MESSAGES_CSV, messageHeaders, USERS_CSV, PETS_CSV) => {
    const router = express.Router();

    router.patch('/accept', async (req, res) => {
        try {
            const { fromUser, toUser } = req.body;
            let messages = await mlPipeline.readCsv(MESSAGES_CSV);
            const msgIndex = messages.findIndex(m => m.fromUser === fromUser && m.toUser === toUser);
            if (msgIndex > -1) {
                messages[msgIndex].status = 'accepted';
                await mlPipeline.writeCsv(MESSAGES_CSV, messageHeaders, messages);
            }
            res.json({ success: true });
        } catch (err) { res.status(500).json({ error: 'Failed to accept message' }); }
    });

    router.patch('/reject', async (req, res) => {
        try {
            const { fromUser, toUser } = req.body;
            console.log(`Rejecting message from [${fromUser}] to [${toUser}]`);
            let messages = await mlPipeline.readCsv(MESSAGES_CSV);
            const filteredMessages = messages.filter(m => 
                !(m.fromUser.trim() === fromUser.trim() && m.toUser.trim() === toUser.trim())
            );
            if (messages.length !== filteredMessages.length) {
                await mlPipeline.writeCsv(MESSAGES_CSV, messageHeaders, filteredMessages);
                console.log(`Message removed.`);
            }
            res.json({ success: true });
        } catch (err) { 
            console.error("Reject error:", err);
            res.status(500).json({ error: 'Failed to reject message' }); 
        }
    });

    router.get('/', async (req, res) => {
        try {
            const username = req.query.username;
            if (!username) return res.status(400).json({ error: 'Username required' });
            
            let messages = await mlPipeline.readCsv(MESSAGES_CSV);
            messages = messages.filter(m => m.toUser === username || m.fromUser === username);
            
            const users = await mlPipeline.readCsv(USERS_CSV);
            const pets = await mlPipeline.readCsv(PETS_CSV);
            
            // Enrich messages with other user's photo and pet's photo
            const enrichedMessages = messages.map(m => {
                const otherUser = m.fromUser === username ? m.toUser : m.fromUser;
                const user = users.find(u => u.username === otherUser);
                const pet = pets.find(p => p.username === otherUser);
                return {
                    ...m,
                    otherUserPhoto: user ? user.photoPath : null,
                    otherPetPhoto: pet ? pet.photoPath : null,
                    petName: pet ? pet.petName : 'Unknown Pet'
                };
            });
            
            res.json({ success: true, messages: enrichedMessages });
        } catch (err) { res.status(500).json({ error: 'Failed to fetch messages' }); }
    });

    return router;
};
