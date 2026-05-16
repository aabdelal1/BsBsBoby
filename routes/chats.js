const express = require('express');

module.exports = (mlPipeline, CHAT_CSV, chatHeaders) => {
    const router = express.Router();

    router.get('/', async (req, res) => {
        try {
            const { userA, userB } = req.query;
            if (!userA || !userB) return res.status(400).json({ error: 'Users required' });
            let chats = await mlPipeline.readCsv(CHAT_CSV);
            chats = chats.filter(c => (c.fromUser === userA && c.toUser === userB) || (c.fromUser === userB && c.toUser === userA));
            res.json({ success: true, chats });
        } catch (err) { res.status(500).json({ error: 'Failed to fetch chat' }); }
    });

    router.post('/', async (req, res) => {
        try {
            const { fromUser, toUser, message } = req.body;
            let chats = await mlPipeline.readCsv(CHAT_CSV);
            chats.push({ fromUser, toUser, message, timestamp: Date.now() });
            await mlPipeline.writeCsv(CHAT_CSV, chatHeaders, chats);
            res.json({ success: true });
        } catch (err) { res.status(500).json({ error: 'Failed to send message' }); }
    });

    return router;
};
