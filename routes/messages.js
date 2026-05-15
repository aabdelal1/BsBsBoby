const express = require('express');

module.exports = (mlPipeline, MESSAGES_CSV, messageHeaders) => {
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

    router.get('/', async (req, res) => {
        try {
            const username = req.query.username;
            if (!username) return res.status(400).json({ error: 'Username required' });
            let messages = await mlPipeline.readCsv(MESSAGES_CSV);
            messages = messages.filter(m => m.toUser === username || m.fromUser === username);
            res.json({ success: true, messages });
        } catch (err) { res.status(500).json({ error: 'Failed to fetch messages' }); }
    });

    return router;
};
