const express = require('express');

module.exports = (mlPipeline, INTERACTIONS_CSV, interactionHeaders, MESSAGES_CSV, messageHeaders) => {
    const router = express.Router();

    router.post('/', async (req, res) => {
        try {
            const { username, targetUsername, action } = req.body;
            let interactions = await mlPipeline.readCsv(INTERACTIONS_CSV);
            interactions.push({ username, targetUsername, action, timestamp: Date.now() });
            await mlPipeline.writeCsv(INTERACTIONS_CSV, interactionHeaders, interactions);
            
            if (action === 'like') {
                let messages = await mlPipeline.readCsv(MESSAGES_CSV);
                const existing = messages.find(m => (m.fromUser === username && m.toUser === targetUsername) || (m.fromUser === targetUsername && m.toUser === username));
                if (!existing) {
                    messages.push({ fromUser: username, toUser: targetUsername, status: 'pending' });
                    await mlPipeline.writeCsv(MESSAGES_CSV, messageHeaders, messages);
                }
            }
            res.json({ success: true });
        } catch (err) { res.status(500).json({ error: 'Failed to record interaction' }); }
    });

    return router;
};
