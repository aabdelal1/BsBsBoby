const express = require('express');
const bodyParser = require('body-parser');
const csv = require('csv-parser');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;
const fs = require('fs');
const path = require('path');
const bcrypt = require('bcrypt');
const multer = require('multer');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(bodyParser.json());
app.use(express.static('.'));
app.use('/uploads', express.static('uploads'));

const DB_DIR = path.join(__dirname, 'DB');
const UPLOADS_DIR = path.join(__dirname, 'uploads');
const USERS_CSV = path.join(DB_DIR, 'users.csv');
const PETS_CSV = path.join(DB_DIR, 'individual_pets.csv');
const SUSPICIOUS_CSV = path.join(DB_DIR, 'suspicious_profiles.csv');
const INTERACTIONS_CSV = path.join(DB_DIR, 'interactions.csv');
const MESSAGES_CSV = path.join(DB_DIR, 'messages.csv');
const CHAT_CSV = path.join(DB_DIR, 'chat.csv');

const mlPipeline = require('./ml_pipeline');

if (!fs.existsSync(DB_DIR)) fs.mkdirSync(DB_DIR);
if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR);

const userHeaders = [
    { id: 'username', title: 'username' }, { id: 'email', title: 'email' },
    { id: 'phone', title: 'phone' }, { id: 'location', title: 'location' },
    { id: 'password', title: 'password' }, { id: 'fullName', title: 'fullName' },
    { id: 'photoPath', title: 'photoPath' }
];

const petHeaders = [
    { id: 'username', title: 'username' }, { id: 'petName', title: 'petName' },
    { id: 'type', title: 'type' }, { id: 'gender', title: 'gender' },
    { id: 'birthYear', title: 'birthYear' }, { id: 'vaccination', title: 'vaccination' },
    { id: 'breed', title: 'breed' }, { id: 'length', title: 'length' },
    { id: 'weight', title: 'weight' }, { id: 'color', title: 'color' },
    { id: 'personality', title: 'personality' }, { id: 'photoPath', title: 'photoPath' },
    { id: 'isFlagged', title: 'isFlagged' }, { id: 'clusterGroup', title: 'clusterGroup' }
];

const suspiciousHeaders = [...petHeaders, { id: 'reason', title: 'reason' }];
const interactionHeaders = [
    { id: 'username', title: 'username' }, { id: 'targetUsername', title: 'targetUsername' },
    { id: 'action', title: 'action' }, { id: 'timestamp', title: 'timestamp' }
];
const messageHeaders = [
    { id: 'fromUser', title: 'fromUser' }, { id: 'toUser', title: 'toUser' },
    { id: 'status', title: 'status' }
];
const chatHeaders = [
    { id: 'fromUser', title: 'fromUser' }, { id: 'toUser', title: 'toUser' },
    { id: 'message', title: 'message' }, { id: 'timestamp', title: 'timestamp' }
];

[
    { path: USERS_CSV, header: userHeaders }, { path: PETS_CSV, header: petHeaders },
    { path: SUSPICIOUS_CSV, header: suspiciousHeaders }, { path: INTERACTIONS_CSV, header: interactionHeaders },
    { path: MESSAGES_CSV, header: messageHeaders }, { path: CHAT_CSV, header: chatHeaders }
].forEach(file => {
    if (!fs.existsSync(file.path)) createCsvWriter(file).writeRecords([]);
});

let breedMap = {};
async function loadBreeds() {
    try {
        const dogs = await mlPipeline.readCsv(path.join(DB_DIR, 'updated_dog_breeds.csv'));
        dogs.forEach(d => { if(d.breed_id) breedMap[d.breed_id] = d.Name; });
        const cats = await mlPipeline.readCsv(path.join(DB_DIR, 'updated_cat_breeds.csv'));
        cats.forEach(c => { if(c.breed_id) breedMap[c.breed_id] = c.Name; });
        console.log("Global breed dictionary loaded.");
    } catch(e) {}
}
loadBreeds();

// --- BACKGROUND MACHINE LEARNING TASKS ---
let globalAprioriRules = [];

const runBackgroundTasks = async () => {
    try {
        await mlPipeline.trainBackgroundModels();
        console.log("Background Pipeline (5D K-Means, AGNES Archetypes, Jaccard Synergy, 7-Rule RF) trained.");
        globalAprioriRules = await mlPipeline.runApriori();
        console.log("Apriori rule sets injected into active recommendation engine.");
    } catch (err) { console.error("Background task error:", err); }
};

setInterval(runBackgroundTasks, 300000); 
setTimeout(runBackgroundTasks, 2000);

const upload = multer({
    storage: multer.diskStorage({
        destination: (req, file, cb) => cb(null, 'uploads/'),
        filename: (req, file, cb) => cb(null, Date.now() + path.extname(file.originalname))
    })
});

// --- ROUTES ---

app.post('/api/signup', async (req, res) => {
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

app.post('/api/login', async (req, res) => {
    try {
        const { username, password } = req.body;
        const users = await mlPipeline.readCsv(USERS_CSV);
        const user = users.find(u => u.username === username);

        if (!user || !(await bcrypt.compare(password, user.password))) return res.status(401).json({ error: 'Invalid username or password' });

        const pets = await mlPipeline.readCsv(PETS_CSV);
        let pet = pets.find(p => p.username === username) || null;
        if (pet) pet = { ...pet, breed: breedMap[pet.breed] || pet.breed };
        
        res.json({ success: true, message: 'Login successful', user: { username: user.username, email: user.email, phone: user.phone, location: user.location, fullName: user.fullName, photoPath: user.photoPath }, pet });
    } catch (err) { res.status(500).json({ error: 'Server error' }); }
});

app.put('/api/user/:username', async (req, res) => {
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

app.post('/api/pet', async (req, res) => {
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

app.post('/api/upload', upload.single('media'), (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
    res.json({ success: true, filePath: `/uploads/${req.file.filename}` });
});

app.get('/api/breeds', async (req, res) => {
    try {
        const type = req.query.type;
        const file = type === 'dog' 
            ? path.join(__dirname, 'DB', 'updated_dog_breeds.csv') 
            : path.join(__dirname, 'DB', 'updated_cat_breeds.csv');
            
        const breeds = await mlPipeline.readCsv(file);
        const breedData = breeds.map(b => ({ id: b.breed_id, name: b.Name })).filter(b => b.name && b.id);
        res.json({ success: true, breeds: breedData });
    } catch (err) { res.status(500).json({ error: 'Server error while fetching breeds' }); }
});

// Admin Route Helper
function sanitizeAgnes(node) {
    if (!node) return null;
    return {
        height: node.height, size: node.size, isLeaf: node.isLeaf,
        children: node.children ? node.children.map(sanitizeAgnes) : []
    };
}

app.get('/api/admin/dashboard', async (req, res) => {
    try {
        const pets = await mlPipeline.readCsv(PETS_CSV);
        const flaggedUsers = pets.filter(p => p.isFlagged === 'true').map(p => ({...p, breed: breedMap[p.breed] || p.breed}));
        const approvedUsers = pets.filter(p => p.isFlagged !== 'true').map(p => ({...p, breed: breedMap[p.breed] || p.breed}));
        const interactions = await mlPipeline.readCsv(INTERACTIONS_CSV);
        
        const agnesTree = mlPipeline.getAgnesTree();
        
        res.json({ 
            success: true, suspicious: flaggedUsers, users: approvedUsers,
            interactions: interactions, agnes: sanitizeAgnes(agnesTree), apriori: globalAprioriRules 
        });
    } catch (err) { res.status(500).json({ error: 'Failed to load dashboard data' }); }
});

app.post('/api/admin/accept', async (req, res) => {
    try {
        const { username } = req.body;
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

app.post('/api/admin/refuse', async (req, res) => {
    try {
        const { username } = req.body;
        let pets = await mlPipeline.readCsv(PETS_CSV);
        pets = pets.filter(p => p.username !== username);
        await mlPipeline.writeCsv(PETS_CSV, petHeaders, pets);
        res.json({ success: true });
    } catch (err) { res.status(500).json({ error: 'Failed to refuse user' }); }
});

app.get('/api/playdates', async (req, res) => {
    try {
        const username = req.query.username;
        if (!username) return res.status(400).json({ error: 'Username required' });
        
        const candidatesRaw = await mlPipeline.getPlaydatesFeed(username);
        const candidates = candidatesRaw.map(c => ({...c, breed: breedMap[c.breed] || c.breed}));
        res.json({ success: true, candidates });
    } catch (err) { console.error(err); res.status(500).json({ error: 'Failed to fetch playdates' }); }
});

app.post('/api/interactions', async (req, res) => {
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

app.post('/api/messages/accept', async (req, res) => {
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

app.get('/api/messages', async (req, res) => {
    try {
        const username = req.query.username;
        if (!username) return res.status(400).json({ error: 'Username required' });
        let messages = await mlPipeline.readCsv(MESSAGES_CSV);
        messages = messages.filter(m => m.toUser === username || m.fromUser === username);
        res.json({ success: true, messages });
    } catch (err) { res.status(500).json({ error: 'Failed to fetch messages' }); }
});

app.get('/api/chat', async (req, res) => {
    try {
        const { userA, userB } = req.query;
        if (!userA || !userB) return res.status(400).json({ error: 'Users required' });
        let chats = await mlPipeline.readCsv(CHAT_CSV);
        chats = chats.filter(c => (c.fromUser === userA && c.toUser === userB) || (c.fromUser === userB && c.toUser === userA));
        res.json({ success: true, chats });
    } catch (err) { res.status(500).json({ error: 'Failed to fetch chat' }); }
});

app.post('/api/chat', async (req, res) => {
    try {
        const { fromUser, toUser, message } = req.body;
        let chats = await mlPipeline.readCsv(CHAT_CSV);
        chats.push({ fromUser, toUser, message, timestamp: Date.now() });
        await mlPipeline.writeCsv(CHAT_CSV, chatHeaders, chats);
        res.json({ success: true });
    } catch (err) { res.status(500).json({ error: 'Failed to send message' }); }
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});