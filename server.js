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
const runBackgroundTasks = async () => {
    try {
        await mlPipeline.trainBackgroundModels();
        await mlPipeline.runApriori(); 
        console.log("Background Pipeline (Fisher-Yates RF, Local State, Hamming AGNES) trained successfully.");
    } catch (err) { console.error("Background task error:", err); }
};

mlPipeline.loadStateFromLocal().then(() => {
    setInterval(runBackgroundTasks, 300000); 
    setTimeout(runBackgroundTasks, 2000);
});

const upload = multer({
    storage: multer.diskStorage({
        destination: (req, file, cb) => cb(null, 'uploads/'),
        filename: (req, file, cb) => cb(null, Date.now() + path.extname(file.originalname))
    })
});

// --- ROUTERS ---
const authRoutes = require('./routes/auth')(mlPipeline, USERS_CSV, PETS_CSV, userHeaders, breedMap);
const userRoutes = require('./routes/users')(mlPipeline, USERS_CSV, userHeaders, breedMap);
const petRoutes = require('./routes/pets')(mlPipeline, PETS_CSV, petHeaders, DB_DIR);
const breedRoutes = require('./routes/breeds')(mlPipeline, DB_DIR);
const adminRoutes = require('./routes/admin')(mlPipeline, PETS_CSV, petHeaders, INTERACTIONS_CSV, breedMap);
const interactionRoutes = require('./routes/interactions')(mlPipeline, INTERACTIONS_CSV, interactionHeaders, MESSAGES_CSV, messageHeaders);
const messageRoutes = require('./routes/messages')(mlPipeline, MESSAGES_CSV, messageHeaders, USERS_CSV, PETS_CSV);
const chatRoutes = require('./routes/chats')(mlPipeline, CHAT_CSV, chatHeaders);

// --- MOUNT ROUTES ---
app.use('/api/auth', authRoutes);
app.use('/api/users', userRoutes);
app.use('/api/pets', petRoutes);
app.use('/api/breeds', breedRoutes);
app.use('/api/admin', adminRoutes);
app.use('/api/interactions', interactionRoutes);
app.use('/api/messages', messageRoutes);
app.use('/api/chats', chatRoutes);

// File Upload Route (keep simple)
app.post('/api/upload', upload.single('media'), (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
    res.json({ success: true, filePath: `/uploads/${req.file.filename}` });
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});