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

// Ensure directories exist
if (!fs.existsSync(DB_DIR)) fs.mkdirSync(DB_DIR);
if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR);

// Ensure CSV files with headers exist
const userHeaders = [
    { id: 'username', title: 'username' },
    { id: 'email', title: 'email' },
    { id: 'phone', title: 'phone' },
    { id: 'location', title: 'location' },
    { id: 'password', title: 'password' },
    { id: 'fullName', title: 'fullName' },
    { id: 'photoPath', title: 'photoPath' }
];

const petHeaders = [
    { id: 'username', title: 'username' },
    { id: 'petName', title: 'petName' },
    { id: 'type', title: 'type' },
    { id: 'gender', title: 'gender' },
    { id: 'birthYear', title: 'birthYear' },
    { id: 'vaccination', title: 'vaccination' },
    { id: 'breed', title: 'breed' },
    { id: 'length', title: 'length' },
    { id: 'weight', title: 'weight' },
    { id: 'color', title: 'color' },
    { id: 'personality', title: 'personality' },
    { id: 'photoPath', title: 'photoPath' }
];

if (!fs.existsSync(USERS_CSV)) {
    const csvWriter = createCsvWriter({ path: USERS_CSV, header: userHeaders });
    csvWriter.writeRecords([]); // Creates the file with headers
}

if (!fs.existsSync(PETS_CSV)) {
    const csvWriter = createCsvWriter({ path: PETS_CSV, header: petHeaders });
    csvWriter.writeRecords([]);
}

// Multer setup for uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname));
    }
});
const upload = multer({ storage: storage });

// Helper to read CSV
const readCsv = (filePath) => {
    return new Promise((resolve, reject) => {
        const results = [];
        if (!fs.existsSync(filePath)) {
            resolve(results);
            return;
        }
        const readStream = fs.createReadStream(filePath);
        
        readStream
            .pipe(csv())
            .on('data', (data) => results.push(data))
            .on('error', (err) => {
                readStream.destroy();
                reject(err);
            });
            
        readStream.on('close', () => {
            resolve(results);
        });
    });
};

// Helper to write CSV
const writeCsv = (filePath, headers, records) => {
    const csvWriter = createCsvWriter({ path: filePath, header: headers });
    return csvWriter.writeRecords(records);
};

// Routes

// Signup
app.post('/api/signup', async (req, res) => {
    try {
        const { username, email, phone, location, password } = req.body;
        
        if (!username || !email || !phone || !location || !password) {
            return res.status(400).json({ error: 'All fields are mandatory' });
        }

        const users = await readCsv(USERS_CSV);
        if (users.find(u => u.username === username)) {
            return res.status(400).json({ error: 'Username already exists' });
        }

        const saltRounds = 10;
        const hashedPassword = await bcrypt.hash(password, saltRounds);

        users.push({
            username,
            email,
            phone,
            location,
            password: hashedPassword,
            fullName: '',
            photoPath: ''
        });

        await writeCsv(USERS_CSV, userHeaders, users);
        res.status(201).json({ success: true, message: 'User created' });
    } catch (err) {
        console.error("Signup error:", err);
        res.status(500).json({ error: 'Server error' });
    }
});

// Login
app.post('/api/login', async (req, res) => {
    try {
        const { username, password } = req.body;
        const users = await readCsv(USERS_CSV);
        const user = users.find(u => u.username === username);

        if (!user) {
            return res.status(401).json({ error: 'Invalid username or password' });
        }

        const match = await bcrypt.compare(password, user.password);
        if (match) {
            // Also return the user profile data
            const pets = await readCsv(PETS_CSV);
            const pet = pets.find(p => p.username === username) || null;
            
            res.json({ success: true, message: 'Login successful', user: { username: user.username, email: user.email, phone: user.phone, location: user.location, fullName: user.fullName, photoPath: user.photoPath }, pet });
        } else {
            res.status(401).json({ error: 'Invalid username or password' });
        }
    } catch (err) {
        console.error("Signup error:", err);
        res.status(500).json({ error: 'Server error' });
    }
});

// Forgot Password (Placeholder)
app.post('/api/forgot-password', async (req, res) => {
    const { username } = req.body;
    const users = await readCsv(USERS_CSV);
    const user = users.find(u => u.username === username);

    if (user) {
        res.json({ success: true, message: 'Password recovery email sent' });
    } else {
        res.status(404).json({ error: 'Username not found' });
    }
});

// Update User Profile
app.put('/api/user/:username', async (req, res) => {
    try {
        const username = req.params.username;
        const { phone, location, fullName, photoPath } = req.body;
        
        let users = await readCsv(USERS_CSV);
        let userIndex = users.findIndex(u => u.username === username);

        if (userIndex === -1) {
            return res.status(404).json({ error: 'User not found' });
        }

        // Email and password remain unchanged
        if (phone !== undefined) users[userIndex].phone = phone;
        if (location !== undefined) users[userIndex].location = location;
        if (fullName !== undefined) users[userIndex].fullName = fullName;
        if (photoPath !== undefined) users[userIndex].photoPath = photoPath;

        await writeCsv(USERS_CSV, userHeaders, users);
        res.json({ success: true, message: 'Profile updated' });
    } catch (err) {
        console.error("Signup error:", err);
        res.status(500).json({ error: 'Server error' });
    }
});

// Create/Update Pet Profile
app.post('/api/pet', async (req, res) => {
    try {
        const { username, petName, type, gender, birthYear, vaccination, breed, length, weight, color, personality, photoPath } = req.body;

        if (!username) {
             return res.status(400).json({ error: 'Username is required' });
        }

        let pets = await readCsv(PETS_CSV);
        let petIndex = pets.findIndex(p => p.username === username);

        const newPetData = {
            username,
            petName: petName || '',
            type: type || '',
            gender: gender || '',
            birthYear: birthYear || '',
            vaccination: vaccination || '',
            breed: breed || '',
            length: length || '',
            weight: weight || '',
            color: color || '',
            personality: personality || '',
            photoPath: photoPath || ''
        };

        if (petIndex > -1) {
            // Update
            pets[petIndex] = { ...pets[petIndex], ...newPetData };
        } else {
            // Create
            pets.push(newPetData);
        }

        await writeCsv(PETS_CSV, petHeaders, pets);
        res.json({ success: true, message: 'Pet profile saved' });

    } catch (err) {
        console.error("Signup error:", err);
        res.status(500).json({ error: 'Server error' });
    }
});

// Media Uploads
app.post('/api/upload', upload.single('media'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }
    const filePath = `/uploads/${req.file.filename}`;
    res.json({ success: true, filePath });
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
