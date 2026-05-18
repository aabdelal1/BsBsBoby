const express = require('express');
const path = require('path');

module.exports = (mlPipeline, PETS_CSV, petHeaders, dbDir) => {
    const router = express.Router();

    router.post('/', async (req, res) => {
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
            
            if (isAnomaly) {
                const path = require('path');
                const SUSPICIOUS_CSV = path.join(dbDir, 'suspicious_profiles.csv');
                
                let reason = 'Anomalous physical dimension ratio flagged by Mahalanobis-Chi2 distance check.';
                const age = new Date().getFullYear() - parseInt(newPetData.birthYear);
                
                if (isNaN(parseFloat(newPetData.weight)) || isNaN(parseInt(newPetData.birthYear)) || isNaN(parseFloat(newPetData.length))) {
                    reason = 'Non-numeric physical values submitted.';
                } else if (age > 25) {
                    reason = `Age anomaly (pet age of ${age} years is realistically too high).`;
                } else if ((newPetData.type || '').toLowerCase() === 'cat' && parseFloat(newPetData.weight) > 40) {
                    reason = `Unrealistic weight submitted for a cat (${newPetData.weight} kg).`;
                } else if (['spam', 'fake', 'test'].includes((newPetData.vaccination || '').toLowerCase())) {
                    reason = `Spam/fake vaccination status keyword entered.`;
                } else if (parseFloat(newPetData.weight) > 150) {
                    reason = `Unrealistic physical dimensions submitted (weight of ${newPetData.weight} kg is too high).`;
                }

                const suspiciousHeaders = [
                    { id: 'username', title: 'username' }, { id: 'petName', title: 'petName' },
                    { id: 'type', title: 'type' }, { id: 'gender', title: 'gender' },
                    { id: 'birthYear', title: 'birthYear' }, { id: 'vaccination', title: 'vaccination' },
                    { id: 'breed', title: 'breed' }, { id: 'length', title: 'length' },
                    { id: 'weight', title: 'weight' }, { id: 'color', title: 'color' },
                    { id: 'personality', title: 'personality' }, { id: 'photoPath', title: 'photoPath' },
                    { id: 'reason', title: 'reason' }
                ];

                let suspiciousList = [];
                try {
                    suspiciousList = await mlPipeline.readCsv(SUSPICIOUS_CSV);
                } catch (e) {}

                suspiciousList = suspiciousList.filter(p => p.username !== username);
                suspiciousList.push({
                    username: newPetData.username,
                    petName: newPetData.petName,
                    type: newPetData.type,
                    gender: newPetData.gender,
                    birthYear: newPetData.birthYear,
                    vaccination: newPetData.vaccination,
                    breed: newPetData.breed,
                    length: newPetData.length,
                    weight: newPetData.weight,
                    color: newPetData.color,
                    personality: newPetData.personality,
                    photoPath: newPetData.photoPath,
                    reason: reason
                });
                await mlPipeline.writeCsv(SUSPICIOUS_CSV, suspiciousHeaders, suspiciousList);

                return res.json({ success: true, flagged: true, message: 'Account creation pending review' });
            }
            res.json({ success: true, message: 'Pet profile saved' });

        } catch (err) { console.error("Pet Profile Error:", err); res.status(500).json({ error: 'Server error while saving pet profile' }); }
    });

    router.get('/search', async (req, res) => {
        try {
            let query = (req.query.q || '').toLowerCase().trim();
            // Handle common typos gracefully
            if (query.includes('sheperd')) {
                query = query.replace('sheperd', 'shepherd');
            }

            const pets = await mlPipeline.readCsv(PETS_CSV);
            
            // Build breed ID to name lookup on the fly
            const breedMap = {};
            try {
                const dogBreeds = await mlPipeline.readCsv(path.join(dbDir, 'updated_dog_breeds.csv'));
                const catBreeds = await mlPipeline.readCsv(path.join(dbDir, 'updated_cat_breeds.csv'));
                
                dogBreeds.forEach(b => {
                    if (b.breed_id && b.Name) {
                        breedMap[b.breed_id.toLowerCase()] = b.Name.toLowerCase();
                    }
                });
                catBreeds.forEach(b => {
                    if (b.breed_id && b.Name) {
                        breedMap[b.breed_id.toLowerCase()] = b.Name.toLowerCase();
                    }
                });
            } catch (e) {
                console.error("Failed to load breed lookup map in backend search:", e);
            }

            const results = pets.filter(p => {
                const petName = (p.petName || '').toLowerCase();
                const type = (p.type || '').toLowerCase();
                const breedId = (p.breed || '').toLowerCase();
                
                const breedName = breedMap[breedId] || '';
                const normalizedBreedId = breedId.replace(/[_-]/g, ' ');

                return petName.includes(query) ||
                       type.includes(query) ||
                       breedId.includes(query) ||
                       normalizedBreedId.includes(query) ||
                       breedName.includes(query);
            });
            
            res.json({ success: true, results });
        } catch (err) { 
            console.error("Search failed:", err);
            res.status(500).json({ error: 'Search failed' }); 
        }
    });

    router.post('/classify-breed', async (req, res) => {
        try {
            const { filePath } = req.body;
            if (!filePath) return res.status(400).json({ error: 'filePath is required' });

            const absolutePath = path.join(process.cwd(), filePath.replace(/^\//, ''));
            
            const fs = require('fs');
            if (!fs.existsSync(absolutePath)) {
                return res.status(400).json({ error: `Image file not found on server: ${filePath}` });
            }

            const { spawn } = require('child_process');
            const scriptPath = path.join(process.cwd(), 'classify_onnx.py');
            
            const pythonProcess = spawn('python', [scriptPath, absolutePath]);
            
            let stdoutData = '';
            let stderrData = '';
            
            pythonProcess.stdout.on('data', (data) => {
                stdoutData += data.toString();
            });
            
            pythonProcess.stderr.on('data', (data) => {
                stderrData += data.toString();
            });
            
            pythonProcess.on('close', (code) => {
                if (code !== 0) {
                    console.error("Python classifier failed with code:", code, stderrData);
                    return res.status(500).json({ error: 'Classifier execution failed', details: stderrData });
                }
                
                try {
                    const result = JSON.parse(stdoutData.trim());
                    if (result.success) {
                        res.json({
                            success: true,
                            prediction: result.prediction,
                            confidence: result.confidence
                        });
                    } else {
                        res.status(500).json({ error: result.error || 'Classification failed' });
                    }
                } catch (e) {
                    console.error("Failed to parse Python output:", stdoutData, e);
                    res.status(500).json({ error: 'Failed to parse classifier output', raw: stdoutData });
                }
            });
            
        } catch (err) {
            console.error("Classify route error:", err);
            res.status(500).json({ error: 'Server error during breed classification' });
        }
    });

    router.get('/:username', async (req, res) => {
        try {
            const username = req.params.username;
            const pets = await mlPipeline.readCsv(PETS_CSV);
            const pet = pets.find(p => p.username === username);
            if (!pet) return res.status(404).json({ error: 'Pet not found' });
            res.json({ success: true, pet });
        } catch (err) { res.status(500).json({ error: 'Server error' }); }
    });

    return router;
};
