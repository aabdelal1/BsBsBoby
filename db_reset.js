const fs = require('fs');
const path = require('path');

const DB_DIR = path.join(__dirname, 'DB');
const UPLOADS_DIR = path.join(__dirname, 'uploads');
const ANIMALS_DIR = path.join(__dirname, 'Animals');

// Ensure uploads directory exists
if (!fs.existsSync(UPLOADS_DIR)) {
    fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

// Precomputed Bcrypt hash of "123" with 10 rounds
const BCRYPT_HASH_123 = '$2b$10$h.sUEhY8JN4qol4jw3Z/CO8KUg568fsG9bD6WL.iP2byYmtv2vpoC';

// Cat breeds folder list to distinguish species
const CAT_BREEDS = [
    'abyssinian', 'bengal', 'birman', 'bombay', 'british_shorthair',
    'egyptian_mau', 'maine_coon', 'persian', 'ragdoll', 'russian_blue',
    'siamese', 'sphynx'
];

// Breed ID mapping lookup - exactly matching IDs in updated_cat_breeds.csv and updated_dog_breeds.csv
const BREED_MAPPING = {
    'abyssinian': 'CAT_001',
    'bengal': 'CAT_011',
    'birman': 'CAT_012',
    'bombay': 'CAT_013',
    'british_shorthair': 'CAT_016',
    'maine_coon': 'CAT_036',
    'persian': 'CAT_041',
    'ragdoll': 'CAT_044',
    'russian_blue': 'CAT_045',
    'siamese': 'CAT_049',
    'sphynx': 'CAT_055',
    'american_bulldog': 'DOG_030',
    'basset_hound': 'DOG_024',
    'beagle': 'DOG_009',
    'boxer': 'DOG_005',
    'chihuahua': 'DOG_031',
    'english_cocker_spaniel': 'DOG_018',
    'great_pyrenees': 'DOG_019',
    'havanese': 'DOG_055',
    'japanese_chin': 'DOG_056',
    'keeshond': 'DOG_057',
    'miniature_pinscher': 'DOG_090',
    'newfoundland': 'DOG_023',
    'pomeranian': 'DOG_013',
    'pug': 'DOG_026',
    'samoyed': 'DOG_076',
    'shiba_inu': 'DOG_043',
    'staffordshire_bull_terrier': 'DOG_092',
    'yorkshire_terrier': 'DOG_096'
};

const ALL_TRAITS = ['active', 'friendly', 'calm', 'touchy', 'sleepy'];

// Authentic Human First & Last Names lists (Egyptian/international mix)
const MALE_HUMAN_NAMES = [
    'Adam', 'Omar', 'Youssef', 'Ali', 'Mostafa', 'Kareem', 'Ahmed', 'Tarek', 'Sherif', 'Hassan',
    'Hussein', 'Amr', 'Mohamed', 'Ziad', 'Marwan', 'Khaled', 'Ramy', 'Waleed', 'Hany', 'Seif',
    'Moustafa', 'Adham', 'Hazem', 'Shady', 'Wael', 'Hisham', 'Ibrahim', 'Mahmoud', 'Samy', 'Raafat'
];

const FEMALE_HUMAN_NAMES = [
    'Farida', 'Layla', 'Malak', 'Mariam', 'Nour', 'Yasmin', 'Jana', 'Salma', 'Hana', 'Maya',
    'Lara', 'Sherine', 'Zeina', 'Nadine', 'Dina', 'Habiba', 'Sara', 'Reem', 'Aya', 'Ghada',
    'Nouran', 'Heba', 'Mona', 'Radwa', 'Sohaila', 'May', 'Ola', 'Mai', 'Menna', 'Rowan'
];

const HUMAN_LAST_NAMES = [
    'El-Masry', 'Mansour', 'Hassan', 'Salem', 'Ibrahim', 'Abdel-Rahman', 'Shahin', 'Farahat', 'Ghanem', 'Osman',
    'Badawy', 'Amer', 'Zaki', 'Soliman', 'Fayed', 'Rashed', 'El-Sayed', 'Khalil', 'Nofal', 'Mostafa',
    'El-Demerdash', 'Shaheen', 'Abou-Taleb', 'El-Ghandour', 'El-Feky', 'El-Banhawy', 'Abdel-Gawad', 'El-Haddad', 'Abdel-Hamid', 'El-Khouly'
];

// Authentic Pet Names lists
const MALE_PET_NAMES = [
    'Max', 'Charlie', 'Cooper', 'Rocky', 'Buddy', 'Leo', 'Milo', 'Jack', 'Oliver', 'Toby',
    'Teddy', 'Bentley', 'Duke', 'Bear', 'Simba', 'Loki', 'Oscar', 'Buster', 'Sam', 'Ziggy',
    'Winston', 'Murphy', 'Gizmo', 'Rex', 'Zeus', 'Coco', 'Shadow', 'Jasper', 'Rusty', 'Otis'
];

const FEMALE_PET_NAMES = [
    'Bella', 'Luna', 'Lucy', 'Daisy', 'Lola', 'Sadie', 'Molly', 'Stella', 'Chloe', 'Maggie',
    'Sophie', 'Lily', 'Coco', 'Ruby', 'Rosie', 'Zoe', 'Mia', 'Penny', 'Nala', 'Angel',
    'Lulu', 'Ginger', 'Roxy', 'Abby', 'Harley', 'Cleo', 'Sasha', 'Gracie', 'Hazel', 'Belle'
];

function generatePersonality() {
    const shuffled = ALL_TRAITS.slice().sort(() => 0.5 - Math.random());
    const count = 2 + Math.floor(Math.random() * 2); // Select 2 or 3 traits
    return shuffled.slice(0, count).join(',');
}

function normalizeBreedName(name) {
    return name.split(/[_-]/).map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

function moveFile(src, dest) {
    try {
        fs.renameSync(src, dest);
    } catch (err) {
        // Fallback for cross-device or permission errors
        fs.copyFileSync(src, dest);
        fs.unlinkSync(src);
    }
}

async function run() {
    console.log("=== STARTING DATABASE RESET & PHOTO IMPORT ===");

    // 1. Define files and headers
    const filesToReset = {
        'users.csv': 'username,email,phone,location,password,fullName,photoPath,isBlocked\n',
        'individual_pets.csv': 'username,petName,type,gender,birthYear,vaccination,breed,length,weight,color,personality,photoPath,isFlagged,clusterGroup\n',
        'chat.csv': 'fromUser,toUser,message,timestamp\n',
        'interactions.csv': 'username,targetUsername,action,timestamp\n',
        'messages.csv': 'fromUser,toUser,status\n',
        'suspicious_profiles.csv': 'username,petName,type,gender,birthYear,vaccination,breed,length,weight,color,personality,photoPath,reason\n'
    };

    // 2. Clear all databases and preserve admin user
    for (const [filename, header] of Object.entries(filesToReset)) {
        const filePath = path.join(DB_DIR, filename);
        if (filename === 'users.csv') {
            const adminRow = 'admin,admin@bsbsboby.local,+00000000000,Admin HQ,$2b$10$rsvlhfTeLtX7.spqkMscdeRbOCmseo5GT7eyVXgSdaw3J4nRwwc/q,Admin User,,false\n';
            fs.writeFileSync(filePath, header + adminRow, 'utf8');
            console.log(`Cleared ${filename} and preserved admin user.`);
        } else {
            fs.writeFileSync(filePath, header, 'utf8');
            console.log(`Cleared ${filename}.`);
        }
    }

    // 3. Scan the Animals folder
    if (!fs.existsSync(ANIMALS_DIR)) {
        console.error(`Animals directory not found at ${ANIMALS_DIR}!`);
        process.exit(1);
    }

    const breedDirs = fs.readdirSync(ANIMALS_DIR).filter(item => {
        return fs.statSync(path.join(ANIMALS_DIR, item)).isDirectory();
    });

    console.log(`Found ${breedDirs.length} breed folders in Animals directory.`);

    let totalImported = 0;
    const usersRows = [];
    const petsRows = [];

    // Constants for alternating mock details
    const locations = ['Cairo', 'Giza', 'Alexandria', 'Mansoura', 'Tanta'];
    const colors = ['brown', 'white', 'black', 'golden', 'grey', 'orange'];
    const vaccinations = ['fully vaccinated', 'partially vaccinated', 'none'];
    const genders = ['male', 'female'];

    for (const breedDir of breedDirs) {
        const breedKey = breedDir.toLowerCase().trim();
        const breedId = BREED_MAPPING[breedKey];

        // Skip breed if it doesn't directly map to an ID in catalog
        if (!breedId) {
            console.log(`Skipping breed [${normalizeBreedName(breedDir)}] - no matching catalog ID found.`);
            continue;
        }

        const isCat = CAT_BREEDS.includes(breedKey);
        const type = isCat ? 'cat' : 'dog';

        const breedPath = path.join(ANIMALS_DIR, breedDir);
        const files = fs.readdirSync(breedPath).filter(file => {
            const ext = path.extname(file).toLowerCase();
            return ['.jpg', '.jpeg', '.png'].includes(ext);
        });

        // We select exactly 5 images from each breed directory
        const filesToUse = files.slice(0, 5);
        console.log(`Breed [${normalizeBreedName(breedDir)}] (${type}): moving ${filesToUse.length} images.`);

        filesToUse.forEach((file, index) => {
            const src = path.join(breedPath, file);
            const dest = path.join(UPLOADS_DIR, file);

            // Move the photo assets selectively as requested
            moveFile(src, dest);

            // Determine Owner Gender & Real Name details
            const ownerGender = totalImported % 2 === 0 ? 'male' : 'female';
            const firstName = ownerGender === 'male'
                ? MALE_HUMAN_NAMES[totalImported % MALE_HUMAN_NAMES.length]
                : FEMALE_HUMAN_NAMES[totalImported % FEMALE_HUMAN_NAMES.length];
            const lastName = HUMAN_LAST_NAMES[(totalImported * 3) % HUMAN_LAST_NAMES.length];
            const fullName = `${firstName} ${lastName}`;
            
            // Unique, clean username and email
            const username = `${firstName.toLowerCase()}_${lastName.toLowerCase()}_${totalImported + 1}`;
            const email = `${firstName.toLowerCase()}.${lastName.toLowerCase()}_${totalImported + 1}@bsbsboby.local`;

            const phone = `+2010${Math.floor(10000000 + Math.random() * 90000000)}`;
            const location = locations[totalImported % locations.length];
            const photoPath = `/uploads/${file}`;

            // Determine Pet Gender & Real Name details
            const gender = genders[totalImported % genders.length];
            const petName = gender === 'male'
                ? MALE_PET_NAMES[totalImported % MALE_PET_NAMES.length]
                : FEMALE_PET_NAMES[totalImported % FEMALE_PET_NAMES.length];

            const birthYear = 2018 + (totalImported % 7); // Years 2018 to 2024
            const vaccination = vaccinations[totalImported % vaccinations.length];
            const color = colors[totalImported % colors.length];
            const personality = generatePersonality();

            // Set realistic physical dimensions to safely pass Mahalanobis checks
            let length, weight;
            if (isCat) {
                // Cats average 4-5 kg, 28-32 cm
                weight = (3.5 + (totalImported % 3) * 0.7).toFixed(1);
                length = (26 + (totalImported % 5) * 2).toFixed(0);
            } else {
                // Dogs average 15-25 kg, 45-55 cm
                weight = (12 + (totalImported % 5) * 3.5).toFixed(1);
                length = (42 + (totalImported % 6) * 3).toFixed(0);
            }

            // Construct user CSV row (username,email,phone,location,password,fullName,photoPath,isBlocked)
            const userRow = `${username},${email},${phone},"${location}",${BCRYPT_HASH_123},"${fullName}",${photoPath},false`;
            usersRows.push(userRow);

            // Construct pet CSV row (username,petName,type,gender,birthYear,vaccination,breed,length,weight,color,personality,photoPath,isFlagged,clusterGroup)
            const petRow = `${username},"${petName}",${type},${gender},${birthYear},${vaccination},"${breedId}",${length},${weight},${color},"${personality}",${photoPath},false,`;
            petsRows.push(petRow);

            totalImported++;
        });
    }

    // Write all new user profiles to users.csv
    const usersCsvPath = path.join(DB_DIR, 'users.csv');
    fs.appendFileSync(usersCsvPath, usersRows.join('\n') + '\n', 'utf8');

    // Write all new pet profiles to individual_pets.csv
    const petsCsvPath = path.join(DB_DIR, 'individual_pets.csv');
    fs.appendFileSync(petsCsvPath, petsRows.join('\n') + '\n', 'utf8');

    console.log(`\n=== IMPORT COMPLETE ===`);
    console.log(`Successfully migrated ${totalImported} photos into 'uploads/'.`);
    console.log(`Created ${usersRows.length} user profiles in DB/users.csv.`);
    console.log(`Created ${petsRows.length} pet profiles in DB/individual_pets.csv.`);
}

run().catch(console.error);
