# System Architecture in Detail

The Pet Matchmaker application is a lightweight, full-stack JavaScript application designed to provide advanced machine learning recommendations without relying on heavy external database dependencies (like PostgreSQL or MongoDB). Instead, it utilizes an **in-memory compute architecture** backed by persistent CSV storage, allowing for rapid prototyping and deployment.

## 1. Directory Structure & Routing Strategy

With the recent RESTful refactoring, the monolithic `server.js` was split into domain-specific modules.

```javascript
// server.js (Mounting Routers)
app.use('/api/auth', authRoutes);
app.use('/api/users', userRoutes);
app.use('/api/pets', petRoutes);
app.use('/api/admin', adminRoutes);
// ...
```
**Design Decision:** By modularizing the backend, we adhere to the Single Responsibility Principle (SRP). `server.js` now acts solely as the application bootstrap, responsible for initializing the background ML interval and binding the Express middleware. This makes the codebase vastly easier to test and prevents Git merge conflicts when multiple developers work on different features (e.g., one on `admin.js`, another on `pets.js`).

## 2. In-Memory ML with CSV Persistence

Because machine learning algorithms (like AGNES and K-Means) require reading the entire dataset to construct their models, performing a database I/O read on every single `/api/playdates` request would be disastrous for performance.

```javascript
// Background loop in server.js
mlPipeline.loadStateFromLocal().then(() => {
    setInterval(runBackgroundTasks, 300000); // 5 minutes
    setTimeout(runBackgroundTasks, 2000);    // Initial boot
});
```
**Design Decision:** We load the CSV data into memory, and the models are trained asynchronously in the background every 5 minutes. The resulting model states (centroids, dendrograms) are serialized to `DB/models/ml_pipeline_state.json`. When a user requests their feed, the system calculates their matches against the *in-memory* state in milliseconds, completely bypassing disk I/O.

## 3. Frontend Architecture (Vanilla SPA)

The frontend is contained entirely within `index.html`, utilizing standard DOM manipulation.

```javascript
// Example from index.html
async function loadPlaydates() {
  const response = await fetch(`/api/users/${currentUser.username}/playdates`);
  const data = await response.json();
  // ... DOM manipulation
}
```
**Design Decision:** We deliberately avoided heavy frameworks (React/Vue) to maintain an extremely lightweight footprint. By making `fetch()` calls to strict RESTful URLs and swapping container visibility (`display: none` to `display: block`), the app behaves like a modern Single Page Application (SPA) without the need for a bundler (Webpack/Vite).

## 4. Normalization and Data Integrity

The system enforces data integrity through lookup maps rather than saving raw strings.

```javascript
// loading in server.js
const dogs = await mlPipeline.readCsv(path.join(DB_DIR, 'updated_dog_breeds.csv'));
dogs.forEach(d => { if(d.breed_id) breedMap[d.breed_id] = d.Name; });
```
**Design Decision:** In the `individual_pets.csv`, a user's breed is stored as a normalized key (e.g., `CAT_001` or `DOG_042`) rather than raw user-input text like "Golden Retriever". The server maintains a global `breedMap`. When an endpoint returns user data, it dynamically swaps the `breed_id` back into the human-readable `Name`. This enforces strict case-insensitive validation on the frontend and ensures the machine learning pipeline is analyzing standardized, clean data.

## 5. Security & Authentication

```javascript
// routes/auth.js
const hashedPassword = await bcrypt.hash(password, 10);
users.push({ username, email, password: hashedPassword, ... });
```
**Design Decision:** Passwords are never stored in plaintext. We utilize `bcrypt` with a salt round of 10 to hash the passwords before committing them to `users.csv`. Currently, the frontend manages "session state" via a global `currentUser` variable; while this is sufficient for a local application, moving to a production environment would require migrating to JSON Web Tokens (JWT) stored in HttpOnly cookies.
