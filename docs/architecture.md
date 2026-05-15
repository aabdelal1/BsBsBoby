# Architecture Overview

The Pet Matchmaker application is a lightweight, full-stack JavaScript application designed to provide advanced machine learning recommendations without relying on heavy external database dependencies like PostgreSQL or MongoDB. Instead, it utilizes an in-memory compute architecture backed by persistent CSV storage.

## System Components

### 1. Frontend (`index.html`)
The frontend is a single-page application (SPA) built purely with HTML, CSS, and Vanilla JavaScript. 
- **DOM Manipulation:** Uses standard JavaScript to toggle views (e.g., login, dashboard, chat) without full page reloads.
- **REST Integration:** Communicates with the backend exclusively through `fetch()` API calls to RESTful endpoints.
- **Data Visualization:** Integrates **Chart.js** to render real-time insights in the Admin Dashboard (Scatter Plots for clusters, Bar Charts for Apriori Rules, and Line Charts for AGNES Merges).

### 2. Backend Server (`server.js` and `routes/`)
The backend is powered by Node.js and Express. It follows a strictly modular RESTful architecture:
- `routes/auth.js`: Handles secure user registration and password validation using `bcrypt`.
- `routes/pets.js`: Manages the core pet profile creation, including triggering the ML gatekeeper and clustering logic.
- `routes/admin.js`: Exposes data for the administrative dashboard and allows for profile moderation.
- `routes/interactions.js` & `routes/messages.js` & `routes/chats.js`: Manage the matching, connection, and real-time messaging pipeline.
- `routes/breeds.js`: Provides lookup tables for dynamic breed auto-completion.

### 3. Machine Learning Pipeline (`ml_pipeline.js`)
This is the core computational engine of the app. It handles heavy background tasks asynchronously to ensure the REST API remains responsive. (See `ml_pipeline.md` for detailed algorithms).

### 4. Database Storage (`DB/`)
Data is stored locally using CSV files to maintain simplicity and portability:
- `users.csv`: Stores hashed user credentials and metadata.
- `individual_pets.csv`: Stores normalized pet data. Breed names are stored as normalized IDs (e.g., `CAT_001`).
- `interactions.csv`: Logs all swipes (likes/skips).
- `messages.csv` & `chat.csv`: Handle match states and dialogue logs.
- `models/ml_pipeline_state.json`: Maintains the state of the trained ML models across server restarts.

## Data Flow
1. **Client Request:** User actions (e.g., swiping a pet) trigger a `fetch()` request to the Express server.
2. **Route Handling:** The Express Router receives the request, validates the payload, and delegates complex operations to the `ml_pipeline`.
3. **Data Mutation:** The state is updated in memory and atomically written to the respective CSV files.
4. **Response:** The server returns a structured JSON payload to the client.
5. **Background Sync:** The `ml_pipeline` periodically awakens (every 5 minutes) to recalculate clusters and matching models based on the new data, saving the resulting state to disk.
