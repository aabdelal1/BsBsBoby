# REST API Reference & Endpoint Design

The backend uses Express.js to expose a strictly RESTful interface. In the recent refactoring, all routes were explicitly designed around standard HTTP verbs (`GET`, `POST`, `PUT`, `PATCH`) to represent CRUD (Create, Read, Update, Delete) operations accurately.

All requests and responses utilize `application/json` format unless otherwise specified (e.g., file uploads).

---

## 1. Authentication (`/api/auth`)

### `POST /api/auth/register`
Creates a new user account.
```javascript
// inside routes/auth.js
const hashedPassword = await bcrypt.hash(password, 10);
users.push({ username, email, phone, location, password: hashedPassword, ... });
```
- **Design Decision:** The system forces `POST` for registration (creation of an entity) and uses `bcrypt` to hash the password before it reaches the CSV file.
- **Request Body:** `{ username, email, phone, location, password }`
- **Response:** `201 Created` on success, `400 Bad Request` if user exists.

### `POST /api/auth/login`
Authenticates a user and retrieves profile data.
- **Design Decision:** Using `POST` for login ensures that the user's password is submitted securely in the request body, not exposed in the URL query string (which a `GET` would do).
- **Request Body:** `{ username, password }`
- **Response:** `200 OK` with user and pet object.

---

## 2. Users & Profiles (`/api/users`)

### `PUT /api/users/:username`
Updates an existing user's profile fields.
```javascript
// inside routes/users.js
router.put('/:username', async (req, res) => {
    const username = req.params.username;
    // ... mutation logic
```
- **Design Decision:** We utilize the `PUT` HTTP method because this action fully replaces/updates the targeted user entity. The target is identified cleanly via the URL path variable `:username` rather than hiding it in the JSON body, which perfectly adheres to REST conventions.
- **Request Body:** `{ phone?, location?, fullName?, photoPath? }`
- **Response:** `200 OK` on success.

### `GET /api/users/:username/playdates`
Retrieves a personalized, ML-sorted feed of potential playdate matches for the specified user.
- **Design Decision:** Following REST resource nesting, `playdates` is considered a sub-resource belonging to a specific `:username`. The response is generated dynamically in memory via the ML pipeline's Fusion Score mechanism.
- **Response:** `{ success: true, candidates: [...] }`

---

## 3. Pets (`/api/pets`)

### `POST /api/pets`
Creates or updates a pet profile.
```javascript
// inside routes/pets.js
const isAnomaly = mlPipeline.gatekeeper(newPetData);
newPetData.isFlagged = isAnomaly ? 'true' : 'false';
```
- **Design Decision:** This endpoint acts as the entry point for the `gatekeeper` algorithm. If a user submits fraudulent data, the backend overrides their `isFlagged` status, returning a `{ flagged: true }` property to instantly notify the frontend UI that their account is pending manual review.
- **Request Body:** `{ username, petName, type, gender, birthYear, vaccination, breed, length, weight, color, personality, photoPath }`
- **Response:** `{ success: true, flagged: boolean, message: string }`

---

## 4. Breeds Dictionary (`/api/breeds`)

### `GET /api/breeds?type={dog|cat}`
Fetches the standardized list of breeds used for the auto-complete registration form.
- **Design Decision:** Instead of hardcoding 500+ breeds in the frontend HTML, the UI fetches this data via query parameters (`?type=dog`). This ensures the UI is always perfectly in sync with the backend database mappings.

---

## 5. Interactions & Messaging

### `POST /api/interactions`
Records a swipe action (like or skip) on another user's pet profile.
```javascript
// inside routes/interactions.js
if (action === 'like') {
    messages.push({ fromUser: username, toUser: targetUsername, status: 'pending' });
}
```
- **Design Decision:** This endpoint serves dual purposes. It logs data for Collaborative Filtering (CF) and seamlessly triggers a pending message connection if the action is a 'like'. This ensures atomicity between logging behavior and initiating communication.
- **Request Body:** `{ username, targetUsername, action: 'like' | 'skip' }`

### `PATCH /api/messages/accept`
Accepts a pending message request, allowing bidirectional chat.
- **Design Decision:** We use `PATCH` here because we are partially modifying an existing resource (changing the `status` string from `pending` to `accepted`) rather than replacing the whole object.

---

## 6. Admin Controls (`/api/admin`)

### `GET /api/admin/dashboard`
Fetches all necessary data to render the Admin Dashboard charts and tables.
- **Design Decision:** In the recent refactor, we deliberately transformed complex server-side data formats (like the Apriori Map object) into flat Arrays directly inside the endpoint. This prevents the frontend `index.html` from having to run complex parsing logic and prevents crashes (like the `TypeError: .sort is not a function` bug).

### `PATCH /api/admin/pets/:username/accept`
Approves a flagged pet profile.
- **Design Decision:** The admin accepts a user by calling this endpoint, which strips the `isFlagged` boolean and forces the `mlPipeline` to immediately recalculate the pet's `clusterGroup`. This immediately injects the previously suspended user into the active matchmaking pool.
