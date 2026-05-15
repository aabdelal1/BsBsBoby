# REST API Reference

The backend exposes a strictly RESTful interface. All requests and responses utilize `application/json` format unless otherwise specified (e.g., file uploads).

## Authentication (`/api/auth`)

### `POST /api/auth/register`
Creates a new user account.
- **Body:** `{ username, email, phone, location, password }`
- **Response:** `201 Created` on success, `400 Bad Request` if user exists.

### `POST /api/auth/login`
Authenticates a user and returns their profile and pet data (if registered).
- **Body:** `{ username, password }`
- **Response:** `200 OK` with user and pet object.

## Users (`/api/users`)

### `PUT /api/users/:username`
Updates user profile fields.
- **Body:** `{ phone?, location?, fullName?, photoPath? }`
- **Response:** `200 OK` on success.

### `GET /api/users/:username/playdates`
Retrieves a personalized, ML-sorted feed of potential playdate matches for the specified user.
- **Response:** `{ success: true, candidates: [...] }`

## Pets (`/api/pets`)

### `POST /api/pets`
Creates or updates a pet profile. The profile is automatically processed by the ML Gatekeeper to flag suspicious attributes.
- **Body:** `{ username, petName, type, gender, birthYear, vaccination, breed, length, weight, color, personality, photoPath }`
- **Response:** `{ success: true, flagged: boolean, message: string }`

## Breeds (`/api/breeds`)

### `GET /api/breeds?type={dog|cat}`
Fetches the standardized list of breeds used for the auto-complete registration form.
- **Response:** `{ success: true, breeds: [{ id, name }, ...] }`

## Interactions (`/api/interactions`)

### `POST /api/interactions`
Records a swipe action (like or skip) on another user's pet profile. If a 'like' is registered, a pending message is automatically created.
- **Body:** `{ username, targetUsername, action: 'like' | 'skip' }`

## Messages (`/api/messages`)

### `GET /api/messages?username={username}`
Fetches all message connections (both pending and accepted) for a user.

### `PATCH /api/messages/accept`
Accepts a pending message request, allowing bidirectional chat.
- **Body:** `{ fromUser, toUser }`

## Chats (`/api/chats`)

### `GET /api/chats?userA={userA}&userB={userB}`
Retrieves the chat history between two connected users.

### `POST /api/chats`
Sends a new text message.
- **Body:** `{ fromUser, toUser, message }`

## Admin (`/api/admin`)

### `GET /api/admin/dashboard`
Fetches all necessary data to render the Admin Dashboard charts and tables.
- **Response:** Includes `suspicious` (flagged users array), `users` (clean users array), `interactions`, `kmeans`, `agnes`, and `apriori`.

### `PATCH /api/admin/pets/:username/accept`
Approves a flagged pet profile, unhiding it from the public matching pool and recalculating its ML cluster.

### `PATCH /api/admin/pets/:username/refuse`
Permanently deletes a flagged pet profile from the database.
