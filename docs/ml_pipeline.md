# Machine Learning Pipeline in Detail

The Pet Matchmaker system relies on a multi-layered machine learning pipeline (`ml_pipeline.js`). The core design philosophy here is **asynchronous, non-blocking compute**—performing heavy ML tasks (Clustering, Random Forest, Apriori) in the background so that the REST API response times remain instantaneous for the user.

## 1. Feature Extraction & Standardization

Before any distance metrics or algorithms can be applied, the raw string/number data from the CSV files must be transformed into a standardized numerical format. 

### Behavioral Encoding (One-Hot)
```javascript
function extractBehavioral(pet) {
    const traits = (pet.personality || '').toLowerCase().split(',').map(t => t.trim());
    return ['active', 'friendly', 'calm', 'touchy', 'sleepy'].map(t => traits.includes(t) ? 1 : 0);
}
```
**Design Decision:** We use one-hot encoding for behavioral traits to allow for exact distance calculations (Hamming Distance). Because traits are categorical and not ordinal, assigning them a simple 1 or 0 ensures the algorithm doesn't falsely assume that 'active' is somehow mathematically "greater" than 'sleepy'.

### Physical Standardization (Z-Score)
```javascript
function standardizePhysical(physicalVectors) {
    // Computes mean and std_dev across all dimensions (age, weight, length)
    // Returns (value - mean) / (stdDev || 1)
}
```
**Design Decision:** A pet's age might be `5` years, while its weight might be `80` lbs. If we used Euclidean distance directly, the `weight` variable would disproportionately dominate the distance calculation simply because its numerical scale is larger. Z-Score normalization solves this by placing all physical features on the same unit variance scale.

## 2. Gatekeeper (Anomaly Detection)

When a new user registers, the data passes through `gatekeeper(petData)`.

```javascript
function gatekeeper(pet) {
    const age = parseInt(pet.birthYear) ? new Date().getFullYear() - parseInt(pet.birthYear) : 0;
    const weight = parseFloat(pet.weight) || 0;
    const length = parseFloat(pet.length) || 0;
    
    // Hard limits
    if (age > 40 || weight > 250 || length > 200) return true;
    
    // Z-Score threshold limits for anomaly detection
    // ...
}
```
**Design Decision:** Open registration platforms are vulnerable to spam or test accounts (e.g., someone typing a weight of `9999` lbs). Instead of strict frontend validation which can be bypassed, the gatekeeper acts as an automated moderator. Flagged accounts are immediately segregated into the Admin Dashboard (`suspicious_profiles.csv`) and excluded from K-Means clustering and matching until a human admin approves them via `/api/admin/pets/:username/accept`.

## 3. Unsupervised Clustering

We use two distinct clustering approaches to organize users.

### K-Means (Physical Clustering)
```javascript
function assignToCluster(pet) {
    // Finds the nearest centroid based on Euclidean distance of physical traits
    // Returns cluster ID (0, 1, or 2)
}
```
**Design Decision:** K-Means runs in the background across all valid users. When a new user registers, we do not re-run K-Means on the entire dataset (which is `O(n*k*i)`); instead, we simply calculate the Euclidean distance to the *existing* centroids. This is highly optimized and keeps the `POST /api/pets` endpoint lightning fast.

### AGNES (Behavioral Hierarchy)
```javascript
const hclust = require('ml-hclust');
const agnes = hclust.agnes(features, { method: 'ward', distanceFunction: hammingDistance });
```
**Design Decision:** We use AGNES (Agglomerative Nesting) with Ward's linkage and Hamming distance specifically for the one-hot encoded behavioral traits. Unlike K-Means, which creates flat groupings, AGNES creates a dendrogram (tree). This allows us to find users who share highly specific behavioral niches, which is visualised dynamically in the Admin Dashboard.

## 4. Apriori (Association Rule Learning)

```javascript
const { Apriori } = require('node-apriori');
// Evaluates frequent itemsets (e.g. breed_street cat, age_young, trait_active)
```
**Design Decision:** Apriori scours the database for hidden trends (e.g., "70% of people with Siamese cats also select the 'active' trait"). These rules dynamically influence the recommendation algorithm. If a candidate match fits an association rule observed across the broader community, their `aprioriScore` gets a boost. This allows the matchmaking logic to "evolve" as the user base grows.

## 5. The Fusion Score & Random Forest

When a user requests their Playdates feed (`GET /api/users/:username/playdates`), we score every potential candidate against them.

```javascript
c.fusionScore = (physScore * 0.3) + 
                (cfScore * 0.3) + 
                (behScore * 0.3) + 
                (aprioriScore * 0.1);
```
**Design Decision:** The system uses a weighted ensemble (Fusion Score) combining:
1. **Physical Similarity** (Euclidean).
2. **Collaborative Filtering (CF)** (Jaccard similarity based on previous user interactions/swipes).
3. **Behavioral Similarity** (AGNES clusters).
4. **Community Trends** (Apriori).

Finally, a **Random Forest Classifier** evaluates this score to give a final verdict:
```javascript
const multiplier = 0.8 + (prob * 0.4); 
c.fusionScore = Math.min(1.0, c.fusionScore * multiplier);
c.matchScore = prob > 0.5 ? 'High' : 'Low'; 
```
**Design Decision:** The Random Forest model provides a non-linear adjustment. It acts as the final "judge," pushing the `fusionScore` up or down based on historical patterns of what constitutes a successful match. We extract multiple decision trees from the Forest to compute an exact probability (`prob`), which determines the UI's 'High' vs. 'Low' match badge.
