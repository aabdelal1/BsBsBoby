# Data Mining Techniques in Pet Matchmaker

This document provides an exhaustive, in-depth explanation of the various Data Mining and Machine Learning algorithms implemented in `ml_pipeline.js`. 

The system relies on a hybrid approach, combining **unsupervised learning** (Clustering, Association Rule Mining), **supervised learning** (Random Forest), and **recommender system heuristics** (Collaborative Filtering) to generate the final "Fusion Score".

---

## 1. Feature Scaling (Z-Score Normalization)

Data mining algorithms that rely on distance metrics (like K-Means) are highly sensitive to the magnitude of the features. A pet's weight might range from `2` to `150` lbs, while its age ranges from `0` to `20` years. If left unscaled, the weight difference would mathematically dominate the age difference simply because the numbers are larger.

```javascript
function standardizePhysical(features) {
    return features.map(row => row.map((val, idx) => 
        (val - globalScalingParams.means[idx]) / globalScalingParams.stdDevs[idx]
    ));
}
```
**Implementation Detail:** Before any physical clustering occurs, the pipeline calculates the mean ($\mu$) and standard deviation ($\sigma$) across all pets for Age, Weight, and Length. Each raw value is then transformed into a Z-Score ($Z = (x - \mu) / \sigma$). This centers the data around 0 with a variance of 1.

---

## 2. Unsupervised Clustering: K-Means & The Elbow Method

The system groups pets into distinct clusters based on their standardized physical features.

### Automated K-Selection (Kneedle Algorithm)
Rather than hardcoding the number of clusters ($k$), the system dynamically calculates the optimal $k$ using the Elbow Method based on the **Within-Cluster Sum of Squares (WCSS)**.

```javascript
function computeWCSS(data, clusters, centroids) {
    let wcss = 0;
    for (let i = 0; i < data.length; i++) {
        const clusterIdx = clusters[i];
        wcss += data[i].reduce((sum, val, d) => sum + Math.pow(val - centroids[clusterIdx][d], 2), 0);
    }
    return wcss;
}
```
**Implementation Detail:** The system evaluates $k=1$ through $k=10$. For each $k$, it runs the K-Means algorithm and records the WCSS. It then draws a vector from the first point ($k=1$) to the last point, and finds the $k$ that has the maximum perpendicular distance from that line (the "knee"). This dynamically adapts the application as the dataset grows in complexity.

---

## 3. Anomaly Detection: Mahalanobis Distance

To prevent spam profiles (e.g., a "cat" weighing 500 lbs), the `gatekeeper` acts as an automated anomaly detection system.

```javascript
function getMahalanobisDistanceSq(scaledPoint, stats) {
    const diff = scaledPoint.map((val, i) => val - stats.meanVector[i]);
    let mahalanobisSq = 0;
    // ... matrix multiplication using stats.invCovMatrix ...
    return mahalanobisSq;
}
```
**Implementation Detail:** While K-Means uses Euclidean distance, Euclidean distance assumes that data is spherically distributed. In reality, weight and length are highly correlated. The system calculates the **Covariance Matrix** of each cluster and inverts it. When a new pet registers, we calculate its **Mahalanobis Distance** from the cluster centroid. If this distance exceeds a strict $\chi^2$ (Chi-Square) threshold, the profile is flagged as an anomaly.

---

## 4. Hierarchical Clustering: AGNES

While physical features use K-Means, behavioral traits (Active, Friendly, Calm, Touchy, Sleepy) are categorical.

```javascript
const agnes = hclust.agnes(features, { method: 'ward', distanceFunction: hammingDistance });
```
**Implementation Detail:** 
1. **One-Hot Encoding:** Behavioral traits are converted to a binary vector (e.g., `[1, 0, 1, 0, 0]`).
2. **Hamming Distance:** Instead of Euclidean distance, the algorithm uses Hamming distance (which simply counts the number of differing bits between two vectors) because the data is strictly categorical.
3. **Ward's Linkage:** The AGNES algorithm merges users into a hierarchical tree (Dendrogram) by minimizing the variance inside the merged clusters at each step.

---

## 5. Association Rule Mining: Apriori

To discover hidden community trends, the system runs the classic Apriori algorithm across a fused dataset of physical buckets and behavioral traits.

```javascript
petProps.push(`age_${targetAge < 2 ? 'young' : targetAge < 8 ? 'adult' : 'senior'}`);
// Extracts itemsets like ["breed_siamese", "trait_active", "age_young"]
```
**Implementation Detail:** The pipeline converts continuous data into categorical buckets (e.g., `age_young`, `age_adult`). It then runs `node-apriori` to find **Frequent Itemsets** that meet a minimum support threshold. 
When generating a match score for a candidate, the system checks if the candidate possesses traits that form a known high-lift association rule. If so, their `aprioriScore` is increased.

---

## 6. Memory-Based Collaborative Filtering

The matchmaking system implements Collaborative Filtering based on the user's historical swipe data logged in `interactions.csv`.

```javascript
const likers = globalItemLikers[c.username] || new Set();
likers.forEach(liker => {
    if (globalUserSimMatrix[username] && globalUserSimMatrix[username][liker]) {
        cfRaw += globalUserSimMatrix[username][liker]; 
    }
});
```
**Implementation Detail:** 
1. **User-User Similarity Matrix:** The system builds a similarity matrix using Jaccard similarity. If User A and User B liked/skipped the exact same profiles, their similarity approaches `1.0`.
2. **Scoring:** If you are evaluating a Candidate, the system looks at everyone who liked that Candidate. If the people who liked the Candidate are *highly similar* to you, the Collaborative Filtering score (`cfScore`) surges.

---

## 7. The Fusion Score & Random Forest

The final step is converting the distances and rules into a definitive prediction.

### The Fusion Score
```javascript
c.fusionScore = (physScore * 0.3) + (cfScore * 0.3) + (behScore * 0.3) + (aprioriScore * 0.1);
```
This is a weighted ensemble aggregating the results of all the aforementioned algorithms into a single continuous variable between `0.0` and `1.0`.

### Supervised Classification (Random Forest)
Because the hardcoded weights (`0.3`, `0.1`) are arbitrary, a Machine Learning classifier is overlaid to "learn" what actually causes a successful match.

```javascript
// During background training:
const rf = new RandomForestClassifier({ nEstimators: 20, maxDepth: 5 });
rf.train(trainingData, trainingLabels);
```
**Implementation Detail:** The background process looks at historical "likes" and treats them as positive labels (`1`), and "skips" as negative labels (`0`). It trains a Random Forest of 20 decision trees on the feature differences between the two users.
During live inference, the Random Forest outputs a probability (`prob`). This probability acts as a multiplier on the `fusionScore`, and dictates whether the UI renders a **"High"** or **"Low"** match badge.
