# Machine Learning Pipeline

The Pet Matchmaker system relies on a multi-layered machine learning pipeline (`ml_pipeline.js`) to provide intelligent recommendations, moderate fake accounts, and generate insights for the Admin Dashboard.

## 1. Feature Extraction & Standardization
Before any algorithm runs, raw pet data is processed:
- **Physical Features:** Weight, length, and age are normalized using Z-score standardization so that different scales (e.g., years vs. pounds) do not distort distance calculations.
- **Behavioral Features:** Personality traits (active, friendly, calm, touchy, sleepy) are one-hot encoded into a 5-dimensional binary vector.

## 2. Gatekeeper (Anomaly Detection)
When a new user registers a pet, the `gatekeeper` evaluates it for suspicious patterns that indicate spam or fake accounts:
- Compares the pet's physical attributes against hardcoded thresholds.
- Evaluates the standard deviations of weight and length against the system mean.
- If flagged, the profile is pushed to the Admin Dashboard for manual review and is hidden from the public matching feed.

## 3. Clustering (K-Means & AGNES)
To group similar pets together, the system uses two distinct unsupervised clustering algorithms:
- **K-Means Clustering:** Assigns each user to a discrete cluster (`0`, `1`, or `2`) based on their physical similarity. This allows the backend to quickly filter highly dissimilar profiles.
- **AGNES (Agglomerative Nesting):** A hierarchical clustering algorithm used to compute the behavioral similarity between users. It uses Hamming distance on the one-hot encoded personality traits and constructs a dendrogram.

## 4. Apriori (Association Rule Learning)
The backend continually parses the dataset using the Apriori algorithm to discover hidden patterns (e.g., "People who own *Senior Dogs* frequently select the *Calm* trait").
- These rules are used to calculate an **Apriori Score** for potential matches, rewarding matches that follow established community trends.
- The top rules are also pushed to the Admin Dashboard for visualization.

## 5. Recommendation Engine (The Fusion Score)
When a user views their "Playdates" feed, the system generates a personalized list of candidates using a weighted **Fusion Score** composed of:
1. **Physical Distance Score:** Exponential decay function based on the physical features difference.
2. **Collaborative Filtering (CF) Score:** Analyzes the `interactions.csv` file. If User A liked User B, and User C also liked User B, the system predicts User A will like User C (Jaccard similarity).
3. **Behavioral Score:** Uses the AGNES clusters to reward overlapping personality traits.
4. **Apriori Score:** Applies the learned association rules.

## 6. Random Forest Classifier
Finally, a lightweight Random Forest model evaluates the Fusion Score and historical interaction data to output a strict binary classification (`High` vs. `Low` match probability) displayed directly on the pet cards in the UI.

## Background Execution
To keep the REST API blazingly fast, all heavy model training (K-Means, AGNES, Apriori, Random Forest) happens in the background via `setInterval`. The trained models are cached in memory and serialized to `DB/models/ml_pipeline_state.json` to persist across server restarts.
