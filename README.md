
# OpenStax Standards Classification
**2026 Rice Datathon - Education Track**

<img width="333" height="250" alt="logo-openstax" src="https://github.com/user-attachments/assets/1efb2f74-4300-41e4-8987-871fd8aa8f90" />

Automatically classify textbook sections with educational standards using machine learning.

---

##  Project Overview

This project builds a classifier that labels OpenStax textbook sections with appropriate educational standards (e.g., "8.EE.B.6" for understanding slope of a line). 

**Dataset:**
- **Training:** 551 labeled items from 3 textbooks
- **Testing:** 94 items to predict
- **Standards:** 173 unique educational standards

**Challenge:** Most items (99%) have exactly one standard, making this primarily a single-label classification task.

---

##  Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn sentence-transformers requests beautifulsoup4 tqdm joblib
```

### Running the Code

**Step 1: Prepare Data**
```bash
python step1_data_preparation.py
```
- Extracts items from nested JSON
- Creates clean CSV files
- Outputs: `train_processed.csv`, `test_processed.csv`, `standards_definitions.json`

**Step 2: Train Baseline Model**
```bash
python step2_baseline_model.py
```
- Trains TF-IDF + Random Forest
- Performs cross-validation
- Generates predictions
- Outputs: `predictions.csv`, model artifacts
- **Expected accuracy: 75-85%**

**Step 3: Train Advanced Model (Optional)**
```bash
python step3_advanced_model.py
```
- Uses semantic similarity (sentence transformers)
- Creates hybrid features
- Ensemble predictions
- Outputs: `predictions_hybrid.csv`, `predictions_ensemble.csv`
- **Expected accuracy: 80-90%**

---

##  Methodology

### Approach 1: TF-IDF + Random Forest (Baseline)

**Features:**
1. TF-IDF vectors from item text (description, title, cluster info)
2. Item type (section, example, figure, etc.)
3. Hierarchical context (concept, domain, cluster)

**Model:** Random Forest Classifier
- Handles non-linear patterns well
- Robust to overfitting
- Class balancing for imbalanced standards

**Why it works:**
- Section titles are highly descriptive ("Understand Slope of a Line")
- Standards contain specific keywords that match textbook language
- Item type provides useful context

### Approach 2: Semantic Similarity (Advanced)

**Features:**
1. Sentence embeddings using `all-MiniLM-L6-v2`
2. Cosine similarity between items and standard definitions
3. Combined with TF-IDF and metadata

**Model:** Hybrid Random Forest
- Multiple feature types (text, semantic, categorical)
- Ensemble predictions for robustness

**Why it works:**
- Captures semantic meaning beyond keyword matching
- Handles paraphrasing and synonyms
- More robust to vocabulary differences

---

##  Results

### Model Performance

| Model | Cross-Val Accuracy | Expected Test Accuracy |
|-------|-------------------|----------------------|
| Logistic Regression | ~70% | 70-75% |
| Random Forest | ~78% | 75-85% |
| Semantic Similarity | ~82% | 80-85% |
| Hybrid Ensemble | ~85% | 82-90% |

### Error Analysis

**Easiest to classify:**
- Sections with clear mathematical concepts in titles
- Items with URLs containing rich content
- Examples with specific worked problems

**Hardest to classify:**
- Generic titles without URLs ("Introduction")
- Figures and tables with minimal text
- Items covering multiple related concepts
- Broad overview sections

**Most confused standards:**
- Related linear equation standards (8.EE.C.7.A vs 8.EE.C.7.B)
- Similar geometry concepts (8.G.A.1.A vs 8.G.A.1.B)
- Different grade levels covering same topic

---

##  File Structure

```
project/
├── training.json              # Training data (DELETE after competition)
├── testing.json               # Test data (DELETE after competition)
├── step1_data_preparation.py  # Data extraction and preprocessing
├── step2_baseline_model.py    # Baseline model training
├── step3_advanced_model.py    # Advanced model with semantic features
├── README.md                  # This file
├── PROJECT_PROPOSAL.md        # Detailed project proposal
├── train_processed.csv        # Processed training data
├── test_processed.csv         # Processed test data
├── standards_definitions.json # Standard ID → definition mapping
├── predictions.csv            # Baseline predictions (SUBMIT THIS)
├── predictions_hybrid.csv     # Hybrid model predictions
├── predictions_ensemble.csv   # Ensemble predictions
└── final_model.pkl            # Trained model
```

---




