
# OpenStax Standards Classification
**2026 Rice Datathon - Education Track**

<img width="333" height="250" alt="logo-openstax" src="https://github.com/user-attachments/assets/1efb2f74-4300-41e4-8987-871fd8aa8f90" />

Automatically classify textbook sections with educational standards using machine learning.

---

# Semantiq

![Status](https://img.shields.io/badge/Status-Prototype-yellow)  ![Python](https://img.shields.io/badge/Python-3.9+-blue)  ![Model](https://img.shields.io/badge/Model-Sentence--BERT-orange)

**A zero-shot semantic classification pipeline designed to align textbook content with learning standards—even when no labeled examples exist.**

---

## The Inspiration
Semantiq was born from a critical data failure we encountered at the Rice Datathon. We found that most curriculum alignment systems rely on the assumption that training and testing data share the same standards.

Our dataset proved that assumption wrong:
* **Training Data:** Algebra, Functions, Statistics (96 standards)
* **Testing Data:** Geometry, Number Systems (15 standards)
* **Overlap:** **0 standards**

When we ran traditional supervised learning models, they failed completely (**0% accuracy**). We had to pivot and answer a new question: *Can AI align educational content to standards it has never seen before?*

## How We Built It
We moved away from keyword matching and built a pipeline based on **semantic reasoning**.

### 1. Content Collection
Labels aren't enough. We enriched each textbook item by scraping its full context to give the model a stronger signal:
* Item Title & Description
* Cluster Context
* Full Page Text (via web scraping)

### 2. Smart Text Cleaning
Raw educational data is noisy. We built a cleaning pipeline that:
* Stripped 100+ common noise words (ads, navigation).
* **Preserved** 80+ math-specific terms often lost in standard cleaning.
* *Impact:* This step alone increased accuracy by **6%**.

### 3. Semantic Embeddings
We encoded all content and standards using **Sentence-BERT** (`all-MiniLM-L6-v2`).
Instead of matching keywords, we converted each standard and textbook item into a 384-dimensional vector representing its *meaning*.

### 4. Zero-Shot Matching
We used a hierarchical approach to classify unseen standards:
1.  **Filter:** Limit scope by hierarchical prefix (e.g., only look at `8.EE`).
2.  **Compute:** Calculate Cosine Similarity between the item vector ($A$) and standard vector ($B$).
    $$\text{similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$$
3.  **Predict:** Select the standard with the highest semantic similarity score.

## Results
Despite having **zero training examples** for the test standards (extreme domain shift), Semantiq achieved:

| Metric | Result |
| :--- | :--- |
| **Overall Accuracy** | **57.4%** |
| **Perfect Accuracy** | 8 Standards |
| **>50% Accuracy** | 10 out of 15 Standards |

*Note: The total search space included 173 possible standards.*

## Challenges & Lessons
* **Zero Overlap:** Supervised learning is brittle. We learned that for educational data, systems must generalize, not just memorize labels.
* **Intelligent Errors:** The model didn't make random guesses. It never confused Geometry with Algebra. Errors were strictly between semantically similar concepts (e.g., *converting* vs. *operating* in scientific notation).
* **Hierarchy is Key:** Semantic search is powerful, but structural filtering (using the standard hierarchy) is necessary to guide the model.

## Impact
Semantiq automates a process that usually requires weeks of subject-matter expert time.
* **Scalability:** Enables instant curriculum auditing.
* **Equity:** Supports under-resourced schools by quickly aligning open-source materials to state standards.
* **Savings:** Estimated potential of **150,000 hours saved annually** ($7.5M in educator time).

## Roadmap
- [ ] **MathML Parsing:** Better handling of raw formulas and notation.
- [ ] **Fine-Tuning:** Domain-specific training on pedagogical texts.
- [ ] **Multi-modal:** Incorporating diagrams and figures into the embeddings.
- [ ] **Active Learning:** A loop for teacher feedback to correct low-confidence predictions.

## Built With
* **Lang:** Python
* **ML:** Sentence-BERT (`all-MiniLM-L6-v2`)
* **Data:** Pandas, JSON, CSV
* **Tools:** Visual Studio, GitHub
