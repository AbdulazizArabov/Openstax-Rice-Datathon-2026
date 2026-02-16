# Rice Datathon 2026 - Educational Standard Classification


## Project Overview

Automatic classification of educational textbook content by Common Core math standards using zero-shot semantic learning.

### The Challenge
We discovered our training and test datasets had **zero overlapping standards**:
- Training: 96 standards (algebra, functions, statistics)
- Testing: 15 standards (geometry, number systems)
- **Overlap: 0 standards**

Traditional supervised machine learning failed completely (0% accuracy), forcing us to develop a creative zero-shot approach.

---

## Final Results

- **Test Accuracy:** 57.4% (54/94 correct)
- **Perfect Standards:** 8 out of 15 (100% accuracy)
- **Good Standards:** 10 out of 15 (50%+ accuracy)
- **Total Standards:** 173 possible classifications

### Key Achievement
Achieved 57.4% accuracy despite having **zero training examples** of the test standards—demonstrating successful zero-shot semantic classification.

---

## Technical Approach

### Architecture
1. **Web Scraping:** Fetch full textbook page content using BeautifulSoup
2. **Smart Text Cleaning:** Remove 100+ noise words, preserve 80+ math terms
3. **Embeddings:** Sentence-BERT (all-MiniLM-L6-v2) generates 384-dimensional semantic vectors
4. **Hierarchical Classification:** Category filtering → semantic matching
5. **Prediction:** Cosine similarity to find closest standard definition

### Key Innovation
Zero-shot semantic classification when traditional supervised learning was impossible due to domain shift.

---

## Repository Structure

```
├── data/
│   ├── training.json                    # Training dataset
│   ├── testing.json                     # Test dataset
│   └── test_with_url_content.csv        # Cached URL content
│
├── predictions/
│   └── predictions_cleaned_urls.csv     # Final predictions (57.4%)
│
├── code/
│   ├── cleaned_url_approach.py          # Final model implementation
│   └── create_presentation_visuals.py   # Generate visualizations
│
├── visualizations/
│   ├── viz1_zero_overlap.png            # The zero-overlap challenge
│   ├── viz2_approach_evolution.png      # Model iteration progress
│   ├── viz3_per_standard_performance.png # Per-standard accuracy
│   ├── viz4_confusion_matrix.png        # Error analysis
│   ├── viz5_architecture.png            # System architecture
│   └── viz6_summary_table.png           # Key metrics summary
│
├── documentation/
│   ├── PRESENTATION_GUIDE.md            # Complete presentation notes
│   └── FULL_3MIN_PRESENTATION.docx      # Video script
│
├── requirements.txt                      # Python dependencies
└── README.md                            # This file
```

---

## How to Run

### Prerequisites
- Python 3.10 or higher
- Required libraries (see requirements.txt)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Generate Predictions

```bash
# Run final model (requires test_with_url_content.csv)
python code/cleaned_url_approach.py
```

Output: `predictions_cleaned_urls.csv`

### Generate Visualizations

```bash
# Create all 6 charts
python code/create_presentation_visuals.py
```

Output: 6 PNG files in current directory

---

## Model Details

### Pre-trained Model
- **Name:** all-MiniLM-L6-v2
- **Type:** Sentence Transformer (BERT-based)
- **Parameters:** 22 million
- **Embedding Dimension:** 384
- **Source:** HuggingFace sentence-transformers

### Training Process
No traditional training—zero-shot approach:
1. Encode all 173 standard definitions
2. Encode each test item
3. Find closest match using cosine similarity

### Performance by Standard

**Perfect (100% accuracy):**
- 8.G.C.9 (Volume formulas): 14/14
- HSG.GMD.A.3 (Apply volume): 11/11
- 8.NS.A.1 (Rational numbers): 8/8
- 8.NS.A.2 (Rational approximations): 4/4
- 8.EE.A.1 (Integer exponents): 4/4
- 8.EE.A.2 (Roots): 4/4
- 8.G.B.7 (Pythagorean): 4/4
- HSG.MG.A.3 (Design): 2/2

**Challenging (0% accuracy):**
- 8.EE.A.3 (Scientific notation conversion): 0/7
- 8.EE.A.4 (Scientific notation operations): 0/5
- HSG.GMD.A.1 (Explain formulas): 0/4
- HSG.MG.A.1 (Modeling): 0/8

---

## Key Insights

### Successes
- Semantic similarity works excellently for standards with distinctive vocabulary
- Hierarchical filtering improves accuracy by narrowing candidate space
- Smart text cleaning (removing noise while preserving mathematical terms) was crucial

### Limitations
- Struggles with semantically similar standards that differ only in operations
- Cannot parse mathematical expressions (MathML)
- Limited to English text
- No domain-specific fine-tuning

### Error Analysis
Errors are **intelligent, not random**:
- Model confuses similar standards within the same domain (e.g., 8.EE.A.3 vs 8.EE.A.4)
- Never confuses geometry with algebra
- Proves the model understands mathematical domains

---

## Future Work

### Path to 70%+ Accuracy

1. **MathML Parsing**
   - Convert mathematical expressions to natural language
   - Capture operations like "multiply" vs "convert"
   - Library: speech-rule-engine

2. **Fine-tuning**
   - Train on educational corpora (math textbooks, problem sets)
   - Learn domain-specific patterns
   - Teacher annotations for difficult cases

3. **Multi-modal Learning**
   - Analyze diagrams, charts, and figures
   - Computer vision for mathematical notation
   - Graph understanding

4. **Active Learning**
   - Human-in-the-loop for low-confidence predictions
   - Teacher feedback integration
   - Iterative improvement

---



## Technologies Used

- **Python 3.10+**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **scikit-learn** - Machine learning utilities
- **sentence-transformers** - Text embeddings
- **beautifulsoup4** - Web scraping
- **requests** - HTTP requests
- **matplotlib** - Visualization
- **seaborn** - Statistical plots

---

## Methodology

### Iterative Development Process

1. **Baseline k-NN (0%)** - Failed due to zero overlap
2. **Definition Matching (45%)** - Direct semantic comparison
3. **Hierarchical (49%)** - Added category filtering
4. **Raw URLs (51%)** - Fetched full content (too noisy)
5. **Cleaned URLs (57.4%)** - Smart text cleaning breakthrough

### Text Processing Pipeline

**Input:** Textbook section
- Title: "Multiply Using Scientific Notation"
- Description: Section overview
- Cluster: "Work with radicals and integer exponents"

**Processing:**
1. Fetch full URL content (3000 chars)
2. Clean text (remove noise, keep math terms → 1250 chars)
3. Combine: title + description + cluster + cleaned URL
4. Generate embedding (384-dim vector)

**Classification:**
1. Filter to category (e.g., "8.EE")
2. Compare to candidate standards (13 options)
3. Predict highest cosine similarity
4. Output: "8.EE.A.4" with confidence 0.85

---

## Acknowledgments

- **Rice Datathon 2026** organizers
- **OpenStax** for educational content
- **HuggingFace** for sentence-transformers
- **Common Core** standards documentation

---

## Contact

For questions or collaboration:
- Email: [Your Email]
- GitHub: [Your GitHub]
- Team: [Your Team Name]

---

## License

This project was created for Rice Datathon 2026. Code and documentation are provided for educational purposes.

---

**Made with ❤️ for education at Rice Datathon 2026**
