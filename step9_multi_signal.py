"""
Advanced Classification for Higher Accuracy
Goal: Get as close to 100% as possible
Strategy: Multi-signal approach with standard definitions
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import json
from collections import Counter
import re

def extract_standard_category(standard):
    """Extract category from standard (e.g., '8.EE' from '8.EE.B.6')"""
    parts = standard.split('.')
    if len(parts) >= 2:
        return '.'.join(parts[:2])
    return standard

def create_standard_features(standards_def):
    """Create features for each standard"""
    
    standard_info = {}
    
    for std_id, std_def in standards_def.items():
        # Extract category
        category = extract_standard_category(std_id)
        
        # Extract grade level
        if std_id.startswith('8.'):
            grade = '8th'
        elif std_id.startswith('HS'):
            grade = 'high school'
        else:
            grade = 'other'
        
        # Extract domain keywords from definition
        definition_lower = std_def.lower()
        
        # Domain classification
        domains = {
            'algebra': ['equation', 'expression', 'variable', 'solve', 'linear', 'quadratic', 'polynomial'],
            'geometry': ['angle', 'triangle', 'circle', 'line', 'parallel', 'perpendicular', 'coordinate'],
            'statistics': ['data', 'mean', 'median', 'distribution', 'probability', 'correlation'],
            'functions': ['function', 'domain', 'range', 'graph', 'transformation'],
            'number': ['rational', 'irrational', 'integer', 'real', 'complex', 'exponent'],
            'trigonometry': ['sine', 'cosine', 'tangent', 'radian', 'angle', 'periodic']
        }
        
        domain_scores = {}
        for domain, keywords in domains.items():
            score = sum(1 for kw in keywords if kw in definition_lower)
            domain_scores[domain] = score
        
        primary_domain = max(domain_scores, key=domain_scores.get) if sum(domain_scores.values()) > 0 else 'unknown'
        
        standard_info[std_id] = {
            'category': category,
            'grade': grade,
            'domain': primary_domain,
            'definition': std_def,
            'definition_lower': definition_lower
        }
    
    return standard_info

def multi_signal_classification(item_row, train_df, standards_def, standard_info, model, train_embeddings, train_standards):
    """
    Use multiple signals to predict standard:
    1. Semantic similarity to standard definitions
    2. Semantic similarity to training examples (k-NN)
    3. Hierarchical filtering (concept/domain match)
    4. Keyword matching
    """
    
    # Get item info
    item_text = item_row['enhanced_text'] if 'enhanced_text' in item_row else item_row['combined_text']
    item_concept = item_row.get('concept', '')
    item_domain = item_row.get('domain', '')
    item_cluster = item_row.get('cluster_description', '')
    
    full_text = f"{item_text} {item_concept} {item_domain} {item_cluster}".lower()
    
    # Signal 1: Match to standard definitions
    definition_scores = {}
    item_embedding = model.encode([full_text])[0]
    
    for std_id, info in standard_info.items():
        # Semantic similarity
        std_embedding = model.encode([info['definition']])[0]
        sem_score = cosine_similarity([item_embedding], [std_embedding])[0][0]
        
        # Keyword overlap
        item_words = set(full_text.split())
        std_words = set(info['definition_lower'].split())
        
        if len(item_words) > 0:
            keyword_score = len(item_words & std_words) / len(item_words)
        else:
            keyword_score = 0
        
        # Combined score
        definition_scores[std_id] = 0.6 * sem_score + 0.4 * keyword_score
    
    # Signal 2: k-NN on training examples
    item_train_similarity = cosine_similarity([item_embedding], train_embeddings)[0]
    top_k_indices = np.argsort(item_train_similarity)[-5:][::-1]
    
    knn_votes = {}
    for idx in top_k_indices:
        std = train_standards[idx]
        similarity = item_train_similarity[idx]
        knn_votes[std] = knn_votes.get(std, 0) + similarity
    
    # Signal 3: Hierarchical filtering
    # Filter standards by concept/domain if available
    relevant_standards = set()
    
    if item_concept or item_domain:
        for std_id in standards_def.keys():
            # Check if standard category matches item hierarchy
            if '8' in item_concept and std_id.startswith('8.'):
                relevant_standards.add(std_id)
            elif 'algebra' in item_domain.lower() and any(x in std_id for x in ['EE', 'REI', 'CED', 'SSE', 'APR']):
                relevant_standards.add(std_id)
            elif 'statistic' in item_domain.lower() and any(x in std_id for x in ['SP', 'IC', 'ID', 'CP']):
                relevant_standards.add(std_id)
            elif 'geometry' in item_domain.lower() and any(x in std_id for x in ['G.', 'CO', 'SRT', 'GPE']):
                relevant_standards.add(std_id)
            elif 'function' in item_domain.lower() and any(x in std_id for x in ['F.', 'BF', 'IF', 'LE', 'TF']):
                relevant_standards.add(std_id)
            elif 'trigonometry' in item_domain.lower() and 'TF' in std_id:
                relevant_standards.add(std_id)
    
    # If no hierarchical match, consider all
    if len(relevant_standards) == 0:
        relevant_standards = set(standards_def.keys())
    
    # Combine signals
    final_scores = {}
    
    for std_id in relevant_standards:
        score = 0
        
        # Definition matching (weight: 0.4)
        if std_id in definition_scores:
            score += 0.4 * definition_scores[std_id]
        
        # k-NN voting (weight: 0.5)
        if std_id in knn_votes:
            # Normalize k-NN votes
            max_knn = max(knn_votes.values()) if knn_votes else 1
            score += 0.5 * (knn_votes[std_id] / max_knn)
        
        # Hierarchical bonus (weight: 0.1)
        if std_id in relevant_standards and len(relevant_standards) < 50:
            score += 0.1
        
        final_scores[std_id] = score
    
    # Return top prediction
    if final_scores:
        best_std = max(final_scores, key=final_scores.get)
        return best_std, final_scores[best_std]
    else:
        # Fallback to top definition match
        return max(definition_scores, key=definition_scores.get), definition_scores[max(definition_scores, key=definition_scores.get)]

if __name__ == "__main__":
    
    print("=" * 80)
    print("ADVANCED MULTI-SIGNAL CLASSIFICATION")
    print("Goal: Maximum Accuracy")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    try:
        train_df = pd.read_csv('train_with_urls.csv')
        test_df = pd.read_csv('test_with_urls.csv')
        text_col = 'enhanced_text'
    except:
        train_df = pd.read_csv('train_processed.csv')
        test_df = pd.read_csv('test_processed.csv')
        text_col = 'combined_text'
    
    with open('standards_definitions.json', 'r') as f:
        standards_def = json.load(f)
    
    print(f"Training: {len(train_df)} items")
    print(f"Testing: {len(test_df)} items")
    print(f"Standards: {len(standards_def)}")
    
    # Analyze standard features
    print("\nAnalyzing standard definitions...")
    standard_info = create_standard_features(standards_def)
    
    # Load model
    print("\nLoading sentence transformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings
    print("\nCreating embeddings...")
    
    # Prepare text
    if text_col == 'enhanced_text':
        train_texts = train_df['enhanced_text'].fillna('').tolist()
        test_texts = test_df['enhanced_text'].fillna('').tolist()
    else:
        train_texts = (train_df['combined_text'].fillna('') + ' ' + 
                      train_df['cluster_description'].fillna('')).tolist()
        test_texts = (test_df['combined_text'].fillna('') + ' ' + 
                     test_df['cluster_description'].fillna('')).tolist()
    
    print(f"Encoding {len(train_texts)} training items...")
    train_embeddings = model.encode(train_texts, show_progress_bar=True)
    
    print(f"Encoding {len(test_texts)} test items...")
    test_embeddings = model.encode(test_texts, show_progress_bar=True)
    
    train_standards = train_df['standard'].values
    
    # Predict for test set
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS WITH MULTI-SIGNAL APPROACH")
    print("=" * 80)
    
    predictions = []
    confidences = []
    
    for idx, row in test_df.iterrows():
        pred_std, confidence = multi_signal_classification(
            row, train_df, standards_def, standard_info, 
            model, train_embeddings, train_standards
        )
        predictions.append(pred_std)
        confidences.append(confidence)
        
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} items...")
    
    # Save predictions
    submission = pd.DataFrame({
        'item_id': test_df['item_id'],
        'predicted_standard': predictions,
        'confidence': confidences
    })
    
    submission[['item_id', 'predicted_standard']].to_csv('predictions_multi_signal.csv', index=False)
    
    print("\n✅ Predictions saved to predictions_multi_signal.csv")
    
    # Show distribution
    print(f"\nPredicted standards distribution:")
    print(pd.Series(predictions).value_counts().head(15))
    
    # Show low confidence predictions
    print(f"\nLowest confidence predictions (might need manual review):")
    low_conf = submission.nsmallest(10, 'confidence')
    print(low_conf[['item_id', 'predicted_standard', 'confidence']])
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Multi-signal approach combines:")
    print("  1. Semantic similarity to standard definitions (40%)")
    print("  2. k-NN voting from similar training examples (50%)")
    print("  3. Hierarchical filtering by concept/domain (10%)")
    print("\nThis should achieve higher accuracy than single-signal methods")
    print("Expected: 70-80% (better than k-NN alone)")
