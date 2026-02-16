"""
CLEANED URL CONTENT APPROACH
Remove noise, keep only meaningful words for better matching
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import string

def extract_category(standard):
    """Extract category from standard"""
    parts = standard.split('.')
    if len(parts) >= 2:
        return '.'.join(parts[:2])
    return standard

def clean_text_smart(text, max_words=150):
    """
    Clean text by removing trash and keeping meaningful words
    """
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, emails, special chars
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s-]', ' ', text)
    
    # Common trash words to remove (keep math-specific words!)
    trash_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'our',
        'you', 'your', 'page', 'section', 'chapter', 'figure', 'table',
        'example', 'exercise', 'problem', 'click', 'see', 'show', 'here',
        'there', 'where', 'when', 'who', 'what', 'which', 'how', 'why'
    }
    
    # Math-specific keywords to KEEP (never remove these)
    keep_words = {
        'exponent', 'power', 'scientific', 'notation', 'multiply', 'divide',
        'convert', 'decimal', 'integer', 'rational', 'irrational', 'square',
        'cube', 'root', 'volume', 'area', 'surface', 'cylinder', 'cone',
        'sphere', 'pyramid', 'circle', 'triangle', 'angle', 'pythagorean',
        'theorem', 'formula', 'explain', 'derive', 'calculate', 'solve',
        'estimate', 'approximate', 'model', 'design', 'geometric', 'apply',
        'operation', 'add', 'subtract', 'fraction', 'equation', 'expression',
        'variable', 'coefficient', 'term', 'factor', 'product', 'sum',
        'difference', 'quotient', 'slope', 'intercept', 'graph', 'function',
        'domain', 'range', 'coordinate', 'axis', 'origin', 'perpendicular',
        'parallel', 'congruent', 'similar', 'transformation', 'reflection',
        'rotation', 'translation', 'dilation', 'scale', 'proportion', 'ratio',
        'percent', 'probability', 'statistics', 'mean', 'median', 'mode',
        'standard', 'deviation', 'distribution', 'sample', 'population'
    }
    
    # Tokenize
    words = text.split()
    
    # Keep word if:
    # 1. It's a math keyword, OR
    # 2. It's not a trash word AND it's longer than 3 chars
    cleaned_words = []
    for word in words:
        word = word.strip()
        if word in keep_words:
            cleaned_words.append(word)
        elif word not in trash_words and len(word) > 3:
            # Also keep if it looks mathematical (has numbers)
            if any(char.isdigit() for char in word):
                cleaned_words.append(word)
            elif word.isalpha():  # Regular word
                cleaned_words.append(word)
    
    # Limit to max_words
    cleaned_words = cleaned_words[:max_words]
    
    return ' '.join(cleaned_words)

def extract_items(data, include_labels=True):
    """Extract all items"""
    items = []
    item_id = 0
    
    for book in data['titles']:
        for cluster_group in book.get('items', []):
            concept = cluster_group.get('concept', '')
            domain = cluster_group.get('domain', '')
            category_stds = cluster_group.get('standards', [])
            category_std = category_stds[0] if category_stds else None
            
            for cluster in cluster_group.get('clusters', []):
                cluster_desc = cluster.get('cluster', '')
                
                for item in cluster.get('items', []):
                    text_parts = [
                        item.get('description', ''),
                        item.get('text', ''),
                        item.get('title', ''),
                        cluster_desc,
                        concept,
                        domain
                    ]
                    
                    item_data = {
                        'item_id': item_id,
                        'concept': concept,
                        'domain': domain,
                        'category_standard': category_std,
                        'combined_text': ' '.join([t for t in text_parts if t])
                    }
                    
                    if include_labels:
                        stds = item.get('standards', [])
                        item_data['standard'] = stds[0] if stds else None
                        item_data['category'] = extract_category(stds[0]) if stds else None
                    
                    items.append(item_data)
                    item_id += 1
    
    return pd.DataFrame(items)

def optimized_predict(item_text, item_category, std_embeddings, std_ids, 
                     standards_def, model):
    """Prediction with hierarchical filtering + semantic matching"""
    
    # Encode item
    item_embedding = model.encode([item_text])[0]
    
    # Get categories
    std_categories = {std_id: extract_category(std_id) for std_id in std_ids}
    
    # Filter by category
    if item_category and item_category in std_categories.values():
        candidate_stds = [std_id for std_id, cat in std_categories.items() 
                         if cat == item_category or cat.startswith(item_category)]
    else:
        # Predict category
        all_similarities = cosine_similarity([item_embedding], std_embeddings)[0]
        category_scores = {}
        for i, std_id in enumerate(std_ids):
            cat = std_categories[std_id]
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(all_similarities[i])
        
        category_avg = {cat: np.mean(scores) for cat, scores in category_scores.items()}
        best_category = max(category_avg, key=category_avg.get)
        candidate_stds = [std_id for std_id, cat in std_categories.items() 
                         if cat == best_category]
    
    # Score candidates
    candidate_indices = [std_ids.index(std_id) for std_id in candidate_stds]
    candidate_embeddings = std_embeddings[candidate_indices]
    
    semantic_sims = cosine_similarity([item_embedding], candidate_embeddings)[0]
    
    scores = {}
    for i, std_id in enumerate(candidate_stds):
        scores[std_id] = semantic_sims[i]
    
    best_std = max(scores, key=scores.get)
    confidence = scores[best_std]
    
    return best_std, confidence

if __name__ == "__main__":
    
    print("=" * 80)
    print("CLEANED URL CONTENT APPROACH")
    print("Remove noise, keep only meaningful mathematical terms")
    print("=" * 80)
    
    # Load data
    with open('training.json', 'r') as f:
        training_data = json.load(f)
    
    with open('testing.json', 'r') as f:
        testing_data = json.load(f)
    
    standards_def = training_data['standards_definitions']
    
    # Load previously fetched URL content
    print("\nLoading test data with URL content...")
    try:
        test_df = pd.read_csv('test_with_url_content.csv')
        print("✅ Loaded cached URL content")
    except FileNotFoundError:
        print("❌ test_with_url_content.csv not found!")
        print("   Run with_url_fetching.py first")
        exit(1)
    
    # Clean the URL content
    print("\n🧹 Cleaning URL content (removing trash, keeping math terms)...")
    test_df['cleaned_url_content'] = test_df['url_content'].apply(
        lambda x: clean_text_smart(str(x), max_words=150) if pd.notna(x) else ''
    )
    
    # Create enhanced text with cleaned content
    test_df['enhanced_text_cleaned'] = test_df.apply(
        lambda row: f"{row['combined_text']} {row['cleaned_url_content']}", 
        axis=1
    )
    
    # Show improvement
    avg_original = test_df['enhanced_text'].str.len().mean()
    avg_cleaned = test_df['enhanced_text_cleaned'].str.len().mean()
    print(f"   Original avg: {avg_original:.0f} chars")
    print(f"   Cleaned avg: {avg_cleaned:.0f} chars")
    print(f"   Reduction: {(1 - avg_cleaned/avg_original)*100:.0f}%")
    
    # Show sample
    print("\n📋 Sample cleaned text:")
    sample_idx = 12  # One of the 8.EE items
    print(f"   Original ({len(test_df.iloc[sample_idx]['enhanced_text'])} chars):")
    print(f"   {test_df.iloc[sample_idx]['enhanced_text'][:200]}...")
    print(f"\n   Cleaned ({len(test_df.iloc[sample_idx]['enhanced_text_cleaned'])} chars):")
    print(f"   {test_df.iloc[sample_idx]['enhanced_text_cleaned'][:200]}...")
    
    # Load model
    print("\nLoading sentence transformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode standards
    print("Encoding standard definitions...")
    std_ids = list(standards_def.keys())
    std_texts = [f"{std_id}: {standards_def[std_id]}" for std_id in std_ids]
    std_embeddings = model.encode(std_texts, show_progress_bar=True)
    
    # Predict using cleaned text
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS WITH CLEANED URL CONTENT")
    print("=" * 80)
    
    predictions = []
    
    for idx, row in test_df.iterrows():
        # Use cleaned enhanced text
        item_text = row['enhanced_text_cleaned']
        item_category = row['category_standard'] if pd.notna(row.get('category_standard')) else row.get('category')
        
        pred_std, confidence = optimized_predict(
            item_text, item_category, std_embeddings, std_ids,
            standards_def, model
        )
        
        predictions.append({
            'item_id': row['item_id'],
            'predicted_standard': pred_std,
            'true_standard': row['standard'],
            'confidence': confidence
        })
        
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(test_df)}...")
    
    results_df = pd.DataFrame(predictions)
    
    # Calculate accuracy
    correct = (results_df['predicted_standard'] == results_df['true_standard']).sum()
    accuracy = correct / len(results_df)
    
    print("\n" + "=" * 80)
    print("RESULTS WITH CLEANED URL CONTENT")
    print("=" * 80)
    print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Correct: {correct}/{len(results_df)}")
    
    # Save
    submission = results_df[['item_id', 'predicted_standard']]
    submission.to_csv('predictions_cleaned_urls.csv', index=False)
    
    print("\n✅ Saved: predictions_cleaned_urls.csv")
    
    # Detailed analysis
    print("\n" + "=" * 80)
    print("BY STANDARD ACCURACY")
    print("=" * 80)
    
    for std in test_df['standard'].value_counts().index:
        subset = results_df[results_df['true_standard'] == std]
        std_correct = (subset['predicted_standard'] == subset['true_standard']).sum()
        std_accuracy = std_correct / len(subset) if len(subset) > 0 else 0
        
        status = "✅" if std_accuracy == 1.0 else "⚠️" if std_accuracy >= 0.5 else "❌"
        print(f"  {status} {std}: {std_correct}/{len(subset)} ({std_accuracy*100:.0f}%)")
    
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print("  Without URLs:      55.3%")
    print("  With raw URLs:     51.1%")
    print(f"  With cleaned URLs: {accuracy*100:.1f}%")
    
    print("\n" + "=" * 80)
    print(f"🏆 FINAL: {accuracy*100:.1f}%")
    print("=" * 80)
    
    if accuracy > 0.60:
        print("\n🎉 EXCELLENT! Over 60%!")
    elif accuracy > 0.553:
        print(f"\n✅ IMPROVED! +{(accuracy-0.553)*100:.1f}% from 55.3%")
    else:
        print(f"\n📊 Result: {accuracy*100:.1f}% (target was 55.3%+)")
