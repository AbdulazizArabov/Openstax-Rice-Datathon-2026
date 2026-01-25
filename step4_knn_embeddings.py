"""
OpenStax Standards Classification - Improved Semantic Similarity
Using embeddings + k-nearest neighbors (similar to Milvus approach)
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import json
import joblib
from collections import Counter

# ============================================================================
# EMBEDDING-BASED CLASSIFICATION (Like Milvus + Cosine Similarity)
# ============================================================================

def create_embeddings(df, model_name='all-MiniLM-L6-v2'):
    """
    Create embeddings for all items using sentence transformers.
    This is similar to storing in Milvus vector DB.
    """
    print(f"Loading sentence transformer model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Combine all text fields
    texts = (
        df['combined_text'].fillna('') + ' ' +
        df['cluster_description'].fillna('') + ' ' +
        df['concept'].fillna('') + ' ' +
        df['domain'].fillna('')
    ).tolist()
    
    print(f"Generating embeddings for {len(texts)} items...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return embeddings, model


def knn_with_embeddings(train_embeddings, train_labels, test_embeddings, k=5):
    """
    Use k-nearest neighbors on embeddings.
    This simulates: query Milvus → get k most similar → vote on label
    """
    print(f"\nUsing k-NN with k={k} neighbors...")
    
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights='distance')
    knn.fit(train_embeddings, train_labels)
    
    return knn


def direct_cosine_similarity_vote(train_embeddings, train_labels, test_embeddings, k=5):
    """
    Alternative: Direct cosine similarity + voting
    This is exactly what Milvus does: find k most similar, vote on labels
    """
    print(f"\nUsing direct cosine similarity with k={k} neighbors...")
    
    predictions = []
    
    for test_emb in test_embeddings:
        # Compute cosine similarity to all training examples
        similarities = cosine_similarity([test_emb], train_embeddings)[0]
        
        # Get top k most similar
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_labels = [train_labels[i] for i in top_k_indices]
        
        # Vote (most common label)
        most_common = Counter(top_k_labels).most_common(1)[0][0]
        predictions.append(most_common)
    
    return predictions


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("Loading processed data...")
    train_df = pd.read_csv('train_processed.csv')
    test_df = pd.read_csv('test_processed.csv')
    
    with open('standards_definitions.json', 'r') as f:
        standards_def = json.load(f)
    
    print(f"Training set: {len(train_df)} items")
    print(f"Test set: {len(test_df)} items")
    print(f"Number of standards: {len(standards_def)}")
    
    # ========================================================================
    # STEP 1: Generate Embeddings (Like storing in Milvus)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 1: GENERATING EMBEDDINGS")
    print("=" * 80)
    
    train_embeddings, model = create_embeddings(train_df)
    print(f"Training embeddings shape: {train_embeddings.shape}")
    
    test_embeddings, _ = create_embeddings(test_df)
    print(f"Test embeddings shape: {test_embeddings.shape}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_labels = train_df['standard'].values
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    
    # ========================================================================
    # STEP 2: Try Different k Values for k-NN
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 2: FINDING BEST k FOR k-NN")
    print("=" * 80)
    
    best_k = 5
    best_score = 0
    
    for k in [1, 3, 5, 7, 10, 15]:
        print(f"\nTrying k={k}...")
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights='distance')
        
        # Cross-validation
        try:
            scores = cross_val_score(knn, train_embeddings, train_labels_encoded, cv=5, scoring='accuracy')
            mean_score = scores.mean()
            print(f"  Cross-validation accuracy: {mean_score:.4f} (+/- {scores.std()*2:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\n✅ Best k={best_k} with accuracy {best_score:.4f}")
    
    # ========================================================================
    # STEP 3: Train Final Model with Best k
    # ========================================================================
    
    print("\n" + "=" * 80)
    print(f"STEP 3: TRAINING FINAL MODEL (k={best_k})")
    print("=" * 80)
    
    # Method 1: sklearn KNN
    print("\nMethod 1: sklearn KNeighborsClassifier...")
    final_knn = KNeighborsClassifier(n_neighbors=best_k, metric='cosine', weights='distance')
    final_knn.fit(train_embeddings, train_labels_encoded)
    
    knn_predictions_encoded = final_knn.predict(test_embeddings)
    knn_predictions = label_encoder.inverse_transform(knn_predictions_encoded)
    
    # Method 2: Direct cosine similarity (manual, like Milvus)
    print(f"\nMethod 2: Direct cosine similarity (Milvus-style)...")
    cosine_predictions = direct_cosine_similarity_vote(
        train_embeddings, train_labels, test_embeddings, k=best_k
    )
    
    # ========================================================================
    # STEP 4: Generate Predictions
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING PREDICTIONS")
    print("=" * 80)
    
    # KNN predictions
    knn_submission = pd.DataFrame({
        'item_id': test_df['item_id'],
        'predicted_standard': knn_predictions
    })
    knn_submission.to_csv('predictions_knn.csv', index=False)
    print("✅ KNN predictions saved to predictions_knn.csv")
    
    # Cosine similarity predictions
    cosine_submission = pd.DataFrame({
        'item_id': test_df['item_id'],
        'predicted_standard': cosine_predictions
    })
    cosine_submission.to_csv('predictions_cosine.csv', index=False)
    print("✅ Cosine similarity predictions saved to predictions_cosine.csv")
    
    # Check agreement
    agreement = (knn_predictions == cosine_predictions).sum()
    print(f"\nMethods agree on {agreement}/{len(test_df)} predictions ({agreement/len(test_df)*100:.1f}%)")
    
    # Save model
    joblib.dump(final_knn, 'knn_model.pkl')
    joblib.dump(label_encoder, 'knn_label_encoder.pkl')
    print("✅ Models saved")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Approach: Embeddings + k-NN (similar to Milvus + cosine similarity)")
    print(f"Best k: {best_k}")
    print(f"Cross-validation accuracy: {best_score:.4f}")
    print(f"\nPrediction files created:")
    print(f"  - predictions_knn.csv (recommended)")
    print(f"  - predictions_cosine.csv (alternative)")
    print(f"\nThis approach:")
    print(f"  ✅ Uses semantic embeddings (sentence transformers)")
    print(f"  ✅ Finds k most similar training examples (like Milvus)")
    print(f"  ✅ Votes on their labels (cosine similarity)")
    print(f"\n💡 This is similar to what your teammates are doing with Milvus!")
