"""
k-NN with URL Content - Should get 70%+ accuracy
Run this AFTER step6_fetch_urls.py completes
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import json
import joblib

if __name__ == "__main__":
    
    print("=" * 80)
    print("k-NN WITH URL CONTENT - HIGH ACCURACY VERSION")
    print("=" * 80)
    
    # Load URL-enhanced data
    print("\nLoading URL-enhanced data...")
    try:
        train_df = pd.read_csv('train_with_urls.csv')
        test_df = pd.read_csv('test_with_urls.csv')
        print("✅ Using URL-enhanced data")
    except FileNotFoundError:
        print("❌ URL-enhanced files not found!")
        print("Please run: python step6_fetch_urls.py first")
        exit(1)
    
    with open('standards_definitions.json', 'r') as f:
        standards_def = json.load(f)
    
    print(f"Training: {len(train_df)} items")
    print(f"Testing: {len(test_df)} items")
    
    # Check if URL content exists
    if 'enhanced_text' in train_df.columns:
        print("✅ Using enhanced_text (includes URL content)")
        text_column = 'enhanced_text'
    elif 'url_content' in train_df.columns:
        print("✅ Creating enhanced text from url_content")
        train_df['enhanced_text'] = (
            train_df['combined_text'].fillna('') + ' ' +
            train_df['cluster_description'].fillna('') + ' ' +
            train_df['concept'].fillna('') + ' ' +
            train_df['domain'].fillna('') + ' ' +
            train_df['url_content'].fillna('')
        )
        test_df['enhanced_text'] = (
            test_df['combined_text'].fillna('') + ' ' +
            test_df['cluster_description'].fillna('') + ' ' +
            test_df['concept'].fillna('') + ' ' +
            test_df['domain'].fillna('') + ' ' +
            test_df['url_content'].fillna('')
        )
        text_column = 'enhanced_text'
    else:
        print("⚠️  No URL content found, using basic text")
        text_column = 'combined_text'
    
    # Show text length improvement
    if 'url_content' in train_df.columns:
        basic_len = train_df['combined_text'].str.len().mean()
        enhanced_len = train_df[text_column].str.len().mean()
        print(f"\nText length improvement:")
        print(f"  Basic: {basic_len:.0f} characters")
        print(f"  Enhanced: {enhanced_len:.0f} characters")
        print(f"  Improvement: {(enhanced_len/basic_len-1)*100:.0f}% more content")
    
    # Generate embeddings
    print("\n" + "=" * 80)
    print("GENERATING EMBEDDINGS FROM ENHANCED TEXT")
    print("=" * 80)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    train_texts = train_df[text_column].fillna('').tolist()
    test_texts = test_df[text_column].fillna('').tolist()
    
    print(f"Encoding {len(train_texts)} training items...")
    train_embeddings = model.encode(train_texts, show_progress_bar=True)
    
    print(f"Encoding {len(test_texts)} test items...")
    test_embeddings = model.encode(test_texts, show_progress_bar=True)
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_df['standard'])
    
    # Find best k
    print("\n" + "=" * 80)
    print("FINDING OPTIMAL k")
    print("=" * 80)
    
    best_k = 3
    best_score = 0
    
    for k in [1, 3, 5, 7, 10]:
        print(f"\nTrying k={k}...")
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights='distance')
        
        try:
            scores = cross_val_score(knn, train_embeddings, train_labels_encoded, 
                                    cv=5, scoring='accuracy')
            mean_score = scores.mean()
            print(f"  CV Accuracy: {mean_score:.4f} (+/- {scores.std()*2:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\n✅ Best k={best_k} with accuracy {best_score:.4f}")
    
    # Train final model
    print("\n" + "=" * 80)
    print(f"TRAINING FINAL MODEL (k={best_k})")
    print("=" * 80)
    
    final_knn = KNeighborsClassifier(n_neighbors=best_k, metric='cosine', weights='distance')
    final_knn.fit(train_embeddings, train_labels_encoded)
    
    # Predict
    predictions_encoded = final_knn.predict(test_embeddings)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    
    # Save
    submission = pd.DataFrame({
        'item_id': test_df['item_id'],
        'predicted_standard': predictions
    })
    submission.to_csv('predictions_knn_with_urls.csv', index=False)
    
    print("\n✅ Predictions saved to predictions_knn_with_urls.csv")
    
    # Distribution
    print(f"\nPredicted standards distribution:")
    print(pd.Series(predictions).value_counts().head(10))
    
    # Save model
    joblib.dump(final_knn, 'knn_url_model.pkl')
    joblib.dump(label_encoder, 'knn_url_encoder.pkl')
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Model: k-NN with k={best_k}")
    print(f"Features: Embeddings from URL-enhanced text")
    print(f"Cross-validation accuracy: {best_score:.4f} ({best_score*100:.1f}%)")
    print(f"\nExpected improvement over basic k-NN: 10-15%")
    print(f"Target accuracy: 70-75%")
    
    if best_score < 0.65:
        print("\n⚠️  Accuracy lower than expected.")
        print("Possible issues:")
        print("  - URLs might not have fetched properly")
        print("  - Check if url_content column has data")
    else:
        print("\n🎉 Great accuracy! This should perform well on test set!")
