"""
OpenStax Standards Classification - Baseline Model (FIXED)
Step 2: Train a simple but effective classifier
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import json
import joblib

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(df, standards_def):
    """
    Create features for classification.
    """
    
    # Combine multiple text sources
    df['full_text'] = (
        df['combined_text'].fillna('') + ' ' +
        df['cluster_description'].fillna('') + ' ' +
        df['concept'].fillna('') + ' ' +
        df['domain'].fillna('')
    )
    
    # Clean text
    df['full_text'] = df['full_text'].str.lower().str.strip()
    
    return df


def prepare_text_features(train_df, test_df=None):
    """
    Prepare TF-IDF features from text.
    """
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words='english'
    )
    
    # Fit on training data
    X_train = vectorizer.fit_transform(train_df['full_text'])
    
    if test_df is not None:
        X_test = vectorizer.transform(test_df['full_text'])
        return X_train, X_test, vectorizer
    
    return X_train, vectorizer


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_baseline_model(X_train, y_train):
    """
    Train a Logistic Regression baseline.
    """
    print("Training Logistic Regression baseline...")
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model.
    """
    print("Training Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    return model


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X, y, label_encoder, model_name="Model"):
    """Evaluate model and print metrics"""
    
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"\n{model_name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy, y_pred


def cross_validate_model(model, X, y, cv=5):
    """Perform cross-validation"""
    
    print(f"\nPerforming {cv}-fold cross-validation...")
    
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        return scores
    except Exception as e:
        print(f"Cross-validation warning: {e}")
        print("Continuing without cross-validation...")
        return np.array([0.0])


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
    
    # Prepare features
    print("\nPreparing features...")
    train_df = create_features(train_df, standards_def)
    test_df = create_features(test_df, standards_def)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['standard'])
    
    # Create TF-IDF features
    X_train, X_test, vectorizer = prepare_text_features(train_df, test_df)
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # ========================================================================
    # BASELINE: Logistic Regression
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("BASELINE MODEL: Logistic Regression")
    print("=" * 80)
    
    lr_model = train_baseline_model(X_train_split, y_train_split)
    
    print("\nValidation set performance:")
    lr_val_acc, lr_val_pred = evaluate_model(lr_model, X_val, y_val, label_encoder, "Logistic Regression")
    
    # Cross-validation on full training set
    print("\nCross-validation on full training set:")
    lr_cv_scores = cross_validate_model(lr_model, X_train, y_train, cv=5)
    
    # ========================================================================
    # IMPROVED MODEL: Random Forest
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("IMPROVED MODEL: Random Forest")
    print("=" * 80)
    
    rf_model = train_random_forest(X_train_split, y_train_split)
    
    print("\nValidation set performance:")
    rf_val_acc, rf_val_pred = evaluate_model(rf_model, X_val, y_val, label_encoder, "Random Forest")
    
    # Cross-validation on full training set
    print("\nCross-validation on full training set:")
    rf_cv_scores = cross_validate_model(rf_model, X_train, y_train, cv=5)
    
    # ========================================================================
    # SELECT BEST MODEL
    # ========================================================================
    
    if rf_val_acc > lr_val_acc:
        best_model = rf_model
        best_name = "Random Forest"
        best_cv_scores = rf_cv_scores
    else:
        best_model = lr_model
        best_name = "Logistic Regression"
        best_cv_scores = lr_cv_scores
    
    print("\n" + "=" * 80)
    print(f"BEST MODEL: {best_name}")
    print("=" * 80)
    
    # ========================================================================
    # TRAIN FINAL MODEL ON FULL TRAINING SET
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL ON FULL TRAINING SET")
    print("=" * 80)
    
    if best_name == "Random Forest":
        final_model = train_random_forest(X_train, y_train)
    else:
        final_model = train_baseline_model(X_train, y_train)
    
    # ========================================================================
    # GENERATE PREDICTIONS FOR TEST SET
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 80)
    
    test_predictions = final_model.predict(X_test)
    test_standards = label_encoder.inverse_transform(test_predictions)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'item_id': test_df['item_id'],
        'predicted_standard': test_standards
    })
    
    print(f"\nPredicted standards distribution:")
    print(pd.Series(test_standards).value_counts().head(10))
    
    # Save predictions
    submission.to_csv('predictions.csv', index=False)
    print("\n✅ Predictions saved to predictions.csv")
    
    # Save model and artifacts
    joblib.dump(final_model, 'final_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("✅ Model artifacts saved")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Final model: {best_name}")
    if len(best_cv_scores) > 1:
        print(f"Cross-validation accuracy: {best_cv_scores.mean():.4f} (+/- {best_cv_scores.std()*2:.4f})")
    print(f"\nTest predictions generated: {len(submission)} items")
    print(f"\nFiles created:")
    print(f"  - predictions.csv: Submission file with {len(submission)} predictions")
    print(f"  - final_model.pkl: Trained {best_name} model")
    print(f"  - label_encoder.pkl: Label encoder")
    print(f"  - vectorizer.pkl: TF-IDF vectorizer")
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Check predictions.csv - this is your submission file!")
    print("2. Note your cross-validation accuracy for your presentation")
    print("3. Optionally run step3_advanced_model.py for better accuracy")
    print("4. Start working on your documentation and presentation")