"""
OpenStax Standards Classification - Starter Code
Step 1: Data Exploration and Preparation
"""

import json
import pandas as pd
from collections import Counter
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time

# ============================================================================
# LOAD DATA
# ============================================================================

def load_data():
    """Load training and testing JSON files"""
    with open('training.json', 'r') as f:
        training = json.load(f)
    
    with open('testing.json', 'r') as f:
        testing = json.load(f)
    
    return training, testing


# ============================================================================
# EXTRACT ITEMS FROM NESTED STRUCTURE
# ============================================================================

def extract_items(data, include_labels=True):
    """
    Extract all items from the nested JSON structure into a flat list.
    
    Args:
        data: JSON data (training or testing)
        include_labels: Whether to include standard labels (False for test set)
    
    Returns:
        List of dictionaries, one per item
    """
    items = []
    item_id = 0
    
    # Get standards definitions
    standards_def = data.get('standards_definitions', {})
    
    # Iterate through books
    for book in data['titles']:
        book_title = book['title']
        
        # Iterate through concept/domain groups
        for cluster_group in book.get('items', []):
            concept = cluster_group.get('concept', '')
            domain = cluster_group.get('domain', '')
            
            # Iterate through clusters
            for cluster in cluster_group.get('clusters', []):
                cluster_description = cluster.get('cluster', '')
                
                # Iterate through items in cluster
                for item in cluster.get('items', []):
                    item_data = {
                        'item_id': item_id,
                        'book_title': book_title,
                        'concept': concept,
                        'domain': domain,
                        'cluster_description': cluster_description,
                        'item_type': item.get('type', ''),
                        'description': item.get('description', ''),
                        'text': item.get('text', ''),
                        'title': item.get('title', ''),
                        'numbers': ','.join(item.get('numbers', [])),
                        'url': item.get('url', ''),
                    }
                    
                    # Combine text fields into one
                    text_parts = [
                        item_data['description'],
                        item_data['text'],
                        item_data['title']
                    ]
                    item_data['combined_text'] = ' '.join([t for t in text_parts if t])
                    
                    # Add labels for training data
                    if include_labels:
                        standards = item.get('standards', [])
                        # Most items have 1 standard, take the first one
                        item_data['standard'] = standards[0] if standards else None
                        item_data['all_standards'] = ','.join(standards)
                    
                    items.append(item_data)
                    item_id += 1
    
    return pd.DataFrame(items)


# ============================================================================
# FETCH URL CONTENT
# ============================================================================

def fetch_url_content(url, max_retries=3):
    """
    Fetch and extract text content from a URL.
    
    Args:
        url: URL to fetch
        max_retries: Number of retry attempts
    
    Returns:
        Extracted text content or empty string
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
                continue
            else:
                print(f"Failed to fetch {url}: {e}")
                return ""
    
    return ""


def add_url_content(df, max_items=None):
    """
    Add URL content to dataframe.
    
    Args:
        df: DataFrame with 'url' column
        max_items: Limit number of URLs to fetch (for testing)
    
    Returns:
        DataFrame with 'url_content' column added
    """
    df['url_content'] = ''
    
    # Filter items with URLs
    urls_to_fetch = df[df['url'] != ''].index
    
    if max_items:
        urls_to_fetch = urls_to_fetch[:max_items]
    
    print(f"Fetching content from {len(urls_to_fetch)} URLs...")
    
    for idx in tqdm(urls_to_fetch):
        url = df.loc[idx, 'url']
        content = fetch_url_content(url)
        df.loc[idx, 'url_content'] = content
        time.sleep(0.1)  # Be nice to the server
    
    return df


# ============================================================================
# DATA ANALYSIS
# ============================================================================

def analyze_data(train_df, standards_def):
    """Print useful statistics about the data"""
    
    print("=" * 80)
    print("DATA ANALYSIS")
    print("=" * 80)
    
    print(f"\nTotal training items: {len(train_df)}")
    print(f"Total unique standards: {len(standards_def)}")
    print(f"Items with URLs: {train_df['url'].notna().sum()} ({train_df['url'].notna().sum()/len(train_df)*100:.1f}%)")
    print(f"Items with text: {(train_df['combined_text'] != '').sum()}")
    
    print("\nItem type distribution:")
    print(train_df['item_type'].value_counts())
    
    print("\nBooks in training set:")
    print(train_df['book_title'].value_counts())
    
    print("\nTop 10 most common standards:")
    print(train_df['standard'].value_counts().head(10))
    
    print("\nStandards per item distribution:")
    standards_count = train_df['all_standards'].apply(lambda x: len(x.split(',')) if x else 0)
    print(standards_count.value_counts())
    
    print("\nSample items:")
    print(train_df[['item_type', 'combined_text', 'standard']].head())
    
    # Check for missing labels
    missing_labels = train_df['standard'].isna().sum()
    if missing_labels > 0:
        print(f"\n⚠️  WARNING: {missing_labels} items have no standard label!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    training_data, testing_data = load_data()
    
    # Extract items
    print("Extracting items from training data...")
    train_df = extract_items(training_data, include_labels=True)
    
    print("Extracting items from testing data...")
    test_df = extract_items(testing_data, include_labels=False)
    
    # Get standards definitions
    standards_def = training_data['standards_definitions']
    
    # Analyze data
    analyze_data(train_df, standards_def)
    
    # Save processed data
    print("\nSaving processed data...")
    train_df.to_csv('train_processed.csv', index=False)
    test_df.to_csv('test_processed.csv', index=False)
    
    # Save standards definitions
    with open('standards_definitions.json', 'w') as f:
        json.dump(standards_def, f, indent=2)
    
    print("\n✅ Data preparation complete!")
    print(f"   - train_processed.csv: {len(train_df)} items")
    print(f"   - test_processed.csv: {len(test_df)} items")
    print(f"   - standards_definitions.json: {len(standards_def)} standards")
    
    # Optional: Fetch URL content (UNCOMMENT TO ENABLE)
    # This takes time, so start with a small sample
    # print("\nFetching URL content (sample)...")
    # train_df = add_url_content(train_df, max_items=10)  # Test with 10 first
    # train_df.to_csv('train_with_urls.csv', index=False)
