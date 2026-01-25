"""
Fetch URL Content for Better Features
This can significantly improve accuracy by using actual textbook content
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import json

def fetch_url_content(url, max_retries=3):
    """Fetch and extract text from URL"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit length (keep first 2000 characters)
            return text[:2000]
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                return ""
    return ""


def add_url_content_to_data(df, desc="Processing"):
    """Add URL content to dataframe"""
    
    df = df.copy()
    df['url_content'] = ''
    
    # Get items with URLs
    urls_to_fetch = df[df['url'] != ''].index
    
    print(f"{desc}: Fetching content from {len(urls_to_fetch)} URLs...")
    print("This will take 5-10 minutes...")
    
    for idx in tqdm(urls_to_fetch):
        url = df.loc[idx, 'url']
        content = fetch_url_content(url)
        df.loc[idx, 'url_content'] = content
        time.sleep(0.2)  # Be polite to server
    
    return df


def create_enhanced_text(df):
    """Combine all text sources"""
    
    df['enhanced_text'] = (
        df['combined_text'].fillna('') + ' ' +
        df['cluster_description'].fillna('') + ' ' +
        df['concept'].fillna('') + ' ' +
        df['domain'].fillna('') + ' ' +
        df['url_content'].fillna('')
    )
    
    return df


if __name__ == "__main__":
    
    print("=" * 80)
    print("FETCHING URL CONTENT FOR IMPROVED FEATURES")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('train_processed.csv')
    test_df = pd.read_csv('test_processed.csv')
    
    print(f"Training items: {len(train_df)}")
    print(f"Testing items: {len(test_df)}")
    
    # Check URLs
    train_with_urls = (train_df['url'] != '').sum()
    test_with_urls = (test_df['url'] != '').sum()
    
    print(f"\nItems with URLs:")
    print(f"  Training: {train_with_urls}/{len(train_df)} ({train_with_urls/len(train_df)*100:.1f}%)")
    print(f"  Testing: {test_with_urls}/{len(test_df)} ({test_with_urls/len(test_df)*100:.1f}%)")
    
    # Fetch content
    print("\n" + "=" * 80)
    user_input = input("Fetch URL content? This takes ~10 minutes. (yes/no): ")
    
    if user_input.lower() in ['yes', 'y']:
        train_df = add_url_content_to_data(train_df, "Training set")
        test_df = add_url_content_to_data(test_df, "Test set")
        
        # Create enhanced text
        train_df = create_enhanced_text(train_df)
        test_df = create_enhanced_text(test_df)
        
        # Save
        train_df.to_csv('train_with_urls.csv', index=False)
        test_df.to_csv('test_with_urls.csv', index=False)
        
        print("\n✅ Enhanced data saved:")
        print("  - train_with_urls.csv")
        print("  - test_with_urls.csv")
        
        print("\n📊 Text length statistics:")
        print(f"  Avg training text length: {train_df['enhanced_text'].str.len().mean():.0f} chars")
        print(f"  Avg test text length: {test_df['enhanced_text'].str.len().mean():.0f} chars")
        
        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("Now you can use train_with_urls.csv and test_with_urls.csv")
        print("with ANY of the previous models for improved accuracy!")
        print("\nExpected improvement: 10-15% accuracy boost")
        
    else:
        print("\nSkipping URL fetch. Use existing data.")
