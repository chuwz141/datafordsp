# -*- coding: utf-8 -*-
"""
Preprocessing Module for SA-RFM + Semantic MBA Architecture
============================================================
Handles:
  - Loading raw CSV datasets
  - Text cleaning & normalization
  - Missing value imputation
  - Date feature engineering
  - Saving cleaned datasets to dataset/after_EDA/
"""

import pandas as pd
import numpy as np
import re
import os
import warnings
warnings.filterwarnings('ignore')


# ===========================================================================
# 1. LOADING
# ===========================================================================

def load_raw_data(base_path: str = 'dataset/before_EDA') -> dict:
    """Load all 4 raw CSV files and return as a dict of DataFrames."""
    data = {}
    data['reviews'] = pd.read_csv(os.path.join(base_path, 'data_reviews_purchase.csv'))
    data['products'] = pd.read_csv(os.path.join(base_path, 'data_product.csv'))
    data['attributes'] = pd.read_csv(os.path.join(base_path, 'data_product_attribute.csv'))
    data['shops'] = pd.read_csv(os.path.join(base_path, 'data_shop.csv'))
    
    print("=== Raw Data Loaded ===")
    for name, df in data.items():
        print(f"  {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    
    return data


# ===========================================================================
# 2. TEXT CLEANING
# ===========================================================================

def clean_text(text: str) -> str:
    """Basic Vietnamese text cleaning."""
    if pd.isna(text) or not isinstance(text, str):
        return ''
    text = text.strip().lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_text_columns(data: dict) -> dict:
    """Apply text cleaning to all text columns across DataFrames."""
    # Reviews: processed_comment is already preprocessed, light cleaning only
    data['reviews']['processed_comment'] = (
        data['reviews']['processed_comment'].apply(clean_text)
    )
    
    # Products: clean description
    data['products']['processed_description'] = (
        data['products']['processed_description'].apply(clean_text)
    )
    
    # Attributes: clean ingredient, feature, skin_type
    for col in ['ingredient', 'feature', 'skin_type']:
        data['attributes'][col] = data['attributes'][col].apply(clean_text)
    
    print("=== Text Columns Cleaned ===")
    return data


# ===========================================================================
# 3. MISSING VALUE HANDLING
# ===========================================================================

def handle_missing_values(data: dict) -> dict:
    """Impute missing values according to the architecture plan."""
    
    # Reviews: processed_comment nulls -> empty string (will get neutral sentiment)
    null_comments = data['reviews']['processed_comment'].isna().sum()
    data['reviews']['processed_comment'] = (
        data['reviews']['processed_comment'].fillna('')
    )
    
    # Products: no missing values (verified in analysis)
    
    # Attributes: fill missing text fields
    data['attributes']['ingredient'] = data['attributes']['ingredient'].fillna('unknown')
    data['attributes']['skin_type'] = data['attributes']['skin_type'].fillna('all_skin')
    data['attributes']['capacity'] = data['attributes']['capacity'].fillna('unknown')
    data['attributes']['design'] = data['attributes']['design'].fillna('unknown')
    data['attributes']['brand'] = data['attributes']['brand'].fillna('no_brand')
    data['attributes']['expiry'] = data['attributes']['expiry'].fillna('unknown')
    data['attributes']['origin'] = data['attributes']['origin'].fillna('unknown')
    
    # Shops: drop the unnamed index if present
    if 'Unnamed: 0' in data['reviews'].columns:
        data['reviews'] = data['reviews'].drop(columns=['Unnamed: 0'])
    
    print(f"=== Missing Values Handled ===")
    print(f"  Reviews: {null_comments} null comments filled")
    print(f"  Attributes: missing fields imputed")
    
    # Verify no remaining nulls in critical columns
    for name, df in data.items():
        remaining = df.isnull().sum().sum()
        if remaining > 0:
            null_cols = df.columns[df.isnull().any()].tolist()
            print(f"  ⚠️ {name}: {remaining} nulls remaining in {null_cols}")
        else:
            print(f"  ✅ {name}: no nulls remaining")
    
    return data


# ===========================================================================
# 4. DATE ENGINEERING
# ===========================================================================

def engineer_date_features(data: dict) -> dict:
    """Parse cmt_date and extract temporal features."""
    df = data['reviews']
    
    df['cmt_date'] = pd.to_datetime(df['cmt_date'])
    df['purchase_year'] = df['cmt_date'].dt.year
    df['purchase_month'] = df['cmt_date'].dt.month
    df['purchase_day_of_week'] = df['cmt_date'].dt.dayofweek  # 0=Mon, 6=Sun
    df['purchase_hour'] = df['cmt_date'].dt.hour
    
    print(f"=== Date Features Engineered ===")
    print(f"  Date range: {df['cmt_date'].min()} → {df['cmt_date'].max()}")
    print(f"  New columns: purchase_year, purchase_month, purchase_day_of_week, purchase_hour")
    
    data['reviews'] = df
    return data


# ===========================================================================
# 5. RFM COMPUTATION
# ===========================================================================

def compute_rfm(data: dict) -> pd.DataFrame:
    """
    Compute RFM features per user.
    
    R (Recency): days since last purchase
    F (Frequency): number of purchases
    M (Monetary): total spending (VND)
    """
    reviews = data['reviews']
    products = data['products'][['product_id', 'price']]
    
    # Merge to get price per transaction
    merged = reviews.merge(products, on='product_id', how='left')
    
    # Snapshot date = 1 day after last transaction
    snapshot_date = reviews['cmt_date'].max() + pd.Timedelta(days=1)
    
    # Aggregate per user
    rfm = merged.groupby('user_id').agg(
        last_purchase=('cmt_date', 'max'),
        frequency=('product_id', 'count'),
        monetary=('price', 'sum')
    ).reset_index()
    
    # Compute recency
    rfm['recency'] = (snapshot_date - rfm['last_purchase']).dt.days
    
    # Select final columns
    rfm = rfm[['user_id', 'recency', 'frequency', 'monetary']]
    
    print(f"=== RFM Computed ===")
    print(f"  Users: {len(rfm):,}")
    print(f"  Snapshot date: {snapshot_date.date()}")
    print(f"\n  RFM Statistics:")
    print(rfm[['recency', 'frequency', 'monetary']].describe().to_string())
    
    return rfm


def normalize_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Apply Min-Max normalization to RFM features."""
    rfm_norm = rfm.copy()
    
    for col in ['recency', 'frequency', 'monetary']:
        min_val = rfm_norm[col].min()
        max_val = rfm_norm[col].max()
        if max_val > min_val:
            rfm_norm[f'{col}_norm'] = (rfm_norm[col] - min_val) / (max_val - min_val)
        else:
            rfm_norm[f'{col}_norm'] = 0.0
    
    # Invert recency: low recency (recent) should have HIGH score
    rfm_norm['recency_norm'] = 1.0 - rfm_norm['recency_norm']
    
    print(f"\n=== RFM Normalized (Min-Max) ===")
    print(f"  Note: recency_norm is INVERTED (1.0 = most recent)")
    print(rfm_norm[['recency_norm', 'frequency_norm', 'monetary_norm']].describe().to_string())
    
    return rfm_norm


# ===========================================================================
# 6. SAVE CLEANED DATA
# ===========================================================================

def save_cleaned_data(data: dict, rfm: pd.DataFrame, output_path: str = 'dataset/after_EDA'):
    """Save all cleaned DataFrames and RFM table."""
    os.makedirs(output_path, exist_ok=True)
    
    data['reviews'].to_csv(os.path.join(output_path, 'reviews_cleaned.csv'), index=False)
    data['products'].to_csv(os.path.join(output_path, 'products_cleaned.csv'), index=False)
    data['attributes'].to_csv(os.path.join(output_path, 'attributes_cleaned.csv'), index=False)
    data['shops'].to_csv(os.path.join(output_path, 'shops_cleaned.csv'), index=False)
    rfm.to_csv(os.path.join(output_path, 'rfm_table.csv'), index=False)
    
    print(f"\n=== Cleaned Data Saved to {output_path}/ ===")
    for f in os.listdir(output_path):
        size = os.path.getsize(os.path.join(output_path, f))
        print(f"  {f}: {size/1024:.1f} KB")


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def run_preprocessing_pipeline():
    """Execute the full preprocessing pipeline."""
    print("=" * 60)
    print("PHASE 1: DATA PREPROCESSING & RFM ENGINEERING")
    print("=" * 60)
    
    # Step 1: Load
    data = load_raw_data()
    
    # Step 2: Clean text
    data = clean_text_columns(data)
    
    # Step 3: Handle missing values
    data = handle_missing_values(data)
    
    # Step 4: Date engineering
    data = engineer_date_features(data)
    
    # Step 5: Compute RFM
    rfm = compute_rfm(data)
    
    # Step 6: Normalize RFM
    rfm = normalize_rfm(rfm)
    
    # Step 7: Save
    save_cleaned_data(data, rfm)
    
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    
    return data, rfm


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    data, rfm = run_preprocessing_pipeline()
