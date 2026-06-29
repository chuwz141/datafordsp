import sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

# Deep dataset stats
print("=" * 80)
print("COMPREHENSIVE DATASET ANALYSIS")
print("=" * 80)

# 1. Reviews Purchase
print("\n### DATA_REVIEWS_PURCHASE ###")
df_r = pd.read_csv('dataset/before_EDA/data_reviews_purchase.csv')
print(f"Shape: {df_r.shape}")
print(f"Unique users: {df_r['user_id'].nunique()}")
print(f"Unique products: {df_r['product_id'].nunique()}")
print(f"Unique shops: {df_r['shop_id'].nunique()}")
print(f"Rating distribution:")
print(df_r['rating'].value_counts().sort_index())
print(f"\nDate range: {df_r['cmt_date'].min()} to {df_r['cmt_date'].max()}")
print(f"Missing values:")
print(df_r.isnull().sum())

# User purchase frequency
user_freq = df_r.groupby('user_id').size()
print(f"\nUser purchase frequency stats:")
print(user_freq.describe())
print(f"Users with >= 2 purchases: {(user_freq >= 2).sum()}")
print(f"Users with >= 3 purchases: {(user_freq >= 3).sum()}")
print(f"Users with >= 5 purchases: {(user_freq >= 5).sum()}")

# Check if same user buys multiple products (MBA feasibility)
user_products = df_r.groupby('user_id')['product_id'].nunique()
print(f"\nProducts per user stats:")
print(user_products.describe())
print(f"Users buying >= 2 unique products: {(user_products >= 2).sum()}")
print(f"Users buying >= 3 unique products: {(user_products >= 3).sum()}")

# Check processed_comment
print(f"\nProcessed comments non-null: {df_r['processed_comment'].notna().sum()}")
print(f"Processed comments null: {df_r['processed_comment'].isna().sum()}")
avg_comment_len = df_r['processed_comment'].dropna().str.len().mean()
print(f"Average comment length: {avg_comment_len:.1f} chars")

# 2. Products
print("\n\n### DATA_PRODUCT ###")
df_p = pd.read_csv('dataset/before_EDA/data_product.csv')
print(f"Shape: {df_p.shape}")
print(f"Unique products: {df_p['product_id'].nunique()}")
print(f"Unique shops: {df_p['shop_id'].nunique()}")
print(f"\nPrice stats:")
print(df_p['price'].describe())
print(f"\nMissing values:")
print(df_p.isnull().sum())
print(f"\nBrand distribution (top 10):")
print(df_p['brand'].value_counts().head(10))
print(f"\nType distribution:")
print(df_p['type'].value_counts())
print(f"\nProcessed description non-null: {df_p['processed_description'].notna().sum()}")
avg_desc_len = df_p['processed_description'].dropna().str.len().mean()
print(f"Average description length: {avg_desc_len:.1f} chars")

# 3. Product Attributes
print("\n\n### DATA_PRODUCT_ATTRIBUTE ###")
df_a = pd.read_csv('dataset/before_EDA/data_product_attribute.csv')
print(f"Shape: {df_a.shape}")
print(f"\nMissing values:")
print(df_a.isnull().sum())
print(f"\nSkin type distribution:")
print(df_a['skin_type'].value_counts().head(10))
print(f"\nOrigin distribution:")
print(df_a['origin'].value_counts().head(10))

# 4. Shop
print("\n\n### DATA_SHOP ###")
df_s = pd.read_csv('dataset/before_EDA/data_shop.csv')
print(f"Shape: {df_s.shape}")
print(f"\nMissing values:")
print(df_s.isnull().sum())
print(f"\nis_official_shop distribution:")
print(df_s['is_official_shop'].value_counts())
print(f"\nis_shopee_verified distribution:")
print(df_s['is_shopee_verified'].value_counts())

# 5. Check image_path
print("\n\n### IMAGE PATH CHECK ###")
print(f"image_path non-null: {df_p['image_path'].notna().sum()}")
print(f"image_path null: {df_p['image_path'].isna().sum()}")
exts = df_p['image_path'].dropna().str.split('.').str[-1].value_counts()
print(f"Image extensions:")
print(exts)

# 6. RFM Feasibility check
print("\n\n### RFM FEASIBILITY ###")
df_r['cmt_date'] = pd.to_datetime(df_r['cmt_date'])
snapshot_date = df_r['cmt_date'].max() + pd.Timedelta(days=1)
rfm = df_r.merge(df_p[['product_id', 'price']], on='product_id', how='left')
rfm_table = rfm.groupby('user_id').agg(
    recency=('cmt_date', lambda x: (snapshot_date - x.max()).days),
    frequency=('product_id', 'count'),
    monetary=('price', 'sum')
).reset_index()
print(f"\nRFM table shape: {rfm_table.shape}")
print(f"\nRFM stats:")
print(rfm_table[['recency', 'frequency', 'monetary']].describe())
