#!/usr/bin/env python3
"""
Generate Embeddings - Sentence Transformers Version
====================================================
Generates embeddings using Sentence Transformers (local, no API required).
Uses 'all-MiniLM-L6-v2' model - fast, efficient, and good quality.

Input:  embedding_pipeline/data/descriptions_preprocessed.csv
Output: embedding_pipeline/data/embeddings_sentencetransformer.npz
        embedding_pipeline/data/metadata_sentencetransformer.csv
"""

import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

print("=" * 70)
print("STEP 3B: GENERATE EMBEDDINGS (Sentence Transformers - Local)")
print("=" * 70)

# Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'  # 384-dim, fast, good quality
BATCH_SIZE = 256  # Adjust based on available memory
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n⚙️ Configuration:")
print(f"   Model: {MODEL_NAME}")
print(f"   Device: {DEVICE}")
print(f"   Batch size: {BATCH_SIZE}")

# Load model
print(f"\n📥 Loading Sentence Transformer model...")
print(f"   This may take a few minutes on first run (downloading model)...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"   ✅ Model loaded successfully")
print(f"   Embedding dimension: {embedding_dim}")

# Load preprocessed data
print("\n📂 Loading preprocessed descriptions...")
df = pd.read_csv('embedding_pipeline/data/descriptions_preprocessed.csv')
print(f"   Loaded {len(df):,} records")

# Generate embeddings
print(f"\n🔄 Generating embeddings...")
print(f"   Processing {len(df):,} descriptions")
print(f"   This will take some time depending on your hardware...")

descriptions = df['description'].tolist()

# Use model.encode with batch processing and progress bar
embeddings = model.encode(
    descriptions,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  # L2 normalization for better similarity computation
)

print(f"\n✅ Embedding generation complete!")
print(f"   Total embeddings: {len(embeddings):,}")
print(f"   Embedding shape: {embeddings.shape}")

# Save embeddings as compressed numpy array
print(f"\n💾 Saving embeddings...")
embeddings_path = 'embedding_pipeline/data/embeddings_sentencetransformer.npz'
np.savez_compressed(embeddings_path, embeddings=embeddings)
file_size = os.path.getsize(embeddings_path) / 1024**2
print(f"   Embeddings saved: {embeddings_path}")
print(f"   File size: {file_size:.2f} MB")

# Save metadata (IDs, years, targets) for reference
metadata_path = 'embedding_pipeline/data/metadata_sentencetransformer.csv'
metadata_df = df[['id', 'year', 'target']].copy()
metadata_df.to_csv(metadata_path, index=False)
print(f"   Metadata saved: {metadata_path}")

# Calculate and display statistics
print(f"\n📊 Embedding statistics:")
norms = np.linalg.norm(embeddings, axis=1)
print(f"   Mean embedding norm: {norms.mean():.4f}")
print(f"   Std embedding norm: {norms.std():.4f}")
print(f"   Min embedding value: {embeddings.min():.4f}")
print(f"   Max embedding value: {embeddings.max():.4f}")

# Show year distribution
print(f"\n📊 Data distribution by year:")
year_dist = df['year'].value_counts().sort_index()
for year, count in year_dist.items():
    exploited = df[df['year'] == year]['target'].sum()
    print(f"   {year}: {count:,} CVEs ({exploited} exploited)")

print("\n💡 Model Info:")
print(f"   Model: {MODEL_NAME}")
print(f"   Max sequence length: {model.max_seq_length} tokens")
print(f"   Embedding dimension: {embedding_dim}")
print(f"   Device used: {DEVICE}")

print("\n" + "=" * 70)
print("SENTENCE TRANSFORMER EMBEDDINGS COMPLETE - Ready for training")
print("=" * 70)
