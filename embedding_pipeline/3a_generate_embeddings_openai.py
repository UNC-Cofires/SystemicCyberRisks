#!/usr/bin/env python3
"""
Generate Embeddings - OpenAI API Version
=========================================
Generates embeddings using OpenAI's text-embedding-3-small model.
Requires OPENAI_API_KEY in .env file.

Input:  embedding_pipeline/data/descriptions_preprocessed.csv
Output: embedding_pipeline/data/embeddings_openai.npz
        embedding_pipeline/data/metadata_openai.csv

Setup:
    1. Create .env file in project root with: OPENAI_API_KEY=your_key_here
    2. Run this script: python embedding_pipeline/3a_generate_embeddings_openai.py
"""

import pandas as pd
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import time

print("=" * 70)
print("STEP 3A: GENERATE EMBEDDINGS (OpenAI API)")
print("=" * 70)

# Load environment variables
print("\n🔑 Loading API credentials...")
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("❌ ERROR: OPENAI_API_KEY not found in .env file!")
    print("\n📝 Setup instructions:")
    print("   1. Create a .env file in the project root directory")
    print("   2. Add line: OPENAI_API_KEY=your_actual_api_key_here")
    print("   3. Get API key from: https://platform.openai.com/api-keys")
    exit(1)

print("   ✅ API key loaded successfully")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 1000  # OpenAI supports large batches
RATE_LIMIT_DELAY = 0.1  # seconds between batches to avoid rate limits

print(f"\n⚙️ Configuration:")
print(f"   Model: {EMBEDDING_MODEL}")
print(f"   Embedding dimension: {EMBEDDING_DIM}")
print(f"   Batch size: {BATCH_SIZE}")

# Load preprocessed data
print("\n📂 Loading preprocessed descriptions...")
df = pd.read_csv('embedding_pipeline/data/descriptions_preprocessed.csv')
print(f"   Loaded {len(df):,} records")

def get_embeddings_batch(texts, model=EMBEDDING_MODEL):
    """Get embeddings for a batch of texts using OpenAI API."""
    try:
        response = client.embeddings.create(
            model=model,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"\n   ⚠️  Error in batch: {str(e)}")
        return None

# Generate embeddings in batches
print(f"\n🔄 Generating embeddings...")
print(f"   Processing {len(df):,} descriptions in batches of {BATCH_SIZE}")

all_embeddings = []
failed_indices = []

for i in tqdm(range(0, len(df), BATCH_SIZE), desc="   Embedding batches"):
    batch_end = min(i + BATCH_SIZE, len(df))
    batch_texts = df['description'].iloc[i:batch_end].tolist()

    embeddings = get_embeddings_batch(batch_texts)

    if embeddings:
        all_embeddings.extend(embeddings)
    else:
        # If batch fails, try one by one
        print(f"\n   Retrying batch {i}-{batch_end} individually...")
        for j, text in enumerate(batch_texts):
            single_emb = get_embeddings_batch([text])
            if single_emb:
                all_embeddings.append(single_emb[0])
            else:
                all_embeddings.append([0.0] * EMBEDDING_DIM)
                failed_indices.append(i + j)

    # Rate limiting
    \if i + BATCH_SIZE < leb v 
    n(df):
        time.sleep(RATE_LIMIT_DELAY)

# Convert to numpy array
embeddings_array = np.array(all_embeddings, dtype=np.float32)

print(f"\n✅ Embedding generation complete!")
print(f"   Total embeddings: {len(embeddings_array):,}")
print(f"   Embedding shape: {embeddings_array.shape}")
print(f"   Failed embeddings: {len(failed_indices)}")

# Save embeddings as compressed numpy array
print(f"\n💾 Saving embeddings...")
embeddings_path = 'embedding_pipeline/data/embeddings_openai.npz'
np.savez_compressed(embeddings_path, embeddings=embeddings_array)
file_size = os.path.getsize(embeddings_path) / 1024**2
print(f"   Embeddings saved: {embeddings_path}")
print(f"   File size: {file_size:.2f} MB")

# Save metadata (IDs, years, targets) for reference
metadata_path = 'embedding_pipeline/data/metadata_openai.csv'
metadata_df = df[['id', 'year', 'target']].copy()
metadata_df.to_csv(metadata_path, index=False)
print(f"   Metadata saved: {metadata_path}")

# Calculate and display statistics
print(f"\n📊 Embedding statistics:")
print(f"   Mean embedding norm: {np.linalg.norm(embeddings_array, axis=1).mean():.4f}")
print(f"   Std embedding norm: {np.linalg.norm(embeddings_array, axis=1).std():.4f}")
print(f"   Min embedding value: {embeddings_array.min():.4f}")
print(f"   Max embedding value: {embeddings_array.max():.4f}")

# Show cost estimate (approximate)
total_tokens = df['description'].str.split().str.len().sum()
estimated_cost = (total_tokens / 1000) * 0.00002  # $0.00002 per 1K tokens for text-embedding-3-small
print(f"\n💰 Estimated API cost:")
print(f"   Total tokens processed: ~{total_tokens:,}")
print(f"   Estimated cost: ~${estimated_cost:.2f} USD")

print("\n" + "=" * 70)
print("OPENAI EMBEDDINGS COMPLETE - Ready for training")
print("=" * 70)
