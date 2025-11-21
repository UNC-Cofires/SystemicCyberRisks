#!/bin/bash
# Master script to run the complete embedding pipeline
# Usage: ./embedding_pipeline/run_full_pipeline.sh [sentencetransformer|openai]

set -e  # Exit on error

EMBEDDING_TYPE=${1:-sentencetransformer}

echo "======================================================================="
echo "RUNNING COMPLETE EMBEDDING PIPELINE"
echo "Embedding Type: $EMBEDDING_TYPE"
echo "======================================================================="

# Activate virtual environment
source venv/bin/activate

echo ""
echo "STEP 1/4: Extracting descriptions..."
python embedding_pipeline/1_extract_descriptions.py

echo ""
echo "STEP 2/4: Preprocessing text..."
python embedding_pipeline/2_preprocess_text.py

echo ""
echo "STEP 3/4: Generating embeddings ($EMBEDDING_TYPE)..."
if [ "$EMBEDDING_TYPE" = "openai" ]; then
    # Check for API key
    if [ ! -f .env ]; then
        echo "ERROR: .env file not found!"
        echo "Please create .env file with your OPENAI_API_KEY"
        echo "See .env.template for reference"
        exit 1
    fi
    python embedding_pipeline/3a_generate_embeddings_openai.py
else
    python embedding_pipeline/3b_generate_embeddings_sentencetransformer.py
fi

echo ""
echo "STEP 4/4: Training models..."
python embedding_pipeline/4_train_models.py --embedding-type "$EMBEDDING_TYPE"

echo ""
echo "======================================================================="
echo "PIPELINE COMPLETE!"
echo "======================================================================="
echo ""
echo "Results saved in:"
echo "  - embedding_pipeline/results/"
echo "  - embedding_pipeline/visualizations/"
echo ""
echo "Key files:"
echo "  - results/results_${EMBEDDING_TYPE}.json"
echo "  - visualizations/model_comparison_${EMBEDDING_TYPE}.png"
echo ""
