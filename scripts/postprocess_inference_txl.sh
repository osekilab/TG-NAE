#!/bin/bash

set -e  # Set script to stop immediately on error

# Activate virtual environment
if ! source .venv/bin/activate; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

INPUT_DIR="./outputs/seed=42/txl_scores"
OUTPUT_DIR="./outputs/seed=42/txl_scores.word"

# Check directory existence and create
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# File processing function
process_all_files() {
    local json_file=$1
    local rel_path=${json_file#$INPUT_DIR/}
    local dir_path=$(dirname "$rel_path")
    local output_file="$OUTPUT_DIR/${rel_path%.json}.csv"

    mkdir -p "$OUTPUT_DIR/$dir_path"

    if ! python ./src/compress_subword2word.py --input "$json_file" --output "$output_file"; then
        echo "Error: Failed to process file: $json_file"
        exit 1
    fi
}
export -f process_all_files
export INPUT_DIR
export OUTPUT_DIR

# Error handling for parallel execution
if ! find "$INPUT_DIR" -type f -name "*.json" | \
    parallel --halt now,fail=1 --will-cite --bar --jobs $(nproc) process_all_files; then
    echo "Error: Parallel processing failed"
    exit 1
fi

echo "Initial processing complete!"

# No aggregation step here: the Transformer reads token sequences alone, so
# there is a single score per token rather than one per syntactic structure.

# Token compression processing
ID_FILE="./data/naturalstories/ids/word_ids.csv"
FINAL_OUTPUT="./outputs/seed=42/txl.csv"

if [ ! -f "$ID_FILE" ]; then
    echo "Error: ID file $ID_FILE does not exist"
    exit 1
fi

if ! python ./src/compress_word2treebankword.py \
    --id_file "$ID_FILE" \
    --sequence_dir "$OUTPUT_DIR" \
    --output_file "$FINAL_OUTPUT"; then
    echo "Error: Failed to compress tokens"
    exit 1
fi

echo "All processing completed successfully!"
