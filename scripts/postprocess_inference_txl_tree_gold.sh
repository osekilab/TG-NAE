#!/bin/bash

set -e  # Set to stop script immediately on error

# Activate virtual environment
if ! source .venv/bin/activate; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

INPUT_DIR="./outputs/seed=42/txl_tree_scores"
OUTPUT_DIR="./outputs/seed=42/txl_tree_scores.word"

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

# Error handling during parallel execution
if ! find "$INPUT_DIR" -type f -name "*.json" | \
    parallel --halt now,fail=1 --will-cite --bar --jobs $(nproc) process_all_files; then
    echo "Error: Parallel processing failed"
    exit 1
fi

echo "Initial processing complete!"

# Aggregation processing
METRICS_INPUT_DIR="$OUTPUT_DIR"
METRICS_OUTPUT_DIR="./outputs/seed=42/txl_tree_scores.word"

if ! python ./src/aggregate_beam_metrics.py \
    --input-dir "$METRICS_INPUT_DIR" \
    --output-dir "$METRICS_OUTPUT_DIR" \
    --use-weighted-nae; then
    echo "Error: Failed to aggregate metrics"
    exit 1
fi

# Token compression processing
ID_FILE="./data/naturalstories/ids/word_ids.csv"
FINAL_OUTPUT="./outputs/seed=42/txl_tree.csv"

if [ ! -f "$ID_FILE" ]; then
    echo "Error: ID file $ID_FILE does not exist"
    exit 1
fi

if ! python ./src/compress_word2treebankword.py \
    --id_file "$ID_FILE" \
    --sequence_dir "$METRICS_OUTPUT_DIR" \
    --output_file "$FINAL_OUTPUT"; then
    echo "Error: Failed to compress tokens"
    exit 1
fi

echo "All processing completed successfully!"
