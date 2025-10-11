#!/bin/bash

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --csv_dir)
            csv_dir="$2"
            shift 2
            ;;
        --score_output_dir)
            score_output_dir="$2"
            shift 2
            ;;
        --checkpoint)
            checkpoint="$2"
            shift 2
            ;;
        --tokenizer)
            tokenizer="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 --csv_dir <csv_directory> --score_output_dir <score_output_directory> --checkpoint <checkpoint_file> --tokenizer <tokenizer_model>"
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$csv_dir" ] || [ -z "$score_output_dir" ] || [ -z "$checkpoint" ] || [ -z "$tokenizer" ]; then
    echo "Usage: $0 --csv_dir <csv_directory> --score_output_dir <score_output_directory> --checkpoint <checkpoint_file> --tokenizer <tokenizer_model>"
    exit 1
fi

# Create temporary and output directories
temp_dir=$(mktemp -d)
mkdir -p "$score_output_dir"

echo "Step 1: Concatenating all files..."
# Create a mapping file to track original file locations
mapping_file="$temp_dir/file_mapping.txt"
combined_file="$temp_dir/combined_input.csv"
touch "$mapping_file"
touch "$combined_file"

# Concatenate all files and create mapping
line_count=0
for sent_dir in "$csv_dir"/sent_*; do
    if [ -d "$sent_dir" ]; then
        sent_num=$(basename "$sent_dir")
        for pointer_file in "$sent_dir"/pointer_*.csv; do
            if [ -f "$pointer_file" ]; then
                pointer_num=$(basename "$pointer_file" .csv)

                # Count lines in current file
                current_lines=$(wc -l < "$pointer_file")

                # Add file mapping entry
                echo "$sent_num,$pointer_num,$line_count,$current_lines" >> "$mapping_file"

                # Concatenate file content
                cat "$pointer_file" >> "$combined_file"

                # Update line counter
                line_count=$((line_count + current_lines))
            fi
        done
    fi
done

echo "Step 2: Processing combined file..."
# Process the combined file
combined_output="$temp_dir/combined_output"
python ./src/syntactic_attention_based_metric_transformer_grammars/score.py --checkpoint "$checkpoint" --tokenizer "$tokenizer" --input "$combined_file" --output "$combined_output"

echo "Step 3: Distributing results..."
# Read the mapping file to get the start line and number of lines for each original file
current_sequence=1
while IFS=, read -r sent_num pointer_num start_line num_lines; do
    # Create output directory for this pointer
    output_dir="$score_output_dir/$sent_num/$pointer_num"
    mkdir -p "$output_dir"

    # For each line in the original file, copy corresponding sequence file
    for ((i=0; i<num_lines; i++)); do
        if [ -f "$combined_output/sequence_${current_sequence}.json" ]; then
            cp "$combined_output/sequence_${current_sequence}.json" "$output_dir/sequence_$((i+1)).json"
            echo "Copied sequence_${current_sequence}.json to $output_dir/sequence_$((i+1)).json"
        else
            echo "Warning: sequence_${current_sequence}.json not found"
        fi
        ((current_sequence++))
    done
done < "$mapping_file"

# Cleanup
rm -rf "$temp_dir"

echo "Processing completed successfully!"
echo "Results are saved in: $score_output_dir"
