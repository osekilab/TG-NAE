#!/bin/bash
# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_file)
            input_file="$2"
            shift 2
            ;;
        --brackets_dir)
            brackets_dir="$2"
            shift 2
            ;;
        --charniak_dir)
            charniak_dir="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --tokenized_dir)
            tokenized_dir="$2"
            shift 2
            ;;
        --csv_dir)
            csv_dir="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 --input_file <input_file> --brackets_dir <brackets_directory> --charniak_dir <charniak_directory> --output_dir <final_output_directory> --tokenized_dir <tokenized_directory> --csv_dir <csv_directory>"
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$input_file" ] || [ -z "$brackets_dir" ] || [ -z "$charniak_dir" ] || [ -z "$output_dir" ] || [ -z "$tokenized_dir" ] || [ -z "$csv_dir" ]; then
    echo "Usage: $0 --input_file <input_file> --brackets_dir <brackets_directory> --charniak_dir <charniak_directory> --output_dir <final_output_directory> --tokenized_dir <tokenized_directory> --csv_dir <csv_directory>"
    exit 1
fi

# Check if GNU parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "GNU parallel is not installed. Please install it first."
    echo "On Ubuntu/Debian: sudo apt-get install parallel"
    exit 1
fi

# Create directories
mkdir -p "$brackets_dir" "$charniak_dir" "$output_dir" "$tokenized_dir" "$csv_dir"

echo "Step 1: Completing brackets..."
python convert_rnng_trees.py --input "$input_file" --output_dir "$brackets_dir"

echo "Step 2: Converting to Choe-Charniak format..."
# Create temporary file for charniak conversion tasks
charniak_tasks=$(mktemp)

# Process each sentence directory for charniak conversion
for sent_dir in "$brackets_dir"/sent_*; do
    if [ -d "$sent_dir" ]; then
        sent_num=$(basename "$sent_dir")
        mkdir -p "$charniak_dir/$sent_num"

        for pointer_file in "$sent_dir"/pointer_*.txt; do
            if [ -f "$pointer_file" ]; then
                pointer_num=$(basename "$pointer_file")
                output_file="$charniak_dir/$sent_num/$pointer_num"
                echo "python tools/convert_to_choe_charniak.py --input \"$pointer_file\" --output \"$output_file\"" >> "$charniak_tasks"
            fi
        done
    fi
done

echo "Running Choe-Charniak conversion in parallel..."
parallel -j+0 --bar < "$charniak_tasks"
rm "$charniak_tasks"

echo "Step 3: Removing labeled brackets..."
# Create temporary file for bracket removal tasks
removal_tasks=$(mktemp)

# Process each sentence directory for bracket removal
for sent_dir in "$charniak_dir"/sent_*; do
    if [ -d "$sent_dir" ]; then
        sent_num=$(basename "$sent_dir")
        mkdir -p "$output_dir/$sent_num"

        for pointer_file in "$sent_dir"/pointer_*.txt; do
            if [ -f "$pointer_file" ]; then
                pointer_num=$(basename "$pointer_file")
                output_file="$output_dir/$sent_num/$pointer_num"
                echo "python remove_tail_right_bracket.py --input \"$pointer_file\" --output \"$output_file\"" >> "$removal_tasks"
            fi
        done
    fi
done

echo "Running bracket removal in parallel..."
parallel -j+0 --bar < "$removal_tasks"
rm "$removal_tasks"

echo "Step 4: Tokenizing files..."
# Check if spm_encode is available in PATH
if ! command -v spm_encode &> /dev/null; then
    echo "spm_encode command not found. Please ensure sentencepiece is installed and in your PATH"
    exit 1
fi

# Ensure SPM model path is provided and exists
if [ -z "$SPM" ]; then
    echo "Please set SPM environment variable pointing to the directory containing spm.model"
    exit 1
fi

if [ ! -f "${SPM}/spm.model" ]; then
    echo "spm.model not found at ${SPM}/spm.model"
    exit 1
fi

# Create temporary file for tokenization tasks
tokenize_tasks=$(mktemp)

# Process each sentence directory for tokenization
for sent_dir in "$output_dir"/sent_*; do
    if [ -d "$sent_dir" ]; then
        sent_num=$(basename "$sent_dir")
        mkdir -p "$tokenized_dir/$sent_num"

        for pointer_file in "$sent_dir"/pointer_*.txt; do
            if [ -f "$pointer_file" ]; then
                pointer_num=$(basename "$pointer_file")
                output_file="$tokenized_dir/$sent_num/$pointer_num"
                echo "spm_encode --output_format=id --model=${SPM}/spm.model --input=\"$pointer_file\" --output=\"$output_file\"" >> "$tokenize_tasks"
            fi
        done
    fi
done

echo "Running tokenization in parallel..."
parallel -j+0 --bar < "$tokenize_tasks"
rm "$tokenize_tasks"

echo "Step 5: Converting to CSV format..."
# Check if vocabulary file exists
if [ ! -f "${SPM}/spm.vocab" ]; then
    echo "spm.vocab not found at ${SPM}/spm.vocab"
    exit 1
fi

# Create temporary file for CSV conversion tasks
csv_tasks=$(mktemp)

# Process each sentence directory for CSV conversion
for sent_dir in "$tokenized_dir"/sent_*; do
    if [ -d "$sent_dir" ]; then
        sent_num=$(basename "$sent_dir")
        mkdir -p "$csv_dir/$sent_num"

        for pointer_file in "$sent_dir"/pointer_*.txt; do
            if [ -f "$pointer_file" ]; then
                pointer_num=$(basename "$pointer_file")
                output_file="$csv_dir/$sent_num/${pointer_num%.txt}.csv"  # Change extension to .csv
                echo "python tools/postprocess_encoded_docs.py --input \"$pointer_file\" --output \"$output_file\" --vocab \"${SPM}/spm.vocab\"" >> "$csv_tasks"
            fi
        done
    fi
done

echo "Running CSV conversion in parallel..."
parallel -j+0 --bar < "$csv_tasks"
rm "$csv_tasks"

echo "Pipeline completed successfully!"
echo "Results are saved in:"
echo "- Bracket completion: $brackets_dir"
echo "- Choe-Charniak format: $charniak_dir"
echo "- Intermediate output: $output_dir"
echo "- Tokenized output: $tokenized_dir"
echo "- Final CSV output: $csv_dir"
