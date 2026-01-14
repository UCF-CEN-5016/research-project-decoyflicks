#!/bin/bash

# Works on: macOS, Linux, Windows (Git Bash / WSL)
#
# Check if the correct number of arguments were passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 start_bug_id end_bug_id"
    exit 1
fi

start_bug_id=$1
end_bug_id=$2

# Validate that start and end are integers
if ! [[ "$start_bug_id" =~ ^[0-9]+$ ]] || ! [[ "$end_bug_id" =~ ^[0-9]+$ ]]; then
    echo "Error: Both start_bug_id and end_bug_id must be integers."
    exit 1
fi

# Define arrays of models and techniques to iterate over
models=("qwen2.5-7b" "deepseek-r1-7b" "qwen3-8b" "llama3-8b" "llama3-70b" "deepseek-r1-685b" "gpt-4.1" "qwen2.5-coder")
techniques=("zero_shot" "few_shot" "cot")

# Loop from start_bug_id to end_bug_id
for (( bug_id=$start_bug_id; bug_id<=$end_bug_id; bug_id++ ))
do
    # Format bug_id as 3-digit number with leading zeros
    formatted_bug_id=$(printf "%03d" $bug_id)
    
    for model in "${models[@]}"
    do
        for technique in "${techniques[@]}"
        do
            echo "Running script for bug_id $formatted_bug_id with model $model and technique $technique"
            # Change the Bug ID, Model, Prompting Technique, and Number of Examples (if applicable)
            python baselines.py --bug_id "$formatted_bug_id" --model "$model" --technique "$technique" --examples 3 > "output_${formatted_bug_id}_${model}_${technique}.txt" 2>&1
        done
    done
done

echo "Completed all runs from bug_id $(printf "%03d" $start_bug_id) to bug_id $(printf "%03d" $end_bug_id)."