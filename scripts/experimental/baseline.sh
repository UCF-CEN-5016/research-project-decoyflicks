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

# Define array of models (using exact API/Ollama model names)
models=(
    "qwen2.5:7b"            # Ollama
    "deepseek-r1:7b"        # Ollama
    "qwen2.5-coder:7b"      # Ollama
    "llama3:8b"             # Ollama
    "llama-3.3-70b-versatile" # Groq
    "deepseek-reasoner"     # DeepSeek API (approx. 671B params)
    "gpt-4-turbo-2024-04-09" # OpenAI (mapped as gpt-4.1 in your tool previously)
)

techniques=("zero_shot" "few_shot" "cot")

# Helper function to determine backend based on model name
get_backend() {
    local model_name=$1
    if [[ "$model_name" == *"gpt"* ]]; then
        echo "openai"
    elif [[ "$model_name" == *"versatile"* ]]; then
        echo "groq"
    elif [[ "$model_name" == *"deepseek-reasoner"* ]]; then
        echo "deepseek"
    else
        echo "ollama"
    fi
}

# Loop from start_bug_id to end_bug_id
for (( bug_id=$start_bug_id; bug_id<=$end_bug_id; bug_id++ ))
do
    # Format bug_id as 3-digit number with leading zeros
    formatted_bug_id=$(printf "%03d" $bug_id)
    
    for model in "${models[@]}"
    do
        # Determine the backend for this model
        backend=$(get_backend "$model")
        
        for technique in "${techniques[@]}"
        do
            echo "Running: Bug ID $formatted_bug_id | Model: $model | Backend: $backend | Technique: $technique"
            
            # Run the python script with the new arguments
            python baselines.py \
                --bug_id "$formatted_bug_id" \
                --backend "$backend" \
                --model "$model" \
                --technique "$technique" \
                --examples 3 > "output_${formatted_bug_id}_${model//:/-}_${technique}.txt" 2>&1
        done
    done
done

echo "Completed all runs from bug_id $(printf "%03d" $start_bug_id) to bug_id $(printf "%03d" $end_bug_id)."