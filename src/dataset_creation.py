#!/usr/bin/env python3
import os
import csv
import subprocess
import re
import requests
import argparse
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Create dataset from GitHub issues')
parser.add_argument('--start-id', type=int, default=3, help='Starting bug ID (default: 3)')
parser.add_argument('--start-row', type=int, default=2, help='Starting row in CSV (default: 2, which is the 2nd data row after header)')
parser.add_argument('--end-row', type=int, default=None, help='Ending row in CSV (default: None, process all rows)')
parser.add_argument('--csv-file', type=str, default='issues.csv', help='CSV file with issues data (default: issues.csv)')
parser.add_argument('--dataset-dir', type=str, default='dataset', help='Base directory for the dataset (default: dataset)')
args = parser.parse_args()

# Base directory for the dataset
DATASET_DIR = args.dataset_dir

# Function to extract the issue body from GitHub issue URL
def get_issue_body(issue_url):
    api_url = issue_url.replace("github.com", "api.github.com/repos")
    
    # Add user-agent to avoid GitHub API limitations
    headers = {'User-Agent': 'Dataset-Creator-Script'}
    
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        issue_data = response.json()
        return issue_data.get('title', '') + '\n\n' + issue_data.get('body', '')
    else:
        return f"Error fetching issue: {response.status_code}"

# Create base dataset directory
os.makedirs(DATASET_DIR, exist_ok=True)

# Parse the input CSV file
with open(args.csv_file, 'r') as csv_file:
    reader = csv.reader(csv_file)
    rows = list(reader)
    
    # Skip header row
    header = rows[0]
    data_rows = rows[1:]
    
    # Validate row arguments
    if args.start_row < 1:
        print("Error: start-row must be at least 1")
        exit(1)
    
    if args.start_row > len(data_rows):
        print(f"Error: start-row ({args.start_row}) exceeds the number of data rows ({len(data_rows)})")
        exit(1)
    
    # Determine which rows to process
    start_idx = args.start_row - 1  # Convert to 0-based index
    end_idx = args.end_row if args.end_row is not None and args.end_row <= len(data_rows) else len(data_rows)
    
    # Initialize bug counter
    bug_counter = args.start_id
    
    print(f"Processing rows {args.start_row} to {end_idx} (out of {len(data_rows)} data rows)")
    print(f"Starting with bug ID: {bug_counter:03d}")
    
    # Process selected rows
    for i, row in enumerate(data_rows[start_idx:end_idx], start=start_idx+1):
        # Create bug folder with format 0xx
        bug_id = f"{bug_counter:03d}"
        print(f"Processing row {i}, bug {bug_id}...")
        
        issue_url = row[0]
        commit_hash = row[2]
        
        # Extract owner and repo from URL
        match = re.match(r'https://github.com/([^/]+)/([^/]+)', issue_url)
        if not match:
            print(f"Cannot parse URL: {issue_url}")
            continue
            
        owner, repo = match.groups()
        
        # Setup directory paths
        bug_dir = os.path.join(DATASET_DIR, bug_id)
        bug_report_dir = os.path.join(bug_dir, "bug_report")
        code_dir = os.path.join(bug_dir, "code")
        temp_repo_dir = os.path.join(DATASET_DIR, f"temp_{repo}")
        
        # Create the bug directory structure
        os.makedirs(bug_dir, exist_ok=True)
        os.makedirs(bug_report_dir, exist_ok=True)
        
        # Clone repository
        print(f"  Cloning {owner}/{repo}...")
        subprocess.run(["git", "clone", f"https://github.com/{owner}/{repo}.git", 
                        temp_repo_dir], check=True)
        
        # Checkout specific commit
        print(f"  Checking out commit {commit_hash}...")
        subprocess.run(["git", "-C", temp_repo_dir, "checkout", commit_hash], check=True)
        
        # Move the repo to code directory
        subprocess.run(["mv", temp_repo_dir, code_dir], check=True)
        
        # Create bug report
        print(f"  Creating bug report...")
        issue_body = get_issue_body(issue_url)
        with open(os.path.join(bug_report_dir, f"{bug_id}.txt"), 'w') as bug_file:
            bug_file.write(issue_body)
        
        print(f"  Completed bug {bug_id}")
        
        # Increment counter for next bug
        bug_counter += 1

print("Dataset creation completed!")