import json
import os
import numpy as np
import argparse

def calculate_statistics(input_file, output_file):
    # Store the data read from file
    records = []
    
    # Read input JSONL file
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():  # Ensure the line is not empty
                records.append(json.loads(line.strip()))
    
    if not records:
        print(f"No records found in {input_file}.")
        return
    
    # Get dataset name (assuming all records have the same Dataset)
    dataset = records[-1]["Dataset"]
    
    # Filter the last 5 records with the same Dataset
    last_five_records = [record for record in records if record["Dataset"] == dataset][-5:]
    
    if len(last_five_records) < 5:
        print(f"Warning: Only found {len(last_five_records)} records for dataset '{dataset}'. Expected 5.")
    
    # Extract Acc and Auc data
    acc_values = [record["Test Acc"] for record in last_five_records]
    auc_values = [record["Test Auc"] for record in last_five_records]
    
    # Calculate mean and standard deviation
    acc_mean = np.mean(acc_values)
    acc_std = np.std(acc_values)
    auc_mean = np.mean(auc_values)
    auc_std = np.std(auc_values)
    
    # Construct result data
    result_data = {
        "Dataset": dataset,
        "Acc Mean": round(acc_mean, 6),
        "Acc Std": round(acc_std, 6),
        "Auc Mean": round(auc_mean, 6),
        "Auc Std": round(auc_std, 6)
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write results to output JSONL file
    with open(output_file, 'a') as f:
        f.write(json.dumps(result_data) + '\n')
    
    print(f"Statistics saved to {output_file}")

if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="Calculate statistics from results.jsonl")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset name to calculate statistics for")
    parser.add_argument("--path", type=str, default="result", help="The path where results.jsonl is located and statistics.jsonl will be saved")
    
    args = parser.parse_args()
    
    # Construct input and output file paths
    args.path = os.path.join(args.path, f'{args.dataset}')
    input_file = os.path.join(args.path, f'results.jsonl')  # Input file
    output_file = os.path.join(args.path, f'statistics.jsonl')  # Output file
    
    # Check if input file exists
    if not os.path.isfile(input_file):
        print(f"Input file {input_file} does not exist.")
    else:
        # Call statistics function
        calculate_statistics(input_file, output_file)