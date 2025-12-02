import os
import re
import subprocess
import sys

# Config
WEIGHTS_DIR = "RLMRec_results/my_weights"
PYTHON_EXEC = sys.executable # Use current python

def get_tasks():
    """
    Scans the weights directory and identifies unique (dataset, backbone) pairs.
    Assumes filename format: {model}-{dataset}-...
    and model names like 'lightgcn', 'lightgcn_plus', 'lightgcn_gene'.
    We want to group by the 'base' backbone.
    """
    if not os.path.exists(WEIGHTS_DIR):
        print(f"Error: Weights directory not found: {WEIGHTS_DIR}")
        return []

    files = os.listdir(WEIGHTS_DIR)
    tasks = set()

    # Regex to capture model and dataset
    # Matches: lightgcn-amazon-..., lightgcn_plus-amazon-...
    pattern = re.compile(r"([a-zA-Z0-9_]+)-([a-zA-Z0-9]+)-\d+-best-.*\.pth")

    for f in files:
        match = pattern.match(f)
        if match:
            full_model_name = match.group(1) # e.g., lightgcn_plus
            dataset = match.group(2)         # e.g., amazon
            
            # Determine backbone
            if full_model_name.endswith("_plus"):
                backbone = full_model_name.replace("_plus", "")
            elif full_model_name.endswith("_gene"):
                backbone = full_model_name.replace("_gene", "")
            else:
                backbone = full_model_name
            
            tasks.add((dataset, backbone))
            
    return sorted(list(tasks))

def find_weight_file(dataset, model_name):
    """Finds the specific .pth file for a given dataset and exact model name."""
    files = os.listdir(WEIGHTS_DIR)
    # Pattern to match exact model name at start
    # e.g. for model_name='lightgcn', we want 'lightgcn-amazon...', NOT 'lightgcn_plus-amazon...'
    # The previous regex approach might have been too loose.
    
    candidate = None
    for f in files:
        if f.startswith(f"{model_name}-{dataset}-"):
            return os.path.join(WEIGHTS_DIR, f)
    return None

def run_command(cmd):
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"Error executing command: {cmd}")

def main():
    print("=== RLMRec One-Click Analysis Runner ===")
    tasks = get_tasks()
    print(f"Found {len(tasks)} unique (Dataset, Backbone) pairs to analyze.")
    
    for i, (dataset, backbone) in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Processing: Dataset={dataset}, Backbone={backbone}")
        
        # 1. Dump Embeddings for all 3 variants
        variants = [backbone, f"{backbone}_plus", f"{backbone}_gene"]
        
        for variant in variants:
            weight_file = find_weight_file(dataset, variant)
            if weight_file:
                cmd = f"{PYTHON_EXEC} encoder/dump_emb.py --model {variant} --dataset {dataset} --weight_file {weight_file}"
                run_command(cmd)
            else:
                print(f"Warning: Weight file not found for {variant} on {dataset}. Skipping dump.")

        # 2. Run Analysis
        # This will look for saved_emb/{dataset}/{backbone}_item.npy etc.
        cmd_analysis = f"{PYTHON_EXEC} analysis.py --dataset {dataset} --backbone {backbone}"
        run_command(cmd_analysis)
        
        # 3. Run Correlation Analysis
        cmd_corr = f"{PYTHON_EXEC} analysis_correlation.py --dataset {dataset} --backbone {backbone}"
        run_command(cmd_corr)
        
    print("\n=== All Tasks Completed ===")
    print("Results are stored in 'analysis_results/{dataset}_{backbone}/'")

if __name__ == "__main__":
    main()
