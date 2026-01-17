import re
import os
from pathlib import Path
from collections import defaultdict

# Read raw results files
results_dir = Path(__file__).parent
output_dir = results_dir / "results_cleaned"
output_dir.mkdir(exist_ok=True)

# Methods to process
methods = ["msp", "maxlogit", "maxeEntropy", "rba"]

def parse_results_file(filepath):
    """Parse raw results file and extract structured data"""
    results = []
    current_entry = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith("DATASET"):
            if current_entry and 'auprc' in current_entry:
                results.append(current_entry)
            current_entry = {'dataset': line.replace("DATASET", "").strip()}
        elif line.startswith("METHOD"):
            current_entry['method'] = line.replace("METHOD", "").strip()
        elif line.startswith("TEMP"):
            temp_str = line.replace("TEMP", "").strip()
            current_entry['temp'] = float(temp_str)
        elif "AUPRC score:" in line:
            # Extract AUPRC and FPR@95TPR from same line
            match = re.search(r'AUPRC score:([\d.]+)\s+FPR@TPR95:([\d.]+)', line)
            if match:
                current_entry['auprc'] = float(match.group(1))
                current_entry['fpr95'] = float(match.group(2))
    
    # Add last entry
    if current_entry and 'auprc' in current_entry:
        results.append(current_entry)
    
    return results

# Process each method
all_results = defaultdict(list)

for method in methods:
    filepath = results_dir / f"results_{method}.txt"
    if filepath.exists():
        print(f"Processing {method}...")
        results = parse_results_file(filepath)
        
        # Filter by actual method (remove mismatched entries)
        filtered = [r for r in results if r.get('method') == method or r.get('method') is None]
        all_results[method] = results
        
        print(f"  Found {len(results)} entries")

# Clean and reorganize by method
cleaned_results = defaultdict(lambda: defaultdict(dict))

for method, entries in all_results.items():
    for entry in entries:
        dataset = entry.get('dataset', '').replace('/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject ', '')
        dataset = dataset.replace('dataset/anomaly/Validation_Dataset/', '').replace('/images/*.*', '')
        method_name = entry.get('method', method).lower()
        temp = entry.get('temp', 1.0)
        
        key = (dataset, temp)
        cleaned_results[method_name][key] = {
            'auprc': entry.get('auprc'),
            'fpr95': entry.get('fpr95')
        }

# Save cleaned results per method
for method in ["msp", "maxlogit", "maxentropy", "rba"]:
    output_file = output_dir / f"{method}_results.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"METHOD: {method.upper()} - CLEANED RESULTS\n")
        f.write(f"{'='*80}\n\n")
        
        # Sort by dataset and temperature
        if method in cleaned_results:
            by_dataset = defaultdict(dict)
            for (dataset, temp), metrics in sorted(cleaned_results[method].items()):
                by_dataset[dataset][temp] = metrics
            
            for dataset in sorted(by_dataset.keys()):
                f.write(f"\nDATASET: {dataset}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"{'Temperature':<15} {'AUPRC':<20} {'FPR@95TPR':<20}\n")
                f.write(f"{'-'*80}\n")
                
                for temp in sorted(by_dataset[dataset].keys()):
                    metrics = by_dataset[dataset][temp]
                    auprc = metrics.get('auprc', 'N/A')
                    fpr95 = metrics.get('fpr95', 'N/A')
                    
                    if isinstance(auprc, float):
                        auprc_str = f"{auprc:.2f}%"
                    else:
                        auprc_str = str(auprc)
                    
                    if isinstance(fpr95, float):
                        fpr95_str = f"{fpr95:.2f}%"
                    else:
                        fpr95_str = str(fpr95)
                    
                    f.write(f"{temp:<15.2f} {auprc_str:<20} {fpr95_str:<20}\n")

print(f"\n✅ Cleaned results saved to {output_dir}")
