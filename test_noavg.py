import json
import os
import numpy as np

# Base paths for the experiments (add as many as you like)
experiments = {
    "EEGProjectLayer_RN50": "/work3/s193209/data/ubp_exp/eeg_intra-subject_ubp_EEGProjectLayer_RN50",
}

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
results = {exp_name: {} for exp_name in experiments}

# Parse JSONs
for exp_name, base_path in experiments.items():
    for sub in subjects:
        json_path = os.path.join(base_path, f"{sub}_seed0", "test_results.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                entry = data[0] if isinstance(data, list) else data
                top1 = entry.get("test_top1_acc", 0.0) * 100
                top5 = entry.get("test_top5_acc", 0.0) * 100
                results[exp_name][sub] = (top1, top5)
        else:
            results[exp_name][sub] = (np.nan, np.nan)

# Compute Mean ± STD dynamically
stats = {}
for exp_name in experiments:
    t1 = [v[0] for v in results[exp_name].values() if not np.isnan(v[0])]
    t5 = [v[1] for v in results[exp_name].values() if not np.isnan(v[1])]
    m1, s1 = np.mean(t1), (np.std(t1, ddof=1) if len(t1) > 1 else 0.0)
    m5, s5 = np.mean(t5), (np.std(t5, ddof=1) if len(t5) > 1 else 0.0)
    stats[exp_name] = (f"{m1:.2f} ± {s1:.2f}%", f"{m5:.2f} ± {s5:.2f}%")

# --- Dynamic Formatted Table Output ---
sub_col_w = 14
metric_col_w = 18
exp_block_w = metric_col_w * 2 + 3  # Two metric columns + separator spacing

# Construct table separators and headers dynamically
all_col_widths = [sub_col_w] + [metric_col_w] * (2 * len(experiments))
sep_line = "|" + "|".join("-" * (w + 2) for w in all_col_widths) + "|"

# Header Line 1: Experiment Names
header_1_parts = [f"| {'Subject':<{sub_col_w}} "]
for exp_name in experiments:
    header_1_parts.append(f"| {exp_name:^{exp_block_w}} ")
header_1 = "".join(header_1_parts) + "|"

# Header Line 2: Top-1 and Top-5 sub-headers
header_2_parts = [f"| {'':<{sub_col_w}} "]
for _ in experiments:
    header_2_parts.append(f"| {'Top-1 Acc (%)':^{metric_col_w}} | {'Top-5 Acc (%)':^{metric_col_w}} ")
header_2 = "".join(header_2_parts) + "|"

# Print Table
print("\n" + "=" * len(sep_line))
print(header_1)
print(header_2)
print(sep_line)

# Subject Rows
for sub in subjects:
    row_parts = [f"| {sub:<{sub_col_w}} "]
    for exp_name in experiments:
        r = results[exp_name].get(sub, (np.nan, np.nan))
        t1_str = f"{r[0]:.2f}%" if not np.isnan(r[0]) else "N/A"
        t5_str = f"{r[1]:.2f}%" if not np.isnan(r[1]) else "N/A"
        row_parts.append(f"| {t1_str:^{metric_col_w}} | {t5_str:^{metric_col_w}} ")
    print("".join(row_parts) + "|")

print(sep_line)

# Mean ± STD Row
stats_parts = [f"| {'Mean ± STD':<{sub_col_w}} "]
for exp_name in experiments:
    m1_str, m5_str = stats[exp_name]
    stats_parts.append(f"| {m1_str:^{metric_col_w}} | {m5_str:^{metric_col_w}} ")
print("".join(stats_parts) + "|")
print("=" * len(sep_line) + "\n")