import os
import re
import pandas as pd

log_dir = "logs"
output_csv = "results.csv"

# Delete existing CSV if present
if os.path.exists(output_csv):
    os.remove(output_csv)

# Store parsed rows
rows = []

# Regex to match [METRIC] lines
metric_re = re.compile(
    r"\[METRIC\] epoch=(\d+) time=([\d\.]+) val_loss=([\d\.]+) acc=([\d\.]+) peak_mem=(\d+)MB"
)

for filename in os.listdir(log_dir):
    if not filename.endswith(".out"):
        continue

    # Extract method and GPU count from filename (e.g., fsdp_4gpu.out)
    match_file = re.match(r"(\w+)_(\d+)gpu\.out", filename)
    if not match_file:
        continue

    method, gpus = match_file.group(1), int(match_file.group(2))

    with open(os.path.join(log_dir, filename)) as f:
        for line in f:
            if "[METRIC]" in line:
                last_metric_line = line

    if last_metric_line:
            match = metric_re.search(last_metric_line)
            if match:
                epoch, time, val_loss, acc, peak_mem = match.groups()
                rows.append({
                    "method": method,
                    "gpus": gpus,
                    "epoch": int(epoch),
                    "time": float(time),
                    "val_loss": float(val_loss),
                    "acc": float(acc),
                    "peak_mem": int(peak_mem),
                })

    # Write to CSV using pandas
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Final metrics extracted and saved to {output_csv}")

