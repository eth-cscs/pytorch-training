import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results.csv")

# Sort for consistent plots
df = df.sort_values(["method", "gpus"])

# Plot 1: Time per epoch vs GPUs
plt.figure()
for method in df["method"].unique():
    subset = df[df["method"] == method]
    plt.plot(subset["gpus"], subset["time"], marker="o", label=method.upper())

# Add ideal scaling line (normalized to 1-GPU baseline)
baseline_time = df[df["gpus"] == 1]["time"].mean()
ideal_gpus = sorted(df["gpus"].unique())
ideal_time = [baseline_time / g for g in ideal_gpus]
plt.plot(ideal_gpus, ideal_time, "--", color="gray", label="Ideal (1/N)")

plt.xscale("log", base=2)
plt.yscale("log")
plt.title("Time per Epoch vs GPUs")
plt.xlabel("GPUs")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.legend()
plt.savefig("plot_time_per_epoch.png")

# Plot 2: Validation Loss vs GPUs
plt.figure()
for method in df["method"].unique():
    subset = df[df["method"] == method]
    plt.plot(subset["gpus"], subset["val_loss"], marker="o", label=method.upper())

plt.title("Validation Loss vs GPUs")
plt.xlabel("GPUs")
plt.ylabel("Validation Loss")
plt.grid(True)
plt.legend()
plt.savefig("plot_val_loss.png")

# Plot 3: Accuracy vs GPUs
plt.figure()
for method in df["method"].unique():
    subset = df[df["method"] == method]
    plt.plot(subset["gpus"], subset["acc"], marker="o", label=method.upper())

plt.title("Accuracy vs GPUs")
plt.xlabel("GPUs")
plt.ylabel("Accuracy [%]")
plt.grid(True)
plt.legend()
plt.savefig("plot_accuracy.png")

# Plot 4: Peak Memory per GPU vs GPUs
plt.figure()
for method in df["method"].unique():
    subset = df[df["method"] == method]
    mem_per_gpu = subset["peak_mem"] / subset["gpus"]
    plt.plot(subset["gpus"], mem_per_gpu, marker="o", label=method.upper())

plt.xscale("log", base=2)
plt.yscale("log")
plt.title("Peak Memory per GPU vs GPUs")
plt.xlabel("GPUs")
plt.ylabel("Memory per GPU (MB)")
plt.grid(True)
plt.legend()
plt.savefig("plot_peak_mem_per_gpu.png")

print("✅ Plots saved as plot_*.png")
