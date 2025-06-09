# this is for comparing the performance of Medium and Smart submodels
# It reads a log file, extracts the number of parameters and test accuracy for each submodel,
# and then plots the results in a scatter plot.
# It also calculates and prints the mean accuracy for both submodels.


# import re
# import matplotlib.pyplot as plt

# # File path
# log_file = "/work/LAS/jannesar-lab/msamani/SuperSAM/logs/2025-06-05--23:01:15.249238_dataset[zh-plus-tiny-imagenet]_trainable[em]_epochs[50]_lr[5e-05]_bs[64]/filtered_log.log"  # Replace with your actual file

# # Storage
# medium_params, medium_acc = [], []
# smart_params, smart_acc = [], []

# # Read and parse the log
# with open(log_file, 'r') as f:
#     for line in f:
#         # Regex to extract info
#         match = re.search(
#             r'(Medium|Smart) submodel.*?params: ([\d.]+).*?test_accuracy: ([\d.]+)',
#             line
#         )
#         if match:
#             model_type = match.group(1)
#             params = float(match.group(2))
#             accuracy = float(match.group(3))
#             if model_type == "Medium":
#                 medium_params.append(params)
#                 medium_acc.append(accuracy)
#             else:
#                 smart_params.append(params)
#                 smart_acc.append(accuracy)

# # Plotting
# mean_medium = sum(medium_acc) / len(medium_acc)
# mean_smart = sum(smart_acc) / len(smart_acc)
# print(f"Mean Medium Accuracy: {mean_medium:.4f}")
# print(f"Mean Smart Accuracy: {mean_smart:.4f}")
# plt.figure(figsize=(8, 6))
# plt.scatter(medium_params, medium_acc, color='blue', label='Medium', marker='o')
# plt.scatter(smart_params, smart_acc, color='green', label='Smart', marker='x')
# plt.xlabel("Number of Parameters (Millions)")
# plt.ylabel("Test Accuracy")
# plt.title("Model Test Accuracy vs Parameters")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# plt.savefig("accuracy_vs_params.png")
# print(22)








# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import re

# # Load log file
# log_file = "/work/LAS/jannesar-lab/msamani/SuperSAM/logs/2025-06-08--21:15:39.492552_dataset[imagenet-1k]_trainable[em]_epochs[50]_lr[5e-05]_bs[64]/experiment_log-2025-06-08-21:15-39-results.log"
# with open(log_file, "r") as f:
#     lines = f.readlines()

# # Fixed regex pattern (handles spaces properly)
# pattern = re.compile(
#     r"Layer (\d+), Head (\d+),\s*Score1: ([\d.]+),\s*Score2: ([\d.]+),\s*Score3: ([\d.]+),\s*Score4: ([\d.]+),\s*Accuracy: ([\d.]+)%"
# )

# data = []
# for line in lines:
#     match = pattern.search(line)
#     if match:
#         layer = int(match.group(1))
#         head = int(match.group(2))
#         scores = list(map(float, match.groups()[2:6]))
#         accuracy_percent = float(match.group(7))  # DO NOT divide by 100 again
#         accuracy = accuracy_percent      # convert % to decimal
#         print(f"Layer: {layer}, Head: {head}, Scores: {scores}, Accuracy: {accuracy:.4f}")
#         data.append({
#             "Layer": layer,
#             "Head": head,
#             "Score1": scores[0],
#             "Score2": scores[1],
#             "Score3": scores[2],
#             "Score4": scores[3],
#             "Accuracy": accuracy
#         })

# df = pd.DataFrame(data)
# baseline_accuracy = 0.72





# # Plot: each score vs accuracy
# plt.figure(figsize=(16, 10))
# score_types = ["Score1", "Score2", "Score3", "Score4"]

# for i, score in enumerate(score_types, 1):
#     plt.subplot(2, 2, i)
#     sns.scatterplot(x=score, y="Accuracy", data=df, s=60)
#     sns.regplot(x=score, y="Accuracy", data=df, scatter=False, line_kws={"color": "red"})
#     plt.axhline(y=baseline_accuracy, color='gray', linestyle='--', label='Baseline Accuracy (0.72)')
#     plt.xlabel(score)
#     plt.ylabel("Accuracy After Dropping")
#     plt.title(f"{score} vs Accuracy")
#     plt.legend()

# plt.tight_layout()
# plt.savefig("score_vs_accuracy_fixed.png")
# plt.show()








import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

# Load log file
log_file = "/work/LAS/jannesar-lab/msamani/SuperSAM/logs/2025-06-08--21:15:39.492552_dataset[imagenet-1k]_trainable[em]_epochs[50]_lr[5e-05]_bs[64]/experiment_log-2025-06-08-21:15-39-results.log"
with open(log_file, "r") as f:
    lines = f.readlines()

# Fixed regex pattern (handles spaces properly)
pattern = re.compile(
    r"Layer (\d+), Head (\d+),\s*Score1: ([\d.]+),\s*Score2: ([\d.]+),\s*Score3: ([\d.]+),\s*Score4: ([\d.]+),\s*Accuracy: ([\d.]+)%"
)

data = []
for line in lines:
    match = pattern.search(line)
    if match:
        layer = int(match.group(1))
        head = int(match.group(2))
        scores = list(map(float, match.groups()[2:6]))
        accuracy_percent = float(match.group(7))  # DO NOT divide by 100 again
        accuracy = accuracy_percent      # keep as is since it's already in percentage
        print(f"Layer: {layer}, Head: {head}, Scores: {scores}, Accuracy: {accuracy:.4f}")
        data.append({
            "Layer": layer,
            "Head": head,
            "Score1": scores[0],
            "Score2": scores[1],
            "Score3": scores[2],
            "Score4": scores[3],
            "Accuracy": accuracy
        })

df = pd.DataFrame(data)
baseline_accuracy = 0.72

# Plot: each score vs accuracy
plt.figure(figsize=(16, 10))
score_types = ["Score1", "Score2", "Score3", "Score4"]

for i, score in enumerate(score_types, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=score, y="Accuracy", data=df, s=60)
    sns.regplot(x=score, y="Accuracy", data=df, scatter=False, line_kws={"color": "red"})
    plt.axhline(y=baseline_accuracy, color='gray', linestyle='--', label='Baseline Accuracy (0.72)')
    plt.xlabel(score)
    plt.ylabel("Accuracy After Dropping")
    plt.title(f"{score} vs Accuracy")
    plt.legend()

plt.tight_layout()
plt.savefig("score_vs_accuracy_fixed.png")
plt.close()

# New chart: Bar plot of correlation coefficients
plt.figure(figsize=(8, 6))
correlations = df[score_types].corrwith(df["Accuracy"])
sns.barplot(x=correlations.index, y=correlations.values)
plt.title("Influence of Scores on Accuracy (Correlation Coefficients)")
plt.xlabel("Score Type")
plt.ylabel("Correlation with Accuracy")
plt.axhline(y=0, color='gray', linestyle='--')
plt.savefig("score_influence_correlation.png")
plt.show()
