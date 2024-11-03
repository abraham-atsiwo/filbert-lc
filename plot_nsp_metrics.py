import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def rename_model_name(name):
    name = name.lower()
    if name == "bert-base-uncased":
        return "BERT-Small"
    elif name == "bert-large-uncased":
        return "BERT-Large"
    return "PBERT NSP"

def calculate_size(size):
    return size*2

metrics = pd.read_csv("results_nsp/metrics.csv")
metrics["model"] = metrics["model_name"].apply(func=rename_model_name)
metrics["size"] = metrics["size"].apply(func=calculate_size)
df = metrics

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot size by accuracy, grouped by model
sns.lineplot(ax=axes[0, 0], x='size', y='accuracy', hue='model', data=df, marker='o', linewidth=2.5)
axes[0, 0].set_title('Test Size vs Accuracy')
axes[0, 0].set_xlabel('Test Size')
axes[0, 0].set_ylabel('Accuracy')

# Plot size by loss, grouped by model
sns.lineplot(ax=axes[0, 1], x='size', y='loss', hue='model', data=df, marker='o', linewidth=2.5)
axes[0, 1].set_title('Test Size vs Loss')
axes[0, 1].set_xlabel('Test Size')
axes[0, 1].set_ylabel('Loss')

# Plot size by f1_macro, grouped by model
sns.lineplot(ax=axes[1, 0], x='size', y='f1_macro', hue='model', data=df, marker='o', linewidth=2.5)
axes[1, 0].set_title('Test Size vs F1 Macro')
axes[1, 0].set_xlabel('Test Size')
axes[1, 0].set_ylabel('F1 Macro')

# Remove the individual legends
axes[0, 0].legend_.remove()
axes[0, 1].legend_.remove()
axes[1, 0].legend_.remove()

# Create an empty plot to fill the 2x2 grid and place the legend there
axes[1, 1].axis('off')  # Hide the last empty subplot
handles, labels = axes[0, 0].get_legend_handles_labels()
axes[1, 1].legend(handles, labels, loc='center', title='Model')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Add extra space for the main title
plt.suptitle('', y=0.98, fontsize=16)  # Add a main title
plt.show()