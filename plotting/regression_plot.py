import numpy as np
import matplotlib.pyplot as plt

# === Step 1: 把你的數據放進 dictionary ===
metrics = ["R2", "RMSE", "MAE"]
models = ["Ridge", "RandomForest", "XGBoost"]
embeddings = ["Textual", "Structural", "Combined"]

# mean values
mean_values = {
    "Textual": {
        "Ridge":       [0.4441, 0.9978, 0.8032],
        "RandomForest":[0.4123, 1.0260, 0.8254],
        "XGBoost":     [0.4584, 0.9848, 0.7800],
    },
    "Structural": {
        "Ridge":       [0.4323, 1.0086, 0.7577],
        "RandomForest":[0.5707, 0.8765, 0.6851],
        "XGBoost":     [0.6154, 0.8301, 0.6374],
    },
    "Combined": {
        "Ridge":       [0.5565, 0.8902, 0.6775],
        "RandomForest":[0.5774, 0.8703, 0.6922],
        "XGBoost":     [0.6501, 0.7920, 0.6271],
    }
}


# std values
std_values = {
    "Textual": {
        "Ridge":       [0.0545, 0.0519, 0.0505],
        "RandomForest":[0.0601, 0.0581, 0.0524],
        "XGBoost":     [0.0542, 0.0498, 0.0437],
    },
    "Structural": {
        "Ridge":       [0.0433, 0.0297, 0.0187],
        "RandomForest":[0.0554, 0.0593, 0.0335],
        "XGBoost":     [0.0403, 0.0456, 0.0287],
    },
    "Combined": {
        "Ridge":       [0.0463, 0.0480, 0.0412],
        "RandomForest":[0.0282, 0.0466, 0.0348],
        "XGBoost":     [0.0310, 0.0539, 0.0374],
    }
}

# === Step 2: 畫圖 ===
x = np.arange(len(models))  # 3 models
width = 0.25                # 每組 bar 的寬度

for metric_idx, metric in enumerate(metrics):
    plt.figure(figsize=(10, 6))

    for i, emb in enumerate(embeddings):
        means = [mean_values[emb][m][metric_idx] for m in models]
        stds  = [std_values[emb][m][metric_idx] for m in models]

        plt.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            capsize=5,
            label=emb
        )

    plt.xticks(x + width, models)
    plt.ylabel(metric)
    plt.title(f"{metric} Comparison Across Models with Error Bars")
    plt.legend()
    plt.tight_layout()
    plt.savefig("regression.png", dpi=300, bbox_inches="tight")
    plt.show()
