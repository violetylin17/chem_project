import numpy as np
import matplotlib.pyplot as plt

models = ["Ridge", "RandomForest", "XGBoost"]
x = np.arange(len(models))

embeddings = ["Textual", "Structural", "Combined"]
colors = ["tab:blue", "tab:orange", "tab:green"]

# === Mean values ===
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

# === Std values ===
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

metrics = ["R2", "RMSE", "MAE"]
metric_index = {"R2": 0, "RMSE": 1, "MAE": 2}

# === 一個 figure，三個 subplot ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

for ax, metric in zip(axes, metrics):
    idx = metric_index[metric]

    for emb, color in zip(embeddings, colors):
        means = [mean_values[emb][m][idx] for m in models]
        stds  = [std_values[emb][m][idx] for m in models]

        ax.errorbar(
            x,
            means,
            yerr=stds,
            label=emb,
            marker="o",
            capsize=5,
            linewidth=2,
            color=color
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} trend")
    ax.grid(alpha=0.3)

# 只在第一個 subplot 放 legend，或你也可以放在 fig 外
axes[0].legend(
    title="Embedding",
    loc="best",
    handlelength=0,    # 不顯示線段，只顯示 marker
    markerscale=1.5,
    ncol=3            # ❗ 一行兩個 legend 項目
)




fig.tight_layout()
plt.savefig("plots.png", dpi=300, bbox_inches="tight")
plt.show()
