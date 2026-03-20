import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


embeddings = "/home/yl9210a-hpc/chem_project/combine_structual_texual/combined_bace_embeddings.json"
df = pd.read_json(embeddings)

# select embedding column
X = np.vstack(df["combined_embedding"].values)    # shape: (N, D)
y = df["pIC50"].values

# ---------------------------------------------------
# Model
# ---------------------------------------------------
models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=300,
        n_jobs=-1,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="rmse"   # prevent warning
    )
}

# ---------------------------------------------------
# 3. Cross-validation
# ---------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {name: {"r2": [], "rmse": [], "mae": []} for name in models}

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        r2 = r2_score(y_test, pred)
        rmse = root_mean_squared_error(y_test, pred) 
        mae = mean_absolute_error(y_test, pred)

        results[name]["r2"].append(r2) 
        results[name]["rmse"].append(rmse) 
        results[name]["mae"].append(mae)
# ---------------------------------------------------
# 4. Output
# ---------------------------------------------------
for name in models:
    print(f"\n=== {name} ===")
    print(f"R2:   {np.mean(results[name]['r2']):.4f} ± {np.std(results[name]['r2']):.4f}")
    print(f"RMSE: {np.mean(results[name]['rmse']):.4f} ± {np.std(results[name]['rmse']):.4f}")
    print(f"MAE:  {np.mean(results[name]['mae']):.4f} ± {np.std(results[name]['mae']):.4f}")

# ---------------------------------------------------
# 4. output + write in file
# ---------------------------------------------------
output_path = "regression_results.txt"

with open(output_path, "w") as f:
    for name in models:
        f.write(f"\n=== {name} ===\n")
        f.write(f"R2:   {np.mean(results[name]['r2']):.4f} ± {np.std(results[name]['r2']):.4f}\n")
        f.write(f"RMSE: {np.mean(results[name]['rmse']):.4f} ± {np.std(results[name]['rmse']):.4f}\n")
        f.write(f"MAE:  {np.mean(results[name]['mae']):.4f} ± {np.std(results[name]['mae']):.4f}\n")

#  stdout → regression_results.log
for name in models:
    print(f"\n=== {name} ===")
    print(f"R2:   {np.mean(results[name]['r2']):.4f} ± {np.std(results[name]['r2']):.4f}")
    print(f"RMSE: {np.mean(results[name]['rmse']):.4f} ± {np.std(results[name]['rmse']):.4f}")
    print(f"MAE:  {np.mean(results[name]['mae']):.4f} ± {np.std(results[name]['mae']):.4f}")
