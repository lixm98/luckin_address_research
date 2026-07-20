"""
XGBoost + SHAP robustness check
=========================================
Function:
  1. Use XGBoost (nonlinear, automatically capture interaction) as a comparison of logistic
  2. Use SHAP to interpret XGBoost's "black box" output
  3. Compare SHAP importance vs logistic coefficient ranking to verify robustness
  4. Discover possible nonlinear effects (Sigmoid, threshold, saturation) and feature interactions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# 1. Load data (preprocessing exactly the same as logistic /GWLR)
df = pd.read_csv("features.csv")
print(f"✓ load: {len(df)} rows")

FEATURES = [
    'office_count', 'residential_count', 'mall_count', 'metro_station_count',
    'dist_to_nearest_luckin', 'dist_to_nearest_starbucks', 'dist_to_nearest_metro_exit',
]
# Note: XGBoost is not sensitive to feature dimensions (unlike logistic which requires standardized distance)
# But in order to be fully comparable with logistic /GWLR, the same standardization is still used
X = df[FEATURES].copy()
dist_cols = [c for c in FEATURES if c.startswith('dist_')]
X[dist_cols] = StandardScaler().fit_transform(X[dist_cols])
y = df['case'].astype(int)


# 2. Training XGBoost
# Key parameter description:
#   n_estimators=300: run 300 trees
#   max_depth=4: Each tree can have up to 4 layers → limit complexity and prevent over-fitting
#                        4 layers means capturing interactions of up to 4 features
#   learning_rate=0.05: The contribution of each tree is small, more trees are needed → more robust
#   subsample=0.8: Each tree uses only 80% samples → bagging to resist overfitting
#   scale_pos_weight=5: case:control = 1:5 unbalanced → give case 5 times the weight
#   eval_metric='auc': Evaluate with AUC (aligned with logistic)

print("\n=== Training XGBoost ===")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=5,
    eval_metric='auc',
    random_state=42,
    use_label_encoder=False,
)
model.fit(X, y)

# Training set AUC
y_proba = model.predict_proba(X)[:, 1]
train_auc = roc_auc_score(y, y_proba)
print(f"  Training AUC: {train_auc:.4f}")


# 3. Cross-validation AUC (more reliable AUC estimate)
# Training AUC is easily overestimated, 5-fold cross-validation gives the "true" AUC
# Each time, use 80% of the data for training, 20% for testing, and run 5 times to take the average.

print("\n=== 5-Fold Cross-Validation AUC ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
print(f"  5-Fold AUC: {cv_aucs.round(4)}")
print(f"  Mean: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")


# 4. Horizontal comparison of three models (logistic/GWLR/XGBoost)
import statsmodels.api as sm
X_const = sm.add_constant(X)
logit = sm.Logit(y, X_const).fit(disp=False)
logit_auc = roc_auc_score(y, logit.predict(X_const))

print("\n===Three-model AUC comparison===")
print(f"  Logistic (global linearity)        AUC = {logit_auc:.4f}")
print(f"  GWLR (spatial local linearity, K=400) AUC = 0.8745  (Previous step result)")
print(f"  XGBoost Training               AUC = {train_auc:.4f}")
print(f"  XGBoost 5-fold CV          AUC = {cv_aucs.mean():.4f}")

print("""
Interpretation:
  -XGBoost training AUC is high = the model can fit nonlinearities
  -XGBoost CV AUC is a fair comparison
  -Gap in CV AUC vs logistic = extra portion explained by non-linear effects
  -if difference < 0.03 → linear is enough → logistic conclusion robust
  -If difference > 0.05 → significant non-linearity → SHAP to find details
""")


# 5. SHAP Interpretation
# TreeExplainer is SHAP's dedicated fast algorithm for tree models (exact solution)
# Unlike neural networks SHAP is an approximation

print("=== Computing SHAP values ===")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
print(f"  shap_values shape: {shap_values.shape}  (n_samples × n_features)")


# 6. SHAP feature importance bar chart
# "Mean absolute SHAP value" of each feature = its average contribution to prediction
# Sorting is the "feature importance" from the perspective of XGBoost

importance = pd.DataFrame({
    'feature': FEATURES,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=True)

print("\n=== SHAP Feature Importance ===")
for _, row in importance[::-1].iterrows():
    bar = '█' * int(row['mean_abs_shap'] * 100)
    print(f"  {row['feature']:<32} {row['mean_abs_shap']:.4f}  {bar}")

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(importance['feature'], importance['mean_abs_shap'], color='steelblue')
ax.set_xlabel("Mean(|SHAP value|)")
ax.set_title("XGBoost feature importance (SHAP)")
ax.bar_label(
    bars,
    fmt='%.4f',
    padding=5,
    fontsize=10
)
ax.set_xlim(right=importance['mean_abs_shap'].max() * 1.15)
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ shap_importance.png saved")


# 7. SHAP Summary Plot (scatter plot + eigenvalue color coding)
# This is the most classic picture of SHAP
# Each point = a sample, X = SHAP value, color = feature value high and low
# It can be seen that high eigenvalues are concentrated in SHAP > 0 or < 0

fig = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, feature_names=FEATURES, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ shap_summary.png saved")


# 8. SHAP Dependence Plot (1 picture for each feature, discover non-linearity)
# X = Eigenvalue, Y = SHAP contribution
# Shape: straight line → linear; S-shaped → threshold effect; inverted U → decrease after saturation

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()

for i, feat in enumerate(FEATURES):
    ax = axes[i]
    feat_idx = FEATURES.index(feat)
    
    # Scatter: each sample
    ax.scatter(X[feat], shap_values[:, feat_idx], alpha=0.3, s=10, c='steelblue')
    
    # Smooth trend line (loess is too slow, use quantiles for rough smoothing)
    sorted_idx = np.argsort(X[feat].values)
    x_sorted = X[feat].values[sorted_idx]
    s_sorted = shap_values[sorted_idx, feat_idx]
    # sliding window average
    window = max(20, len(x_sorted) // 30)
    smooth = pd.Series(s_sorted).rolling(window, center=True, min_periods=1).mean()
    ax.plot(x_sorted, smooth, color='red', lw=2, label='trend')
    
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel("SHAP value", fontsize=9)
    ax.set_title(f"{feat}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

axes[-1].axis('off')
plt.suptitle("SHAP dependence plots — detect non-linear effects\n"
             "(red line = local average; deviations from linear = non-linearity)",
             fontsize=11, y=1.00)
plt.tight_layout()
plt.savefig("shap_dependence.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ shap_dependence.png saved")


# 9. ROC three-model comparison chart
fpr_l, tpr_l, _ = roc_curve(y, logit.predict(X_const))
fpr_x, tpr_x, _ = roc_curve(y, y_proba)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_l, tpr_l, lw=2, label=f'Logistic (AUC={logit_auc:.3f})', color='steelblue')
ax.plot(fpr_x, tpr_x, lw=2, label=f'XGBoost train (AUC={train_auc:.3f})', color='crimson')
ax.plot([0,1], [0,1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC: Logistic vs XGBoost")
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_logistic_vs_xgboost.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ roc_logistic_vs_xgboost.png saved")


# 10. Comparison: SHAP sorting vs logistic coefficient sorting
# Key robustness check: Whether the "important features" seen by the two methods are consistent
# Consistent → main conclusion robust; inconsistent → non-linearities need to be discussed

logit_importance = pd.DataFrame({
    'feature': FEATURES,
    'logit_abs_coef': np.abs(logit.params[1:].values),  # skip intercept
    'logit_pvalue': logit.pvalues[1:].values,
}).sort_values('logit_abs_coef', ascending=False).reset_index(drop=True)

shap_importance = importance[::-1].reset_index(drop=True)

print("\n=== Logistic vs SHAP Feature Importance Ranking ===")
print(f"{'Rank':<5} {'Logistic (|coef|)':<35} {'XGBoost-SHAP':<35}")
print("-" * 80)
for i in range(len(FEATURES)):
    l_feat = logit_importance.iloc[i]['feature']
    l_val = logit_importance.iloc[i]['logit_abs_coef']
    l_p = logit_importance.iloc[i]['logit_pvalue']
    sig = "***" if l_p < 0.001 else "**" if l_p < 0.01 else "*" if l_p < 0.05 else ""
    s_feat = shap_importance.iloc[i]['feature']
    s_val = shap_importance.iloc[i]['mean_abs_shap']
    
    match = "✓" if l_feat == s_feat else " "
    print(f"  {i+1:<3} {l_feat:<25} {l_val:.3f} {sig:<3}   "
          f"{match} {s_feat:<25} {s_val:.4f}")

print("\n✓ indicates consistent ranking between the two methods")
print("More inconsistencies → stronger nonlinear effects")


print("\n=== Completion ===")
print("Generated plots:")
print("  shap_importance.png    Feature importance bar chart")
print("  shap_summary.png       SHAP scatter plot (most classic)")
print("  shap_dependence.png    Nonlinear effect visualization")
print("  roc_logistic_vs_xgboost.png  ROC comparison")