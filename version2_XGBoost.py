"""XGBoost + SHAP — 50m"""
import pandas as pd, numpy as np, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import shap
import statsmodels.api as sm
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

df = pd.read_csv("task1_features_excl50_seed42.csv")
print(f"✓ Loaded: {len(df)} rows")

FEATURES = ['office_count','residential_count','mall_count','metro_station_count',
            'dist_to_nearest_luckin','dist_to_nearest_starbucks','dist_to_nearest_metro_exit']
X = df[FEATURES].copy()
dist_cols = [c for c in FEATURES if c.startswith('dist_')]
X[dist_cols] = StandardScaler().fit_transform(X[dist_cols])
y = df['case'].astype(int)

# ---Training (Super participation in Chapter 4, no parameter adjustment) ---
model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5,
                          eval_metric='auc', random_state=42, use_label_encoder=False)
model.fit(X, y)
y_proba = model.predict_proba(X)[:, 1]
train_auc = roc_auc_score(y, y_proba)
print(f"Training AUC: {train_auc:.4f} (overfitting reference value, not included in paper)")

# ---50% off random CV (preserve the continuity with the old results; use the blocked CV of task 2 for formal comparison) ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
print(f"5-fold CV AUC: {cv_aucs.round(4)}  Mean {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

# ---Compare logistic (GWLR numbers see GWLR script output, no longer hard-coded) ---
X_const = sm.add_constant(X)
logit = sm.Logit(y, X_const).fit(disp=False)
logit_auc = roc_auc_score(y, logit.predict(X_const))
print(f"\nLogistic in-sample AUC = {logit_auc:.4f}")
print(f"XGBoost 5-fold CV AUC  = {cv_aucs.mean():.4f}")

# ---SHAP ---
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

importance = pd.DataFrame({'feature': FEATURES,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)}).sort_values('mean_abs_shap')
print("\n=== SHAP Feature Importance ===")
for _, r in importance[::-1].iterrows():
    print(f"  {r['feature']:<32}{r['mean_abs_shap']:.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(importance['feature'], importance['mean_abs_shap'], color='steelblue')
ax.set_xlabel("Mean(|SHAP value|)"); ax.set_title("XGBoost feature importance")
ax.bar_label(bars, fmt='%.4f', padding=5, fontsize=10)
ax.set_xlim(right=importance['mean_abs_shap'].max()*1.15)
plt.tight_layout(); plt.savefig("shap_importance_excl50.png", dpi=150, bbox_inches='tight'); plt.close()

fig = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, feature_names=FEATURES, show=False)
plt.tight_layout(); plt.savefig("shap_summary_excl50.png", dpi=150, bbox_inches='tight'); plt.close()

# fig, axes = plt.subplots(2, 4, figsize=(20, 9)); axes = axes.flatten()
# for i, feat in enumerate(FEATURES):
#     ax = axes[i]
#     ax.scatter(X[feat], shap_values[:, i], alpha=0.3, s=10, c='steelblue')
#     si = np.argsort(X[feat].values)
#     smooth = pd.Series(shap_values[si, i]).rolling(
#         max(20, len(si)//30), center=True, min_periods=1).mean()
#     ax.plot(X[feat].values[si], smooth, color='red', lw=2)
#     ax.axhline(0, color='gray', lw=0.5)
#     ax.set_title(feat, fontsize=10); ax.grid(alpha=0.3)
# axes[-1].axis('off')
# # plt.suptitle("SHAP dependence plots (50m dataset)", y=1.00)
# plt.tight_layout(); plt.savefig("shap_dependence_excl50.png", dpi=150, bbox_inches='tight'); plt.close()
# ---SHAP dependence (only draw two variables) ---
selected_features = ['office_count', 'dist_to_nearest_starbucks']

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, feat in zip(axes, selected_features):
    i = FEATURES.index(feat)

    ax.scatter(
        X[feat],
        shap_values[:, i],
        alpha=0.3,
        s=10,
        c='steelblue'
    )

    si = np.argsort(X[feat].values)

    smooth = (
        pd.Series(shap_values[si, i])
        .rolling(max(20, len(si)//30), center=True, min_periods=1)
        .mean()
    )

    ax.plot(
        X[feat].values[si],
        smooth,
        color='red',
        lw=2
    )

    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title(feat, fontsize=11)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("selected_shap_dependence_excl50.png", dpi=150, bbox_inches='tight')
plt.close()

# ---Sorting comparison: logistic vs SHAP ---
li = pd.DataFrame({'feature': FEATURES,
    'abs_coef': np.abs(logit.params[1:].values),
    'p': logit.pvalues[1:].values}).sort_values('abs_coef', ascending=False).reset_index(drop=True)
si_ = importance[::-1].reset_index(drop=True)
print("\n=== Feature Importance Sorting Comparison ===")
for i in range(len(FEATURES)):
    m = "✓" if li.iloc[i]['feature'] == si_.iloc[i]['feature'] else " "
    print(f"  {i+1}  {li.iloc[i]['feature']:<28}{li.iloc[i]['abs_coef']:.3f}"
          f"   {m} {si_.iloc[i]['feature']:<28}{si_.iloc[i]['mean_abs_shap']:.4f}")

print("\n✓ Completed, the picture has been saved (*_excl50.png）")