"""
Main model: Logistic Regression
=========================================
Steps:
  1. Load features
  2. VIF collinearity check
  3. Feature standardization (only distance features)
  4. Fit logistic regression
  5. Output coefficient table + odds ratio + AUC
  6. Save the residuals for the next step (Moran's I)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# 1. Load data
df = pd.read_csv("features.csv")
print(f"✓ Load features.csv: {len(df)} rows "
      f"({df['case'].sum()} cases + {(df['case']==0).sum()} controls)")

FEATURES = [
    'office_count',
    'residential_count',
    'mall_count',
    'metro_station_count',
    'dist_to_nearest_luckin',
    'dist_to_nearest_starbucks',
    'dist_to_nearest_metro_exit',
]
X_raw = df[FEATURES].copy()
y = df['case'].astype(int)


# 2. VIF collinearity check
# VIF (Variance Inflation Factor) = the extent to which a variable is explained by other variables
# VIF < 5: OK; 5–10: Caution; >10: Severe collinearity
# You need to add the intercept column first

print("\n=== VIF collinearity check ===")
X_with_const = sm.add_constant(X_raw)
vif_data = pd.DataFrame({
    'feature': X_with_const.columns,
    'VIF': [variance_inflation_factor(X_with_const.values, i)
            for i in range(X_with_const.shape[1])]
})
print(vif_data.to_string(index=False))
print("\nSuggestion: VIF < 5 OK; 5-10 Note; >10 Severe collinearity needs to be dealt with")


# 3. Standardized distance features
# The distance unit is "meter", POI is "number", and the magnitude difference is 2-3 orders of magnitude
# Unstandardized coefficients will lose comparability (the coefficient of dist will be small but actually very important)
#
# Method: Only standardize dist class features (z-score), and keep the original value of count class
# The benefits of non-standardization of the count class: the explanation is intuitive ("1 more office building") and conforms to the convention of the dissertation form

X = X_raw.copy()
dist_cols = [c for c in FEATURES if c.startswith('dist_')]
scaler = StandardScaler()
X[dist_cols] = scaler.fit_transform(X[dist_cols])

print(f"\n✓ Distance features standardized ({len(dist_cols)} columns)")
print(f"  Count features kept original values (for interpretability)")


# 4. Fit Logistic Regression (use statsmodels to obtain academic table output)
# The output of statsmodels is more complete than sklearn: z-values, p-values, confidence intervals, pseudo-R²
# It is a standard tool for economics/sociology/planning papers.

X_const = sm.add_constant(X)  # Add intercept
model = sm.Logit(y, X_const).fit(disp=False)

print("\n" + "=" * 80)
print("Logistic Regression Main Model Results")
print("=" * 80)
print(model.summary())


# 5. Odds Ratio + 95% CI (more intuitive interpretation)
print("\n=== Odds Ratio Table (for dissertation) ===")
params = model.params
conf_int = model.conf_int()
conf_int.columns = ['CI_lower', 'CI_upper']

odds_table = pd.DataFrame({
    'coef': params,
    'p_value': model.pvalues,
    'OR': np.exp(params),
    'OR_CI_low': np.exp(conf_int['CI_lower']),
    'OR_CI_high': np.exp(conf_int['CI_upper']),
})
print(odds_table.round(4).to_string())

# Interpretation instructions
print("OR Interpretation:")
print("OR > 1: Higher values of this feature increase the odds of being a Luckin store")
print("OR < 1: Higher values of this feature decrease the odds of being a Luckin store")
print("Count features (OR directly corresponds to \"1 more unit\"); Distance features (OR corresponds to \"1 standard deviation further\")")


# 6. Model performance: AUC + ROC + confusion matrix
y_pred_proba = model.predict(X_const)
auc = roc_auc_score(y, y_pred_proba)

# Classification under threshold 0.5
y_pred = (y_pred_proba >= 0.5).astype(int)
cm = confusion_matrix(y, y_pred)

print(f"\n=== Model Performance ===")
print(f"AUC: {auc:.4f}")
print(f"Pseudo R² (McFadden): {model.prsquared:.4f}")
print(f"\nConfusion Matrix (Threshold=0.5):")
print(f"              Predicted=0   Predicted=1")
print(f"  Actual=0    {cm[0,0]:5d}    {cm[0,1]:5d}")
print(f"  Actual=1    {cm[1,0]:5d}    {cm[1,1]:5d}")

accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])  # Recall (ability to identify cases)
specificity = cm[0,0] / (cm[0,0] + cm[0,1])  # Ability to identify controls
print(f"\nAccuracy: {accuracy:.3f}")
print(f"Sensitivity (Case Recall): {sensitivity:.3f}")
print(f"Specificity (Control Recall): {specificity:.3f}")


# 7. Visualization: ROC Curve
fpr, tpr, _ = roc_curve(y, y_pred_proba)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'Logistic (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve – Main Logistic Regression Model')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ ROC Curve saved to roc_curve.png")


# 8. Save the residuals for the next step (Moran's I)
# Moran's I checks "whether the model residuals still have spatial structure"
# If the residuals are clustered in space, it means that logistic does not capture the geographical factors.
# → Trigger GWLR phase
#
# Pearson residuals are the most commonly used form

residuals = pd.DataFrame({
    'lon_wgs84': df['lon_wgs84'],
    'lat_wgs84': df['lat_wgs84'],
    'case': y,
    'pred_proba': y_pred_proba,
    'pearson_residual': (y - y_pred_proba) / np.sqrt(y_pred_proba * (1 - y_pred_proba)),
})
residuals.to_csv("residuals.csv", index=False, encoding='utf-8-sig')
print(f"✓ Residuals saved to residuals.csv ({len(residuals)} rows)")

# Save the coefficient table for dissertation use
odds_table.to_csv("logit_coefficients.csv", encoding='utf-8-sig')
print(f"✓ Coefficient table saved to logit_coefficients.csv")