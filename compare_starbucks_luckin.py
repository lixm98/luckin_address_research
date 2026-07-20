"""
Luckin vs Starbucks coefficient comparison + Wald test
=========================================
Function:
  1. Load two logistic coefficient tables
  2. Wald test to test the difference in coefficients
  3. Classification of 4 types of conclusions
  4. Output dissertation table + bar graph
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# 1. Load the coefficient tables on both sides
luckin = pd.read_csv("logit_coefficients.csv", index_col=0)
starbucks = pd.read_csv("starbucks_logit_coefficients.csv", index_col=0)

# remove intercept
luckin = luckin[luckin.index != 'const']
starbucks = starbucks[starbucks.index != 'const']


# 2. Compatibility: Luckin’s old files may not have std_err, so reverse it from CI
# 95% CI width (log-odds scale) = 2 × 1.96 × SE
#   → SE = (log(OR_CI_high) -log(OR_CI_low)) /3.92
# This formula is exactly equivalent to the std_err output by sm.Logit

if 'std_err' not in luckin.columns:
    luckin['std_err'] = (
        np.log(luckin['OR_CI_high']) - np.log(luckin['OR_CI_low'])
    ) / (2 * 1.96)
    print("✓ Luckin std_err 从 OR CI Push back")

if 'std_err' not in starbucks.columns:
    starbucks['std_err'] = (
        np.log(starbucks['OR_CI_high']) - np.log(starbucks['OR_CI_low'])
    ) / (2 * 1.96)

# Make sure the order of features on both sides is consistent
common_features = [f for f in luckin.index if f in starbucks.index]
luckin = luckin.loc[common_features]
starbucks = starbucks.loc[common_features]
print(f"✓ contrast {len(common_features)} 个特征")


# 3. Wald test: Difference in coefficients of two independent samples
# Z = (β_l -β_s) /sqrt(SE_l² + SE_s²) ~ N(0,1) under H0
# |Z| > 1.96 → reject "two coefficients are equal"

comparison = pd.DataFrame(index=common_features)
comparison['luckin_coef']     = luckin['coef']
comparison['luckin_se']       = luckin['std_err']
comparison['luckin_p']        = luckin['p_value']
comparison['luckin_OR']       = luckin['OR']
comparison['sb_coef']         = starbucks['coef']
comparison['sb_se']           = starbucks['std_err']
comparison['sb_p']            = starbucks['p_value']
comparison['sb_OR']           = starbucks['OR']

se_diff = np.sqrt(luckin['std_err']**2 + starbucks['std_err']**2)
comparison['coef_diff']       = luckin['coef'] - starbucks['coef']
comparison['wald_z']          = comparison['coef_diff'] / se_diff
comparison['wald_p']          = 2 * (1 - stats.norm.cdf(np.abs(comparison['wald_z'])))


# 4. Classification
def classify(row):
    l_sig    = row['luckin_p'] < 0.05
    s_sig    = row['sb_p'] < 0.05
    same_dir = np.sign(row['luckin_coef']) == np.sign(row['sb_coef'])
    diff_sig = row['wald_p'] < 0.05

    if l_sig and s_sig and same_dir and diff_sig:
        return "A. Significant difference in intensity in the same direction"
    elif l_sig and s_sig and same_dir and not diff_sig:
        return "A'. Similar strength in the same direction"
    elif l_sig and s_sig and not same_dir:
        return "B. Direction opposite"
    elif l_sig and not s_sig:
        return "C1. Only Luckin significant"
    elif s_sig and not l_sig:
        return "C2. Only Starbucks significant"
    else:
        return "D. Neither is significant"

comparison['conclusion'] = comparison.apply(classify, axis=1)


# 5. Print comparison table
print("\n" + "=" * 130)
print("Luckin vs Starbucks Logistic Coefficients Comparison")
print("=" * 130)
print(f"{'Feature':<32} {'Luckin β (p)':<22} {'Starbucks β (p)':<22} "
      f"{'Wald Z (p)':<18} {'Category':<25}")
print("-" * 130)
for feat in comparison.index:
    row = comparison.loc[feat]
    l_str = f"{row['luckin_coef']:+.3f} ({row['luckin_p']:.3f})"
    s_str = f"{row['sb_coef']:+.3f} ({row['sb_p']:.3f})"
    w_str = f"{row['wald_z']:+.2f} ({row['wald_p']:.3f})"
    print(f"{feat:<32} {l_str:<22} {s_str:<22} {w_str:<18} {row['conclusion']:<25}")


# 6. Summary by category
print("\n" + "=" * 80)
print("Summary by Category")
print("=" * 80)
for category in sorted(comparison['conclusion'].unique()):
    feats = comparison[comparison['conclusion'] == category].index.tolist()
    if not feats:
        continue
    print(f"\n  [{category}]")
    for f in feats:
        l = comparison.loc[f, 'luckin_OR']
        s = comparison.loc[f, 'sb_OR']
        ldir = "↑" if comparison.loc[f, 'luckin_coef'] > 0 else "↓"
        sdir = "↑" if comparison.loc[f, 'sb_coef'] > 0 else "↓"
        print(f"    - {f:<32} Luckin OR={l:.3f}{ldir}  Starbucks OR={s:.3f}{sdir}")


# 7. Bar graph: coefficient + 95% CI (for dissertation)
features = comparison.index.tolist()
n = len(features)
y_pos = np.arange(n)
width = 0.35

fig, ax = plt.subplots(figsize=(11, 7))

luckin_low  = luckin['coef'] - 1.96 * luckin['std_err']
luckin_high = luckin['coef'] + 1.96 * luckin['std_err']
ax.errorbar(luckin['coef'], y_pos - width/2,
            xerr=[luckin['coef'] - luckin_low, luckin_high - luckin['coef']],
            fmt='o', color='crimson', label='Luckin', markersize=8, capsize=4)

sb_low  = starbucks['coef'] - 1.96 * starbucks['std_err']
sb_high = starbucks['coef'] + 1.96 * starbucks['std_err']
ax.errorbar(starbucks['coef'], y_pos + width/2,
            xerr=[starbucks['coef'] - sb_low, sb_high - starbucks['coef']],
            fmt='s', color='forestgreen', label='Starbucks', markersize=8, capsize=4)

ax.axvline(0, color='gray', lw=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.set_xlabel("Coefficient (log-odds)")
ax.set_title("Luckin vs Starbucks: Logistic regression coefficients with 95% CI")
ax.legend(loc='best')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("luckin_vs_starbucks_coefs.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Comparison chart saved to luckin_vs_starbucks_coefs.png")


# 8. Save the comparison table
comparison.to_csv("luckin_vs_starbucks_comparison.csv", encoding='utf-8-sig')
print(f"✓ Comparison table saved to luckin_vs_starbucks_comparison.csv")


# 9. Comparison of overall model performance
try:
    sb_summary = pd.read_csv("starbucks_model_summary.csv")
    print("\n=== Starbucks Model Overall Performance ===")
    for _, row in sb_summary.iterrows():
        print(f"  {row['metric']:<25} = {row['value']}")
except FileNotFoundError:
    pass