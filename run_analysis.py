"""Reproduce every number and figure in the README.

    python run_analysis.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from churn.business import Campaign, best_target, profit_curve
from churn.data import load
from churn.models import evaluate, make_models, split, top_drivers

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df"})

df = load()
X_train, X_test, y_train, y_test = split(df)
models = make_models(df)
metrics, test_scores = evaluate(models, X_train, y_train, X_test, y_test)
print(f"Customers: {len(df):,} | churn rate: {df.churn.mean():.1%}\n")
print(metrics.round(3).to_markdown())

print("\nTop churn drivers (logistic regression coefficients):")
print(top_drivers(models["Logistic regression"]).round(2).rename("coefficient").to_markdown())

fig, ax = plt.subplots(figsize=(5.5, 4.5))
for (name, p), c in zip(test_scores.items(), [GRAY, BLUE]):
    fpr, tpr, _ = roc_curve(y_test, p)
    ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {roc_auc_score(y_test, p):.3f})")
ax.legend(frameon=False, loc="lower right"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.plot([0, 1], [0, 1], ls="--", color="#c3c2b7", lw=1)
ax.set_title("ROC curve, hold-out set", loc="left")
fig.tight_layout(); fig.savefig("figures/roc.png", dpi=150)

campaign = Campaign()
charges = X_test["MonthlyCharges"].values
rng = np.random.default_rng(0)
curves = {name: profit_curve(p, y_test, charges, campaign) for name, p in test_scores.items()}
curves["Random targeting"] = profit_curve(rng.random(len(y_test)), y_test, charges, campaign)
fig, ax = plt.subplots(figsize=(7, 4))
for (name, c), col in zip(curves.items(), [GRAY, BLUE, ORANGE]):
    ax.plot(c.share_targeted * 100, c.profit / 1000, label=name, color=col, lw=2)
best = best_target(curves["XGBoost"])
ax.scatter([best.share_targeted * 100], [best.profit / 1000], color=BLUE, zorder=3, s=40)
ax.annotate(f"best: target top {best.share_targeted:.0%}\n${best.profit/1000:,.0f}K", (best.share_targeted * 100, best.profit / 1000),
            textcoords="offset points", xytext=(12, -38))
ax.axhline(0, color="#52514e", lw=0.8)
ax.set_xlabel("% of customers targeted (highest risk first)"); ax.set_ylabel("Expected campaign profit ($K)")
ax.set_title(f"Retention campaign profit on the hold-out set ({len(y_test):,} customers)", loc="left"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/profit_curve.png", dpi=150)

print(f"\nCampaign assumptions: offer ${campaign.offer_cost:.0f}, save rate {campaign.save_rate:.0%}, "
      f"{campaign.months_retained} months of revenue kept")
for name, c in curves.items():
    b = best_target(c)
    print(f"{name:20s} best: target top {b.share_targeted:.0%} ({int(b.customers):,} customers) -> profit ${b.profit:,.0f}")
everyone = curves["XGBoost"].iloc[-1]
print(f"Targeting everyone: ${everyone.profit:,.0f}")
