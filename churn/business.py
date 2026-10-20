"""Turn churn scores into a retention-campaign decision.

A model's AUC does not say how many customers to contact. Here we rank customers by predicted risk,
target the top k%, and compute the campaign's expected profit under explicit assumptions:

    profit = sum over targeted customers of  churned * save_rate * customer_value  -  offer_cost

`customer_value` is the revenue kept by saving a churner: monthly charge x months_retained.
All assumptions are parameters, so the analysis can be redone with a company's own numbers.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Campaign:
    offer_cost: float = 50.0       # $ cost of the retention offer per targeted customer
    save_rate: float = 0.30        # share of would-be churners the offer keeps
    months_retained: int = 12      # months of revenue kept per saved customer


def profit_curve(scores, churned, monthly_charges, campaign: Campaign = Campaign(), steps: int = 101) -> pd.DataFrame:
    """Expected profit when targeting the top k% of customers by score, for k = 0..100%."""
    order = np.argsort(-np.asarray(scores))
    churned = np.asarray(churned)[order]
    value = np.asarray(monthly_charges)[order] * campaign.months_retained
    gain = churned * campaign.save_rate * value - campaign.offer_cost
    cum = np.concatenate([[0.0], np.cumsum(gain)])
    n = len(order)
    share = np.linspace(0, 1, steps)
    k = np.round(share * n).astype(int)
    return pd.DataFrame({"share_targeted": share, "customers": k, "profit": cum[k]})


def best_target(curve: pd.DataFrame) -> pd.Series:
    return curve.loc[curve["profit"].idxmax()]
