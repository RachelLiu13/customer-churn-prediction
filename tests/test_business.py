import numpy as np
import pytest

from churn.business import Campaign, best_target, profit_curve


def test_profit_curve_counts_only_targeted_customers():
    # two customers: the high-score one churns, the low-score one does not
    curve = profit_curve(scores=[0.9, 0.1], churned=[1, 0], monthly_charges=[100, 100],
                         campaign=Campaign(offer_cost=50, save_rate=0.5, months_retained=12), steps=3)
    # target nobody, the top 50%, everybody
    assert curve["customers"].tolist() == [0, 1, 2]
    assert curve["profit"].tolist() == pytest.approx([0, 0.5 * 1200 - 50, 0.5 * 1200 - 100])


def test_perfect_ranking_beats_reversed_ranking():
    rng = np.random.default_rng(1)
    churned = rng.integers(0, 2, 500)
    charges = rng.uniform(20, 120, 500)
    good = best_target(profit_curve(churned + rng.random(500) * 0.1, churned, charges))
    bad = best_target(profit_curve(-churned + rng.random(500) * 0.1, churned, charges))
    assert good["profit"] > bad["profit"]


def test_no_campaign_when_offer_costs_more_than_any_save():
    curve = profit_curve([0.9, 0.8], [1, 1], [10, 10], Campaign(offer_cost=1000, save_rate=0.3, months_retained=12))
    assert best_target(curve)["customers"] == 0
