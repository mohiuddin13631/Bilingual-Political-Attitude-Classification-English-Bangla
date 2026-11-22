"""Bias & fairness analysis for political sentiment models."""

from .bias_analysis import (
    group_performance,
    sentiment_distribution_by_group,
    subgroup_fairness_gaps,
)

__all__ = [
    "group_performance",
    "sentiment_distribution_by_group",
    "subgroup_fairness_gaps",
]
