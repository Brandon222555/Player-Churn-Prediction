"""
src/ab_test.py
===============
Statistical framework for testing player retention interventions.

Usage:
    python src/ab_test.py
"""

import numpy as np
from scipy import stats
from typing import Optional


def ab_test_retention(
    control_retained: int,
    control_n: int,
    treat_retained: int,
    treat_n: int,
    alpha: float = 0.05,
    test_name: str = "Retention test"
) -> dict:
    """
    Two-proportion z-test for a retention A/B experiment.

    Parameters
    ----------
    control_retained : players retained in control group
    control_n        : total players in control group
    treat_retained   : players retained in treatment group
    treat_n          : total players in treatment group
    alpha            : significance level (default 0.05)
    test_name        : label for the experiment

    Returns
    -------
    dict with p_control, p_treat, lift, p_value, significant
    """
    p_control = control_retained / control_n
    p_treat   = treat_retained   / treat_n

    count = np.array([treat_retained, control_retained])
    nobs  = np.array([treat_n, control_n])
    stat, pval = stats.proportions_ztest(count, nobs)

    lift = (p_treat - p_control) / p_control * 100

    result = {
        'test_name':   test_name,
        'p_control':   round(p_control, 4),
        'p_treat':     round(p_treat, 4),
        'lift_pct':    round(lift, 2),
        'z_stat':      round(stat, 4),
        'p_value':     round(pval, 4),
        'significant': pval < alpha,
        'alpha':       alpha
    }

    print(f"\n{'='*50}")
    print(f"A/B Test: {test_name}")
    print(f"{'='*50}")
    print(f"  Control retention:   {p_control:.1%}  (n={control_n:,})")
    print(f"  Treatment retention: {p_treat:.1%}  (n={treat_n:,})")
    print(f"  Lift:                {lift:+.1f}%")
    print(f"  Z-statistic:         {stat:.4f}")
    print(f"  P-value:             {pval:.4f}")
    print(f"  Significant (a={alpha}): {'YES - ship it!' if pval < alpha else 'NO - keep testing'}")

    return result


def minimum_sample_size(
    baseline_rate: float,
    min_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """
    Calculate minimum sample size per group for a given effect size.

    Parameters
    ----------
    baseline_rate          : current retention rate (e.g. 0.60)
    min_detectable_effect  : smallest lift worth detecting (e.g. 0.05 = 5%)
    alpha                  : false positive rate (default 0.05)
    power                  : desired statistical power (default 0.80)

    Returns
    -------
    n : minimum sample size per group
    """
    p1 = baseline_rate
    p2 = baseline_rate + min_detectable_effect

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)

    p_bar = (p1 + p2) / 2
    n = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
         z_beta  * np.sqrt(p1 * (1-p1) + p2 * (1-p2))) ** 2 / (p2 - p1) ** 2

    n = int(np.ceil(n))
    print(f"\nMinimum sample size per group: {n:,}")
    print(f"  Baseline retention:  {p1:.1%}")
    print(f"  Target retention:    {p2:.1%}")
    print(f"  Min detectable lift: {min_detectable_effect*100:.1f}pp")
    print(f"  Alpha: {alpha} | Power: {power:.0%}")
    return n


def sequential_test(
    daily_conversions: list,
    daily_n: list,
    baseline_rate: float,
    alpha: float = 0.05
) -> None:
    """
    Sequential (peeking-safe) test that checks significance daily
    using Bonferroni correction — prevents false positives from
    checking results too early.

    Parameters
    ----------
    daily_conversions : list of daily retained counts (treatment)
    daily_n           : list of daily sample sizes (treatment)
    baseline_rate     : control group retention rate
    alpha             : overall significance level
    """
    n_days      = len(daily_conversions)
    alpha_adj   = alpha / n_days  # Bonferroni correction
    cum_conv    = 0
    cum_n       = 0

    print(f"\nSequential test (Bonferroni a={alpha_adj:.4f} per day):")
    print(f"{'Day':<6} {'Cum n':<10} {'Cum rate':<12} {'P-value':<12} {'Significant'}")
    print("-" * 55)

    for day, (conv, n) in enumerate(zip(daily_conversions, daily_n), 1):
        cum_conv += conv
        cum_n    += n
        p_treat   = cum_conv / cum_n

        count = np.array([cum_conv, int(baseline_rate * cum_n)])
        nobs  = np.array([cum_n, cum_n])
        try:
            _, pval = stats.proportions_ztest(count, nobs)
        except Exception:
            pval = 1.0

        sig = "STOP - significant!" if pval < alpha_adj else "-"
        print(f"{day:<6} {cum_n:<10,} {p_treat:<12.1%} {pval:<12.4f} {sig}")
        if pval < alpha_adj:
            print(f"\n  Stopping early on day {day} — effect is real!")
            break


if __name__ == '__main__':
    # Example 1: Free bonus item for at-risk players
    ab_test_retention(
        control_retained=1820, control_n=3000,
        treat_retained=2010,   treat_n=3000,
        test_name="Free bonus item — at-risk segment"
    )

    # Example 2: Win-back push notification
    ab_test_retention(
        control_retained=540,  control_n=1000,
        treat_retained=590,    treat_n=1000,
        test_name="Win-back push notification"
    )

    # Example 3: Minimum sample size calculation
    minimum_sample_size(
        baseline_rate=0.60,
        min_detectable_effect=0.05,
        alpha=0.05,
        power=0.80
    )

    # Example 4: Sequential test over 7 days
    sequential_test(
        daily_conversions=[85, 88, 91, 94, 90, 95, 93],
        daily_n=[140, 140, 140, 140, 140, 140, 140],
        baseline_rate=0.60
    )
