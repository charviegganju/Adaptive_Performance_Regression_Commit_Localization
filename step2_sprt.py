# Step 2: the statistical tests.
# Question for every test: "is commit A slower than commit B?"

import math

import numpy as np


class Benchmark:
    """Gives benchmark runs for a commit and counts how many we used.
    Runs are reused: if we already ran a commit 5 times, asking for 5 again is free."""

    def __init__(self, measurements):
        self.measurements = measurements
        self.used = {}
        self.executions = 0

    def runs(self, commit, n):
        already = self.used.get(commit, 0)
        if n > already:
            self.executions += n - already
            self.used[commit] = n
        return np.log(self.measurements[commit][:n])   # log so 5% slower = +0.049


def sprt_test(bench, a, b, delta, alpha=0.01, beta=0.1, max_runs=30):
    """Sequential test: add one run at a time until we are sure.
    Returns ('slower' | 'same' | 'unsure', runs used, evidence after each run)."""
    upper = math.log((1 - beta) / alpha)   # cross this -> A is slower
    lower = math.log(beta / (1 - alpha))   # cross this -> A is not slower
    theta = math.log(1 + delta)            # the slowdown we are looking for
    evidence = []
    for n in range(3, max_runs + 1):
        x, y = bench.runs(a, n), bench.runs(b, n)
        diff = x.mean() - y.mean()
        var = max((x.var(ddof=1) + y.var(ddof=1)) / 2, 1e-12)
        llr = n * (theta * diff - theta ** 2 / 2) / (2 * var)
        evidence.append(llr)
        if llr >= upper:
            return "slower", n, evidence
        if llr <= lower:
            return "same", n, evidence
    return "unsure", max_runs, evidence


def fixed_test(bench, a, b, n=10):
    """Old way: always n runs each, then a t-test."""
    x, y = bench.runs(a, n), bench.runs(b, n)
    t = (x.mean() - y.mean()) / math.sqrt(x.var(ddof=1) / n + y.var(ddof=1) / n + 1e-18)
    return ("slower" if t > 3 else "same"), n, [t]
