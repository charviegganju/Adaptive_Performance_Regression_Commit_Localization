# Step 3: find the culprit in every instance with 4 methods and compare them.

import csv
import glob
import json

import matplotlib.pyplot as plt

from step2_sprt import Benchmark, fixed_test, sprt_test
from step4_hybrid_retrieval import hybrid_rank

KEYWORDS = ["for (", "while (", "new ", "String.format", "contains(", "synchronized", "Files."]


def rank_commits(instance):
    """Simple ranker: commits whose added lines contain 'slow-looking' code come first.
    (Later this is replaced by the LLM.) It never sees culprit_id or measurements."""
    bench_words = instance["benchmark_context"]["benchmark"].lower()
    scores = []
    for c in instance["commits"]:
        added = [l for l in c["diff"].split("\n") if l.startswith("+")]
        score = sum(k in l for l in added for k in KEYWORDS)
        if any(f.split("/")[-1].replace(".java", "").lower() in bench_words for f in c["files"]):
            score += 2
        scores.append(score)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [instance["commits"][i]["id"] for i in order]


def linear_scan(instance, bench, test):
    """Check commits oldest -> newest: is it slower than its parent?"""
    for c in instance["commits"]:
        if test(bench, c["id"], c["parent_id"])[0] == "slower":
            return c["id"]
    return None


def bisection(instance, bench, test):
    """git bisect: compare the middle commit with the good commit, keep the half with the culprit."""
    ids = [c["id"] for c in instance["commits"]]
    lo, hi = -1, len(ids) - 1          # culprit is somewhere in (lo, hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if test(bench, ids[mid], instance["good_id"])[0] == "slower":
            hi = mid
        else:
            lo = mid
    return ids[hi]


def ranked_search(instance, bench, test, ranker=rank_commits):
    """Our method: check the most suspicious commit first.
    A commit is the culprit if it is slower than good but its parent is not."""
    ids = [c["id"] for c in instance["commits"]]
    good = instance["good_id"]
    for c in ranker(instance):
        if test(bench, c, good)[0] == "slower":
            i = ids.index(c)
            if i == 0 or test(bench, ids[i - 1], good)[0] == "same":
                return c
    return None


rows = []
for path in sorted(glob.glob("instances/*.json")):
    instance = json.load(open(path))
    measurements = json.load(open(path.replace("instances", "measurements")))
    delta = max(instance["benchmark_context"]["observed_slowdown"] / 2, 0.01)
    sprt = lambda b, x, y: sprt_test(b, x, y, delta)

    methods = {
        "Linear scan, 10 runs": (linear_scan, fixed_test),
        "Bisection, 10 runs": (bisection, fixed_test),
        "Bisection + SPRT": (bisection, sprt),
        "Keyword ranker + SPRT": (ranked_search, sprt),
        "Hybrid retrieval + SPRT (ours)": (lambda i, b, t: ranked_search(i, b, t, hybrid_rank), sprt),
    }
    for name, (search, test) in methods.items():
        bench = Benchmark(measurements)
        found = search(instance, bench, test)
        correct = found == instance["culprit_id"]
        rows.append({"instance": instance["instance_id"], "delta": instance["meta"]["delta"],
                     "method": name, "correct": correct, "executions": bench.executions})
        print(f"{instance['instance_id']:14} {name:31} correct={correct}  runs={bench.executions}")

with open("results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nSummary")
names, avg_runs = [], []
for name in dict.fromkeys(r["method"] for r in rows):
    mine = [r for r in rows if r["method"] == name]
    accuracy = sum(r["correct"] for r in mine) / len(mine)
    runs = sum(r["executions"] for r in mine) / len(mine)
    print(f"{name:31} accuracy={accuracy:.0%}  average runs={runs:.1f}")
    names.append(name)
    avg_runs.append(runs)

plt.figure(figsize=(8, 4))
plt.barh(names, avg_runs, color=["grey"] * (len(names) - 1) + ["tab:blue"])
plt.xlabel("average benchmark runs needed")
plt.title("Cost of finding the culprit commit")
plt.tight_layout()
plt.savefig("results.png", dpi=150)
print("\nSaved results.csv and results.png")
