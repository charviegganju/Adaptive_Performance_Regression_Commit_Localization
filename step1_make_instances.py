# Step 1: make test instances.
# For each project: take 20 real commits, insert one fake "slow" commit at a random
# position, and create benchmark measurements using real noise from the ICPE dataset.

import json
import os
import random
import subprocess

import numpy as np

random.seed(42)

DATASET = "data/icpe-data-challenge-jmh/timeseries/"

PROJECTS = [
    {
        "name": "HdrHistogram",
        "repo": "data/HdrHistogram",
        "tag": "HdrHistogram-2.1.12",
        "benchmark": "HdrHistogram__HdrHistogram#bench.HdrHistogramEncodingBench.encodeIntoCompressedByteBuffer#latencySeriesName=case3&numberOfSignificantValueDigits=3.json",
        "file": "src/main/java/org/HdrHistogram/AbstractHistogram.java",
        "method": "encodeIntoCompressedByteBuffer",
    },
    {
        "name": "prometheus",
        "repo": "data/client_java",
        "tag": "parent-0.9.0",
        "benchmark": "prometheus__client_java#io.prometheus.benchmark.SummaryBenchmark.prometheusSimpleSummaryChildBenchmark#.json",
        "file": "simpleclient/src/main/java/io/prometheus/client/Summary.java",
        "method": "observe",
    },
]

DELTAS = [0.02, 0.05, 0.10, 0.30]   # how much slower the culprit makes the benchmark

# slow code patterns from our slide 7 (commit message, added code)
PATTERNS = [
    ("Add consistency check", "long checksum = 0;\nfor (int i = 0; i < 1000; i++) { checksum += i * 31; }"),
    ("Add debug context", "StringBuilder context = new StringBuilder();\ncontext.append(getClass().getName()).append(System.identityHashCode(this));"),
    ("Normalize order", "List<Integer> order = new ArrayList<>();\nfor (int i = 0; i < 50; i++) { if (!order.contains(i)) order.add(i); }"),
    ("Add label for logs", "String label = String.format(\"%s-%d\", getClass().getSimpleName(), System.nanoTime());"),
]

WARMUP = 500           # ignore first 500 iterations of every fork
RUNS_PER_COMMIT = 40   # how many benchmark runs we prepare per commit


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def get_commits(repo, tag, n=20):
    """Last n+1 commits before the tag (oldest first). The first one is the 'good' commit."""
    log = git(repo, "log", "--first-parent", f"-{n + 1}", "--format=%H|%P|%an|%aI|%s", tag)
    commits = []
    for line in reversed(log.strip().split("\n")):
        sha, parents, author, date, message = line.split("|", 4)
        parent = parents.split()[0] if parents else None
        files = git(repo, "diff", "--name-only", parent, sha).split() if parent else []
        diff = git(repo, "diff", parent, sha, "--", "*.java")[:2000] if parent else ""
        commits.append({"id": sha, "parent_id": parent, "author": author,
                        "timestamp": date, "message": message, "files": files, "diff": diff})
    return commits


def make_fake_commit(project, parent, author, message, code):
    """A synthetic commit that adds slow code at the start of the benchmarked method."""
    added = "\n".join("+        " + line for line in code.split("\n"))
    diff = (f"diff --git a/{project['file']} b/{project['file']}\n"
            f"@@ public ... {project['method']}(...) {{\n{added}\n")
    return {"id": "%040x" % random.getrandbits(160), "parent_id": parent, "author": author,
            "timestamp": "", "message": message, "files": [project["file"]], "diff": diff}


def one_run(steady):
    """One benchmark run = average of 1000 iterations from a random real fork."""
    fork = random.randrange(steady.shape[0])
    start = random.randrange(steady.shape[1] - 1000)
    return float(steady[fork, start:start + 1000].mean())


os.makedirs("instances", exist_ok=True)
os.makedirs("measurements", exist_ok=True)

for project in PROJECTS:
    data = np.array(json.load(open(DATASET + project["benchmark"])))  # 10 forks x 3000 iterations
    steady = data[:, WARMUP:]
    all_commits = get_commits(project["repo"], project["tag"])
    good, window = all_commits[0], all_commits[1:]

    for i, delta in enumerate(DELTAS):
        message, code = PATTERNS[i % len(PATTERNS)]
        pos = random.randint(0, len(window))                 # where the culprit goes
        parent = good["id"] if pos == 0 else window[pos - 1]["id"]
        fake = make_fake_commit(project, parent, random.choice(window)["author"], message, code)
        commits = window[:pos] + [fake] + window[pos:]
        if pos < len(window):
            commits[pos + 1] = dict(commits[pos + 1], parent_id=fake["id"])

        culprit_range = [c["id"] for c in commits[pos:]]
        runs = {}
        for c in [good] + commits:
            slow = (1 + delta) if c["id"] in culprit_range else 1.0
            runs[c["id"]] = [one_run(steady) * slow for _ in range(RUNS_PER_COMMIT)]

        alert = np.mean(runs[commits[-1]["id"]][:10]) / np.mean(runs[good["id"]][:10]) - 1
        instance_id = f"{project['name']}-{i + 1}"
        instance = {
            "instance_id": instance_id,
            "good_id": good["id"],
            "bad_id": commits[-1]["id"],
            "culprit_id": fake["id"],          # hidden from ranker and SPRT
            "culprit_range": culprit_range,    # hidden from ranker and SPRT
            "commits": commits,
            "benchmark_context": {"project": project["name"], "benchmark": project["benchmark"],
                                  "observed_slowdown": round(float(alert), 4)},
            "meta": {"noise": "real, from ICPE 2023 JMH dataset",
                     "effect": "injected and replayed (real samples x (1 + delta)), not measured live",
                     "delta": delta, "culprit_position": pos + 1},
        }
        json.dump(instance, open(f"instances/{instance_id}.json", "w"), indent=2)
        json.dump(runs, open(f"measurements/{instance_id}.json", "w"))
        print(f"{instance_id}: delta={delta}  culprit at #{pos + 1} of {len(commits)}")
