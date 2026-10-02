# Finding the commit that made a benchmark slower

Performance measurements are noisy, so to know if a commit made code slower we have to run the
benchmark many times. This project tries to find the slow commit using as few runs as possible.

## Files

- `step1_make_instances.py` - takes 20 real commits from a project, inserts one fake slow commit at a
  random position, and creates benchmark runs for every commit using real noise from the ICPE 2023
  JMH dataset (the culprit and later commits are made `delta` slower).
- `step2_sprt.py` - the tests: a fixed test (always 10 runs) and SPRT (adds runs one by one until sure).
- `step3_run.py` - finds the culprit with 5 methods and compares accuracy and number of runs.
- `step4_hybrid_retrieval.py` - hybrid retrieval ranker: BM25 (keywords) + embeddings (meaning),
  fused with Reciprocal Rank Fusion. Run it alone to compare BM25 vs dense vs hybrid.

## Run

```
pip install -r requirements.txt
pip install sentence-transformers   # for the embedding retriever
git clone --depth 1 https://github.com/SEALABQualityGroup/icpe-data-challenge-jmh.git data/icpe-data-challenge-jmh
git clone https://github.com/HdrHistogram/HdrHistogram.git data/HdrHistogram
git clone https://github.com/prometheus/client_java.git data/client_java
python step1_make_instances.py
python step4_hybrid_retrieval.py
python step3_run.py
```

Note: the noise is real (from the dataset) but the slowdown is injected and replayed, not measured live.
