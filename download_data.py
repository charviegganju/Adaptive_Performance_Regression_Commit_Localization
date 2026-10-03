# Downloads only the 2 benchmark files this project needs from the ICPE dataset.
import os
import urllib.parse
import urllib.request

os.chdir(os.path.dirname(os.path.abspath(__file__)))

BASE = "https://raw.githubusercontent.com/SEALABQualityGroup/icpe-data-challenge-jmh/master/timeseries/"
FILES = [
    "HdrHistogram__HdrHistogram#bench.HdrHistogramEncodingBench.encodeIntoCompressedByteBuffer#latencySeriesName=case3&numberOfSignificantValueDigits=3.json",
    "prometheus__client_java#io.prometheus.benchmark.SummaryBenchmark.prometheusSimpleSummaryChildBenchmark#.json",
]
FOLDER = "data/icpe-data-challenge-jmh/timeseries/"

os.makedirs(FOLDER, exist_ok=True)
for name in FILES:
    url = BASE + urllib.parse.quote(name)
    print("Downloading", name[:60], "...")
    urllib.request.urlretrieve(url, FOLDER + name)
print("Done. Files are in", os.path.abspath(FOLDER))
