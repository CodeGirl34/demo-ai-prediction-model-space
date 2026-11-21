"""Download the NASA SMAP/MSL anomaly dataset from Kaggle into data/raw/smap_msl.

Dataset: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
(original source: Hundman et al., 2018 / github.com/khundman/telemanom)
"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "smap_msl"

if __name__ == "__main__":
    if (DEST / "labeled_anomalies.csv").exists():
        print(f"Already downloaded: {DEST}")
    else:
        print("Downloading (~86 MB)...")
        with urllib.request.urlopen(URL) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(DEST)
        print(f"Extracted to {DEST}")
