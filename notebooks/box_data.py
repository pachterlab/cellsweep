"""Download the matrices behind the paper's figures from Box instead of recomputing them.

config/box_urls.yaml maps each file's path (relative to notebooks/) to its Box URL. A .tar.gz entry holds an
.mtx/_genes.csv/_barcodes.csv triplet (SoupX, DecontX) and is extracted next to where it would go.
"""
import os
import subprocess
import tarfile

import yaml

notebooks_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(notebooks_dir, "config", "box_urls.yaml")) as f:
    box_urls = yaml.safe_load(f)


def download_from_box(*prefixes):
    """Download every file in config/box_urls.yaml whose path starts with one of `prefixes` (relative to
    notebooks/, e.g. "data/pbmc8k/idempotency/") and is not on disk yet."""
    for rel, url in box_urls.items():
        if not rel.startswith(prefixes):
            continue
        path = os.path.join(notebooks_dir, rel)
        is_tar = rel.endswith(".tar.gz")
        if os.path.exists(path[:-len(".tar.gz")] + ".mtx" if is_tar else path):
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading {rel} from Box")
        subprocess.run(["wget", "-q", "-O", f"{path}.part", url], check=True)   # .part so a failed download is never taken for the file
        os.rename(f"{path}.part", path)
        if is_tar:
            with tarfile.open(path) as tar:
                tar.extractall(os.path.dirname(path))
            os.remove(path)
