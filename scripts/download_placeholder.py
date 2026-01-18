
"""Placeholder downloader.

Some ICS datasets require manual acceptance, or have multiple mirrors.
Use this as a template:

python scripts/download_placeholder.py --url <URL> --out data/raw.zip

"""
import argparse, requests
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(args.url, stream=True) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
    print("Downloaded", out)

if __name__ == "__main__":
    main()
