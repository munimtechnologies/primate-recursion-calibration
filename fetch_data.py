# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""Download Bosshard et al.'s marmoset sequence table from their public OSF project
(osf.io/4x6ed, the view-only link given in the 2024 paper's data statement) and check it against
the size, MD5 and SHA-256 that OSF's API published (read 2026-11-24). Not redistributed.
Usage: python fetch_data.py DEST_DIR"""
import hashlib
import json
import os
import sys
import time
import urllib.request

NAME = "marmoset_sequencing_data.csv"
URL = "https://osf.io/download/j6dur/?view_only=cf53fbb199b4432d85cb2978ed1eaba8"
SIZE = 120510
MD5 = "d3846a4f76749f83c0fc0048a9542c90"
SHA256 = "b904dc3b51f5db78e600230d9a3440e50ef5113af57f97897f9ab5af4cf78bbe"


def fetch():
    last = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": "research-data-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("download failed: %r" % (last,))


def main(dest):
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, NAME)
    blob = None
    if os.path.exists(path):
        with open(path, "rb") as fh:
            blob = fh.read()
        if hashlib.md5(blob).hexdigest() != MD5:
            blob = None
    if blob is None:
        blob = fetch()
        with open(path, "wb") as fh:
            fh.write(blob)
    md5 = hashlib.md5(blob).hexdigest()
    sha = hashlib.sha256(blob).hexdigest()
    ok = md5 == MD5 and sha == SHA256 and len(blob) == SIZE
    report = {NAME: {"bytes": len(blob), "md5": md5, "sha256": sha, "matches_osf": ok}}
    with open(os.path.join(dest, "fetch_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(NAME, len(blob), md5, "OK" if ok else "MISMATCH", flush=True)
    if not ok:
        print("data check failed; nothing analysed", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
