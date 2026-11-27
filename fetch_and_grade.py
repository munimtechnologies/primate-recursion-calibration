# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""Grade primate-recursion-calibration from the files committed in its public repository.

Downloads the registered grading code (grade.py, test_grade.py, committed 2026-11-25 before any
job output existed) and the three job outputs from the repository, records the SHA-256 of every
file, runs the grading code's own unit tests, and only if they pass runs the grade.

  python fetch_and_grade.py --out-dir /out [--ref main]
Standard library only.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

REPO = "munimtechnologies/primate-recursion-calibration"
FILES = {
    "grade.py": "grade.py",
    "test_grade.py": "test_grade.py",
    "real_results.json": "results/real/real_results.json",
    "worlds_a.json": "results/worlds-a/worlds_results.json",
    "worlds_b.json": "results/worlds-b/worlds_results.json",
}


def fetch(ref, repo_path, dest, tries=12, wait_s=300):
    """Download one file; if it is not in the repo yet (the commit may land just after this job
    starts), retry every wait_s seconds, at most `tries` times."""
    url = "https://raw.githubusercontent.com/%s/%s/%s" % (REPO, ref, repo_path)
    for attempt in range(1, tries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": "mv2-grade-job"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code != 404 or attempt == tries:
                raise
            print("%s not in the repo yet (try %d of %d); waiting %d s" % (repo_path, attempt, tries, wait_s))
            time.sleep(wait_s)
    with open(dest, "wb") as fh:
        fh.write(data)
    return url, hashlib.sha256(data).hexdigest(), len(data)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ref", default="main")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="recgrade-")
    manifest = {"repo": REPO, "ref": a.ref, "files": {}}
    for local, repo_path in FILES.items():
        url, sha, size = fetch(a.ref, repo_path, os.path.join(work, local))
        manifest["files"][local] = {"url": url, "sha256": sha, "bytes": size}
        print("fetched %s (%d bytes, sha256 %s)" % (repo_path, size, sha))
    sys.path.insert(0, work)
    os.chdir(work)
    suite = unittest.defaultTestLoader.loadTestsFromName("test_grade")
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    manifest["tests_run"] = res.testsRun
    manifest["tests_ok"] = res.wasSuccessful()
    with open(os.path.join(a.out_dir, "inputs_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    if not res.wasSuccessful():
        print("GRADE_STOP: grading code's tests failed; no grade computed")
        return 2
    import grade  # noqa: E402  (the registered grading code, fetched above)
    rc = grade.main(["--real", "real_results.json", "--worlds", "worlds_a.json", "worlds_b.json",
                     "--out-dir", a.out_dir])
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
