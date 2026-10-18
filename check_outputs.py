"""Compare the regenerated outputs (manuscript v1_16_1) with the registered ones (expected/expected_outputs.json).

A file passes when its bytes equal the registered file. A JSON record whose bytes differ passes when every value is equal except
fields that record run time or wall-clock time (volatile) or SHA-256 provenance of input files (provenance). The validators and the
inset-path script of Figure 2 record the whole-file hashes of SIRTS1_ref_simulation.py and SIRTS1_ref_coupling.py, and the copies
in this folder differ from the validated sources only by their main guards (see README), so those hashes and the hashes chained
from them change while every number stays equal. Every such field is listed. Any other difference fails, and so does a PDF or
text file whose bytes differ. A NaN in a JSON record equals a NaN at the same place (JSON stores it as NaN, and NaN never equals
itself in Python). Exit status 0 only when every expected output passes.

usage (from this folder): python3 -B check_outputs.py
"""
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VOLATILE = {"secs", "seconds", "elapsed", "wall", "utc", "time", "date", "timestamp", "created", "started", "finished"}


def sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def diff(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(("structure", f"{path}/{k}"))
            else:
                diff(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
        key = path.rsplit("/", 1)[-1].split("[")[0].lower()
        kind = "volatile" if (key in VOLATILE or key.endswith("_utc") or key.endswith("_secs")) else \
               "provenance" if "sha256" in path.lower() else "value"
        out.append((kind, path))


def main():
    exp = json.load(open(os.path.join(HERE, "expected", "expected_outputs.json")))
    a, b = os.path.join(HERE, "work/data/jia2026joint_figstyle.mplstyle"), os.path.join(HERE, "work/scripts/jia2026joint_figstyle.mplstyle")
    ok = sha(a) == sha(b)
    print(f"style files equal: {ok}")
    for rel, rec in sorted(exp.items()):
        p = os.path.join(HERE, rel)
        if not os.path.isfile(p):
            print(f"FAIL  {rel}: missing"); ok = False; continue
        if sha(p) == rec["sha256"]:
            print(f"pass  {rel} ({rec['asset']}, {rec['label']}): byte-equal"); continue
        if rel.endswith(".json"):
            ref = os.path.join(HERE, "expected", rel.replace("/", "__"))
            out = []
            diff(json.load(open(ref)), json.load(open(p)), "", out)
            bad = [x for x in out if x[0] in ("value", "structure")]
            kinds = {k: sum(1 for x in out if x[0] == k) for k in ("volatile", "provenance")}
            if bad:
                ok = False
                print(f"FAIL  {rel} ({rec['asset']}): {len(bad)} value or structure differences, first {bad[:5]}")
            else:
                print(f"pass  {rel} ({rec['asset']}, {rec['label']}): values equal; differing fields: {kinds['volatile']} volatile, "
                      f"{kinds['provenance']} provenance: " + "; ".join(x[1] for x in out))
            continue
        ok = False
        note = " (PDF bytes depend on the Matplotlib version; see SOFTWARE_AND_SEEDS.md)" if rel.endswith(".pdf") else ""
        print(f"FAIL  {rel} ({rec['asset']}): bytes differ from the registered file{note}")
    print("ALL PASS" if ok else "NOT ALL PASS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
