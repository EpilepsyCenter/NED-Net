#!/usr/bin/env python3
"""Refresh the training tree from the live sidecars, dropping Mir's rejections.

The live EDF tree accumulates human review: Mir's adjudicated candidates, the
U-Net's pending detections, and manual additions made in the UI
(`source="manual"`). The training tree (`~/train_nomirneg`) is a flat directory of
symlinked EDFs plus real sidecars, used for training only, so the live tree keeps
the reviews and nothing overwrites them.

With ``--keep-mir-rejections`` (preferred) this is a straight copy of the live
sidecars, and Mir's rows are handled at training time instead -- see below.

Without it, this copies live sidecars into the training tree while removing
**only** Mir's `rejected` rows. Those mean "not a convulsive/behavioural seizure" and were
confirmed by visual review (2026-10-07, a 180 s block of clear bursting on UI
Ch6) to contain real non-convulsive activity, so as seizure-detection hard
negatives they are mislabelled. Everything else is preserved:

  * Mir's `confirmed` seizures        -> kept (training positives)
  * manual additions (`source=manual`) -> kept (training positives)
  * U-Net `pending`                  -> kept; trainers ignore pending
  * U-Net `confirmed`/`rejected`      -> kept; these ARE valid, adjudicated by us
  * SV2A sidecars                    -> untouched (symlinked, not managed here)

Why ``--keep-mir-rejections`` is preferred: those rows have two distinct uses and
deleting them only serves the first.

  1. as **hard negatives** they are WRONG (they mean "not convulsive", not "not a
     seizure"), so they must be excluded -- which deletion achieves; but
  2. as **regions to keep random background out of** they are RIGHT, because they
     contain real non-convulsive activity. Deletion destroys this, so background is
     then sampled from exactly the regions known to contain events.

Keep them, and pass both at training time:

    --hard-neg-exclude-method mir_candidate --bg-avoid-rejected

Run it after every review round. That is the loop: detect -> review -> refresh ->
retrain. **The refresh is the step that fails silently** -- it was last run before
the 2026-10-07/08 review sessions, so arms A/A2/A3 trained without any of that work
(see RUN_LOG 2026-10-08). Always check the printed manual/adjudicated counts.

Usage:
    python scripts/lunarc/refresh_training_tree.py --dry-run
    python scripts/lunarc/refresh_training_tree.py
    python scripts/lunarc/refresh_training_tree.py --only-reviewed   # skip untouched files
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path

LIVE = "/lunarc/nobackup/projects/lu2026-2-60/RAM_GDNF_2025"
TREE = "~/train_nomirneg"


def _is_mir_rejection(a: dict) -> bool:
    f = a.get("features") or {}
    return (a.get("label") == "rejected"
            and a.get("source") == "detector"
            and f.get("detection_method") == "mir_candidate")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default=LIVE)
    ap.add_argument("--tree", default=TREE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-mir-rejections", action="store_true",
                    help="PREFERRED. Keep Mir's rejected rows in the tree and let the "
                         "dataset builder handle them: train with "
                         "--hard-neg-exclude-method mir_candidate --bg-avoid-rejected, "
                         "which excludes them as negatives AND keeps random background "
                         "out of those regions. Deleting them (the default below) "
                         "destroys the region information, so background is then drawn "
                         "from exactly the regions known to contain real activity.")
    ap.add_argument("--only-reviewed", action="store_true",
                    help="only files containing a manual addition or an adjudicated "
                         "U-Net detection, i.e. files a human has actually touched")
    a = ap.parse_args()
    tree = Path(os.path.expanduser(a.tree))
    if not tree.is_dir():
        print(f"training tree not found: {tree}"); return 1

    n_files = n_drop = n_kept = n_man = n_adj = n_skip = n_mir = 0
    for src in sorted(Path(a.live).rglob("*_ned_annotations.json")):
        try:
            d = json.loads(src.read_text())
        except Exception as e:
            print(f"  unreadable, skipped: {src.name} ({e})"); continue
        anns = d.get("annotations", [])

        man = [x for x in anns if x.get("source") == "manual"]
        adj = [x for x in anns
               if (x.get("features") or {}).get("detection_method") == "ml_unet"
               and x.get("label") in ("confirmed", "rejected")]
        if a.only_reviewed and not man and not adj:
            n_skip += 1
            continue

        keep = (list(anns) if a.keep_mir_rejections
                else [x for x in anns if not _is_mir_rejection(x)])
        n_mir += sum(1 for x in anns if _is_mir_rejection(x))
        n_drop += len(anns) - len(keep)
        n_kept += len(keep)
        n_man += len(man)
        n_adj += len(adj)
        n_files += 1

        dest = tree / src.name
        if a.dry_run:
            continue
        out = dict(d)
        out["annotations"] = keep
        out["n_annotations"] = len(keep)
        out["refreshed_from_live"] = True
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=2))
        tmp.replace(dest)

    verb = "would refresh" if a.dry_run else "refreshed"
    print(f"{verb} {n_files} sidecars in {tree}")
    if a.keep_mir_rejections:
        print(f"  KEPT {n_mir} of Mir's rejected rows (exclude them at training time "
              f"with --hard-neg-exclude-method mir_candidate --bg-avoid-rejected)")
    else:
        print(f"  dropped {n_drop} of Mir's rejected rows")
    print(f"  kept {n_kept} annotations, including {n_man} manual additions "
          f"and {n_adj} adjudicated U-Net detections")
    if n_skip:
        print(f"  skipped {n_skip} files with no human review (--only-reviewed)")
    print("\nAfter this, re-run --analyze on the tree before training: the positive")
    print("count should have risen by the number of manual additions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
