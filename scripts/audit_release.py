"""Release audit: invariants over every stored showcase run, read through the real API.

Run: .venv/bin/python scripts/audit_release.py
Writes experiments/end_to_end/results/release_audit/summary.json. Exits non-zero on any failure.
"""

import json
import math
import sys
from collections import Counter

from fastapi.testclient import TestClient

from foresightbev.api.app import create_app
from foresightbev.config import load_settings

settings = load_settings()
CAMERAS = {"CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}
FAULT_LOST = range(14, 26)


def main() -> int:
    failures: list[str] = []
    totals = Counter()
    max_skew = 0.0
    speeds = []
    with TestClient(create_app(settings)) as client:
        runs = [r for s in client.get("/api/v1/sequences").json() for r in s["runs"] if r["id"].startswith("showcase_")]
        for run in runs:
            name = run["id"]
            fault = name.endswith("front_camera_loss")
            timeline = client.get(f"/api/v1/runs/{name}/timeline").json()
            previous: dict[int, dict] = {}
            for row in timeline:
                f = row["frame"]
                r = client.get(f"/api/v1/runs/{name}/frames/{f}/record").json()
                where = f"{name} frame {f}"
                totals["frames"] += 1
                # Cameras and synchronisation.
                if set(r["cameras"]) != CAMERAS:
                    failures.append(f"{where}: cameras {sorted(r['cameras'])}")
                lost = [c for c, v in r["cameras"].items() if v["status"] != "ok"]
                expected = ["CAM_FRONT"] if fault and f in FAULT_LOST else []
                if lost != expected:
                    failures.append(f"{where}: lost {lost}, expected {expected}")
                views = client.get(f"/api/v1/runs/{name}/frames/{f}/cameras").json()
                max_skew = max(max_skew, *(abs(v["skew_ms"]) for v in views if v["skew_ms"] is not None))
                # Occupancy states partition the grid.
                if abs(sum(r["state_fractions"].values()) - 1) > 1e-6:
                    failures.append(f"{where}: state fractions sum {sum(r['state_fractions'].values())}")
                # Detection.
                for d in r["detections"]:
                    if d["cls"] not in ("VEHICLE", "PERSON") or not 0 <= d["score"] <= 1 or d["score"] < .3:
                        failures.append(f"{where}: detection {d['cls']} score {d['score']}")
                totals["detections"] += len(r["detections"])
                # Tracking: unique IDs, ages advance, position follows velocity.
                ids = [t["id"] for t in r["tracks"]]
                if len(ids) != len(set(ids)):
                    failures.append(f"{where}: duplicate track IDs")
                current = {t["id"]: t for t in r["tracks"]}
                for i, t in current.items():
                    if i in previous and t["age"] <= previous[i]["age"]:
                        failures.append(f"{where}: track {i} age did not advance")
                    if not all(math.isfinite(t[k]) for k in ("x", "y", "vx", "vy")):
                        failures.append(f"{where}: track {i} not finite")
                previous = current
                totals["tracks"] += len(ids)
                # Forecast: causal, seconds, increasing.
                h = r["forecast"]["horizons"]
                if r["forecast"]["last_input"] > r["forecast"]["t_ref"] or h != sorted(h) or abs(h[-1] - 3) > 1e-6:
                    failures.append(f"{where}: forecast timing {r['forecast']['last_input']} {r['forecast']['t_ref']} {h[-1]}")
                # Decision: traceable, consistent with the timeline.
                d = r["decision"]
                totals[d["status"]] += 1
                if d["status"] != row["status"] or not d["reasons"]:
                    failures.append(f"{where}: decision {d['status']} {row['status']} {d['reasons']}")
                ids = {c["id"] for c in r["candidates"]}
                if d["selected"] not in ids:
                    failures.append(f"{where}: selected {d['selected']} not a candidate")
                if d["status"] == "fallback" and not next(c for c in r["candidates"] if c["id"] == d["selected"])["fallback"]:
                    failures.append(f"{where}: fallback selected a non-fallback plan")
                if d["status"] == "pass" and not next(c for c in r["candidates"] if c["id"] == d["selected"])["nominal"]:
                    failures.append(f"{where}: pass did not keep the nominal plan")
                for c in r["candidates"]:
                    if not 0 <= c["event_score"] <= 1:
                        failures.append(f"{where}: event score {c['event_score']}")
                if fault and f in FAULT_LOST:
                    selected = next(c for c in r["candidates"] if c["id"] == d["selected"])
                    if d["status"] == "pass" or "CAMERA_COVERAGE_LOST" not in d["reasons"]:
                        speeds.append((f, d["status"], d["reasons"], selected["end_speed"], r["ego"]["speed"]))
    report = dict(runs=len(runs), totals=dict(totals), max_camera_skew_ms=max_skew, fault_frames_passed=speeds, failures=len(failures), first_failures=failures[:20])
    out = settings.data_dir / "results/release_audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
