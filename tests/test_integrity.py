import json

import numpy as np
import pytest
import yaml
from pathlib import Path

from foresightbev.errors import ConfigurationError, JobConflict
from foresightbev.evaluation import integrity as I
from foresightbev.evaluation import metrics as M
from foresightbev.recording import environment, provenance_table, source_hash

ROOT = Path(__file__).resolve().parents[1]
GOOD = dict(closed_loop_seeds=[400, 415], calibration=dict(fit_seeds=[200, 220], test_seeds=[230, 240]))


def test_shipped_configs_have_disjoint_seeds():
    for name in ("experiment_default", "experiment_final"):
        I.check_seed_separation(yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text()))


def test_final_config_avoids_development_seeds():
    I.check_seed_separation(GOOD, final=True)


@pytest.mark.parametrize("change,message", [
    (dict(closed_loop_seeds=[210, 215]), "leakage"), (dict(closed_loop_seeds=[305, 310]), "leakage"),
    (dict(calibration=dict(fit_seeds=[200, 235], test_seeds=[230, 240])), "leakage"), (dict(closed_loop_seeds=[5, 5]), "empty"),
])
def test_seed_overlap_is_refused(change, message):
    with pytest.raises(ConfigurationError, match=message):
        I.check_seed_separation(GOOD | change)


def test_final_run_refuses_development_seeds():
    with pytest.raises(ConfigurationError, match="development"):
        I.check_seed_separation(GOOD | dict(closed_loop_seeds=[105, 120]), final=True)


def test_existing_results_are_not_overwritten(tmp_path):
    (tmp_path / "summary.json").write_text("{}")
    with pytest.raises(JobConflict):
        I.claim_result_directory(tmp_path, force=False)
    I.claim_result_directory(tmp_path, force=True)


def test_records_append_and_never_rewrite(tmp_path):
    I.append_record(tmp_path, dict(name="a"))
    I.append_record(tmp_path, dict(name="b"))
    assert [json.loads(x)["name"] for x in (tmp_path / "experiments.jsonl").read_text().splitlines()] == ["a", "b"]


def test_config_hash_ignores_key_order():
    assert I.config_hash(dict(a=1, b=2)) == I.config_hash(dict(b=2, a=1))


def test_environment_identifies_uncommitted_code():
    assert environment()["source_sha256"] == source_hash() and len(source_hash()) == 64


def test_provenance_reports_what_actually_ran():
    real = provenance_table("recorded", "nuScenes", "segformer_b2_cityscapes")
    assert real[0]["kind"] == "recorded" and real[1]["kind"] == "pretrained"
    assert not any(r["component"] == "Execution" for r in real)
    assert provenance_table("simulated", "x")[1]["kind"] == "none"


def test_provenance_names_the_detector_that_ran():
    rows = {r["component"]: r for r in provenance_table("recorded", "nuScenes", "s", detector="streampetr")}
    assert rows["Detection"]["kind"] == "pretrained" and "unverified" in rows["Detection"]["detail"]
    assert {r["component"]: r for r in provenance_table("recorded", "x", detector="image_size")}["Detection"]["kind"] == "heuristic"


def test_cluster_bootstrap_is_wider_than_pretending_rows_are_independent():
    rng = np.random.default_rng(0)
    groups = [(np.full(50, .5), np.full(50, float(i % 2))) for i in range(10)]
    interval = M.cluster_bootstrap(groups, M.ece, rng)
    assert interval["episodes"] == 10 and interval["high"] > interval["low"]
