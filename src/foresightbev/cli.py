"""Tavyra command line. No network access and no silent downloads."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "experiments/cam2bev_ipm/inputs/1_FRLR_examples"


def data_dir() -> Path:
    from foresightbev.config import load_settings

    return load_settings().data_dir


def runs_dir() -> Path:
    return data_dir() / "runs"


def results_dir() -> Path:
    return data_dir() / "results"


def cmd_sample(args) -> int:
    """Run the pipeline on the one real local sample and score Stages 2-3."""
    from foresightbev.data.calibration import load_rig
    from foresightbev.data.cam2bev_sample import load_sample
    from foresightbev.evaluation.real_sample import evaluate
    from foresightbev.pipeline import ForesightPipeline, PipelineConfig
    from foresightbev.recording import RunRecorder
    from foresightbev.risk.trajectories import RouteContext

    if not SAMPLE.exists():
        print(f"error: real sample not found at {SAMPLE}", file=sys.stderr)
        return 2
    rig_path = ROOT / "configs/rig_cam2bev_1frlr.yaml"
    rig = load_rig(rig_path)
    config = PipelineConfig()
    bundle = load_sample(SAMPLE, rig)
    result = ForesightPipeline(rig, config).step(bundle, RouteContext(desired_speed=8.0, lanes=(0.0,)))
    directory = Path(args.out) if args.out else runs_dir() / "real_cam2bev_sample"
    recorder = RunRecorder(directory, dict(kind="real_sample", title="Cam2BEV real sample", scenario="cam2bev_1frlr_example", seed=None,
                                           mode="pipeline", input="4 simulated semantic cameras, one frame",
                                           caveat="Single frame: tracks cannot confirm, so no motion forecast. Ego speed is declared as 0."),
                           config.grid.bev)
    recorder.add(result)
    metrics = evaluate(SAMPLE, rig_path)
    recorder.finish(dict(status=result.decision.status.value, metrics=metrics))
    print(json.dumps(metrics, indent=1))
    print(f"decision: {result.decision.status.value} {result.decision.selected_id} {[r.value for r in result.decision.reasons]}")
    print(f"output: {directory}")
    return 0


def cmd_run(args) -> int:
    """Record one closed-loop simulated episode."""
    from foresightbev.evaluation.experiments import build_mode, load_experiment, noise_from, scaled_rig
    from foresightbev.recording import RunRecorder
    from foresightbev.risk.calibration import RiskCalibrator
    from foresightbev.sim.backend import BuiltinSim
    from foresightbev.sim.closed_loop import run_episode
    from foresightbev.sim.scenarios import FAMILIES, build_scenario

    if args.scenario not in FAMILIES:
        print(f"error: unknown scenario {args.scenario}; choose from {sorted(FAMILIES)}", file=sys.stderr)
        return 2
    experiment = load_experiment(ROOT / args.config)
    if args.mode not in experiment["modes"]:
        print(f"error: unknown mode {args.mode}; choose from {sorted(experiment['modes'])}", file=sys.stderr)
        return 2
    mode = experiment["modes"][args.mode]
    scenario = build_scenario(args.scenario, args.seed)
    if mode.get("front_dropout"):
        start, length = mode["front_dropout"]
        scenario.dropouts = [("front", start, start + length)]
    rig = scaled_rig(experiment, ROOT)
    sim = BuiltinSim(scenario, rig, noise_from(experiment), seed=args.seed)
    config, oracle = build_mode(mode)
    config.seed = args.seed
    calibrator = None
    calibrator_path = Path(args.calibrator) if args.calibrator else None
    if mode.get("calibrated"):
        if not calibrator_path or not calibrator_path.exists():
            print("error: this mode needs --calibrator <risk_calibrator.json>", file=sys.stderr)
            return 2
        calibrator = RiskCalibrator.load(calibrator_path)
    run_id = args.name or f"{args.scenario}_s{args.seed}_{args.mode}"
    directory = runs_dir() / run_id
    if directory.exists():
        shutil.rmtree(directory)
    recorder = RunRecorder(directory, dict(kind="simulation", title=run_id, scenario=args.scenario, seed=args.seed, mode=args.mode,
                                           input=f"built-in simulator, {len(rig.cameras)} rendered semantic cameras at scale {experiment['image_scale']}",
                                           caveat="Synthetic scripted scenario. Ground truth is stored for display only and never reaches the pipeline."),
                           config.grid.bev)

    def observer(result, simulator):
        truth = [dict(name=a.name, cls=a.semantic.name, **dict(zip(("x", "y"), result.bundle.ego.world_to_ego([a.x, a.y]).tolist())),
                      yaw=a.yaw - result.bundle.ego.yaw, length=a.length, width=a.width) for a in simulator.agents]
        recorder.add(result, truth)

    start = time.perf_counter()
    episode = run_episode(sim, config, args.mode, calibrator, oracle, observer)
    summary = {k: v for k, v in vars(episode).items() if k != "occupancy_stats"}
    summary["wall_s"] = time.perf_counter() - start
    recorder.finish(summary)
    print(json.dumps(summary, indent=1, default=str))
    print(f"output: {directory}")
    return 0


def cmd_eval(args) -> int:
    """Run calibration and the closed-loop ablation matrix."""
    from foresightbev.evaluation import experiments as E
    from foresightbev.evaluation import integrity as I
    from foresightbev.evaluation.real_sample import evaluate
    from foresightbev.evaluation.report import render
    from foresightbev.recording import environment

    config = E.load_experiment(ROOT / args.config)
    if args.smoke:
        config["closed_loop_seeds"] = [100, 101]
        config["calibration"] = dict(fit_seeds=[200, 202], test_seeds=[230, 231])
    ranges = I.check_seed_separation(config, final=args.final)
    out = results_dir() / args.name
    I.claim_result_directory(out, args.force)
    out.mkdir(parents=True, exist_ok=True)
    env = environment()
    record = dict(name=args.name, config=args.config, config_sha256=I.config_hash(config), source_sha256=env["source_sha256"], seeds=ranges, final=args.final)
    (out / "experiment.json").write_text(json.dumps(config, indent=1))
    (out / "environment.json").write_text(json.dumps(env, indent=1))
    I.append_record(data_dir(), record | {"status": "started"})
    try:
        if SAMPLE.exists():
            (out / "real_sample.json").write_text(json.dumps(evaluate(SAMPLE, ROOT / "configs/rig_cam2bev_1frlr.yaml"), indent=1))
        start = time.perf_counter()
        report = E.run_calibration(config, ROOT, out, args.workers)
        print(f"calibration done in {time.perf_counter() - start:.0f}s: raw ECE {report['raw']['ece_10_equal_width']:.4f}, "
              f"calibrated ECE {report['calibrated']['ece_10_equal_width']:.4f}", flush=True)
        episodes = E.run_closed_loop(config, ROOT, out, args.workers, out / "risk_calibrator.json")
        (out / "summary.json").write_text(json.dumps(E.summarize(episodes), indent=1))
        (out / "REPORT.md").write_text(render(out))
    except Exception as error:
        I.append_record(data_dir(), record | {"status": "failed", "error": repr(error)})
        raise
    I.append_record(data_dir(), record | {"status": "completed", "episodes": len(episodes)})
    print(f"evaluation done in {time.perf_counter() - start:.0f}s: {out / 'REPORT.md'}")
    return 0


def _settings():
    from foresightbev.config import load_settings

    return load_settings()


def _sequence_dir(name: str) -> Path:
    return _settings().data_dir / "sequences" / name


def cmd_prepare(args) -> int:
    """Turn a source dataset into label sequences. Inputs are never modified."""
    from foresightbev.data.calibration import load_rig
    from foresightbev.data.cam2bev_sample import sample_to_sequence
    from foresightbev.data.nuscenes import NuScenesDataset
    from foresightbev.data.prepare import prepare_nuscenes_scene
    from foresightbev.errors import ConfigurationError
    from foresightbev.perception.segmentation import SegFormerSegmenter

    settings = _settings()
    if args.source == "cam2bev":
        print(sample_to_sequence(SAMPLE, load_rig(ROOT / "configs/rig_cam2bev_1frlr.yaml"), settings.data_dir / "sequences" / "cam2bev_sample"))
        return 0
    root = Path(args.root).expanduser() if args.root else settings.nuscenes_root
    if root is None:
        raise ConfigurationError("set nuscenes_root in configs/backend.yaml, or pass --root")
    dataset = NuScenesDataset(root, settings.nuscenes_version)
    segmenter = SegFormerSegmenter(settings.segmentation_model)
    scenes = args.scene or [s.name for s in dataset.scenes()]
    for scene in scenes:
        start = time.perf_counter()
        path = prepare_nuscenes_scene(dataset, scene, settings.data_dir / "sequences" / scene, segmenter, args.scale, args.limit)
        print(f"{scene}: {path} in {time.perf_counter() - start:.1f}s", flush=True)
    return 0


def cmd_pipeline(args) -> int:
    """Replay the pipeline over a prepared sequence and record every stage output."""
    from foresightbev.config import load_pipeline_config
    from foresightbev.data.sequence import LabelSequence
    from foresightbev.recording import RunRecorder, provenance_table
    from foresightbev.replay import run_sequence

    settings = _settings()
    sequence = LabelSequence(_sequence_dir(args.sequence))
    config = load_pipeline_config(args.pipeline_config or settings.pipeline_config)
    run_id = args.name or f"replay_{args.sequence}"
    directory = runs_dir() / run_id
    if directory.exists() and not args.force:
        print(f"error: run {run_id} exists. Pass --force or choose --name.", file=sys.stderr)
        return 2
    if directory.exists():
        shutil.rmtree(directory)
    doc = sequence.doc
    table = provenance_table(doc["provenance"], f"{doc['source']}: {len(sequence)} frames, {len(sequence.calibration.cameras)} cameras.", doc.get("segmenter"), detector=config.detector)
    recorder = RunRecorder(directory, dict(kind="replay", title=run_id, scenario=args.sequence, seed=None, mode="open_loop_replay", input=doc["source"],
                                           caveat="Open-loop replay of logged frames. Decisions do not affect later frames.", provenance=table, sequence=str(sequence.directory)), config.grid.bev)
    decisions = run_sequence(sequence, config, on_frame=recorder.add)
    recorder.finish(dict(frames=len(decisions), decisions=decisions))
    print(f"{run_id}: {len(decisions)} frames, statuses {sorted({d['status'] for d in decisions})}; output {directory}")
    return 0


def cmd_eval_nuscenes(args) -> int:
    """Score prepared nuScenes scenes against annotations. Offline only."""
    from foresightbev.config import load_pipeline_config
    from foresightbev.data.nuscenes import NuScenesDataset
    from foresightbev.evaluation import integrity as I
    from foresightbev.evaluation.nuscenes_eval import evaluate_scenes
    from foresightbev.errors import ConfigurationError, JobConflict
    from foresightbev.recording import environment

    settings = _settings()
    if settings.nuscenes_root is None:
        raise ConfigurationError("set nuscenes_root in configs/backend.yaml")
    out = results_dir() / args.name
    if (out / "nuscenes_summary.json").exists() and not args.force:
        raise JobConflict(f"results already exist at {out}. Choose another --name.")
    out.mkdir(parents=True, exist_ok=True)
    scenes = args.scene or [s.name for s in NuScenesDataset(settings.nuscenes_root, settings.nuscenes_version).scenes()]
    if args.split:
        import yaml

        scenes = yaml.safe_load((ROOT / "configs/nuscenes_splits.yaml").read_text())[args.split]
    sequences = {s: _sequence_dir(s) for s in scenes}
    config = load_pipeline_config(ROOT / args.pipeline_config)
    report, decisions = evaluate_scenes(settings.nuscenes_root, settings.nuscenes_version, sequences, config)
    (out / "nuscenes_summary.json").write_text(json.dumps(report, indent=1))
    shutil.copy(ROOT / args.pipeline_config, out / "pipeline_config.yaml")
    (out / "decisions.json").write_text(json.dumps(decisions))
    (out / "environment.json").write_text(json.dumps(environment(), indent=1))
    I.append_record(data_dir(), dict(name=args.name, kind="nuscenes", scenes=scenes, split=args.split, config=args.pipeline_config, source_sha256=environment()["source_sha256"], status="completed"))
    print(json.dumps(report["pooled"], indent=1))
    return 0


def cmd_validate_nuscenes(args) -> int:
    """Validate the configured nuScenes installation and store the report."""
    from foresightbev.data.nuscenes_check import validate
    from foresightbev.errors import ConfigurationError

    settings = _settings()
    root = Path(args.root).expanduser() if args.root else settings.nuscenes_root
    if root is None:
        raise ConfigurationError("set nuscenes_root in configs/backend.yaml, or pass --root")
    report = validate(root, settings.nuscenes_version)
    out = results_dir() / "datasets"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"nuscenes_{settings.nuscenes_version}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("root", "valid", "problems", "occupancy_note")} | dict(rows=report["metadata"]["table_rows"], images=report["cameras"]["keyframe_images_per_camera"],
                     sizes=report["cameras"]["image_sizes"], skew=report["cameras"]["camera_skew_to_front_s"], gap=report["cameras"]["keyframe_gap_s"], decoded=report["images"]["decoded"]), indent=1))
    return 0 if report["valid"] else 1


def cmd_detect_streampetr(args) -> int:
    """Run pretrained StreamPETR over prepared scenes and store detections."""
    import yaml

    from foresightbev.data.nuscenes import NuScenesDataset
    from foresightbev.errors import ConfigurationError
    from foresightbev.perception.streampetr import StreamPETRInference
    from foresightbev.perception.streampetr_runner import run_scene

    settings = _settings()
    if settings.nuscenes_root is None:
        raise ConfigurationError("set nuscenes_root in configs/backend.yaml")
    dataset = NuScenesDataset(settings.nuscenes_root, settings.nuscenes_version)
    scenes = args.scene or yaml.safe_load((ROOT / "configs/nuscenes_splits.yaml").read_text())["mini_val"]
    model = StreamPETRInference(settings.data_dir / "models/streampetr/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth")
    for scene in scenes:
        print(scene, json.dumps(run_scene(dataset, scene, model, _sequence_dir(scene))), flush=True)
    return 0


def cmd_showcase(args) -> int:
    """Process a real scene end to end and store everything the application shows."""
    from foresightbev.showcase import run_showcase

    for scene in args.scene:
        print(json.dumps(run_showcase(scene, ROOT / args.pipeline_config, args.fault), indent=1, default=str), flush=True)
    return 0


def cmd_check(args) -> int:
    """Validate configuration, assets and optional dependencies without running anything."""
    from foresightbev.config import load_pipeline_config, load_settings
    from foresightbev.data.calibration import load_rig
    from foresightbev.data.validation import validate_calibration
    from foresightbev.recording import environment

    settings = load_settings()
    load_pipeline_config(settings.pipeline_config)
    rows = validate_calibration(load_rig(ROOT / "configs/rig_cam2bev_1frlr.yaml"))
    print(json.dumps(dict(settings={k: str(v) for k, v in vars(settings).items()}, calibration_cameras=[r["name"] for r in rows], real_sample=SAMPLE.exists(),
                          segmentation_model=(settings.segmentation_model / "pytorch_model.bin").exists(), nuscenes_configured=settings.nuscenes_root is not None,
                          environment=environment()), indent=1, default=str))
    return 0


def cmd_report(args) -> int:
    from foresightbev.evaluation.report import render

    out = results_dir() / args.name
    (out / "REPORT.md").write_text(render(out))
    print(out / "REPORT.md")
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("error: the backend needs FastAPI and uvicorn, which are not installed in this environment", file=sys.stderr)
        return 2
    uvicorn.run("foresightbev.api.app:app", host=args.host, port=args.port, log_config=None)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="foresightbev", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample", help="run on the real Cam2BEV sample and score Stages 2-3")
    p.add_argument("--out")
    p.set_defaults(func=cmd_sample)
    p = sub.add_parser("run", help="record one simulated closed-loop episode")
    p.add_argument("--scenario", default="occluded_pedestrian")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--mode", default="filter")
    p.add_argument("--config", default="configs/experiment_default.yaml")
    p.add_argument("--calibrator")
    p.add_argument("--name")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("eval", help="run calibration and the ablation matrix")
    p.add_argument("--config", default="configs/experiment_default.yaml")
    p.add_argument("--name", default="default")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="one seed per family, for checking the harness")
    p.add_argument("--final", action="store_true", help="refuse development seeds")
    p.add_argument("--force", action="store_true", help="allow overwriting stored results")
    p.set_defaults(func=cmd_eval)
    p = sub.add_parser("prepare", help="turn a source dataset into label sequences")
    p.add_argument("source", choices=("cam2bev", "nuscenes"))
    p.add_argument("--root", help="dataset root; overrides configs/backend.yaml")
    p.add_argument("--scene", action="append")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_prepare)
    p = sub.add_parser("pipeline", help="replay the pipeline over a prepared sequence")
    p.add_argument("--sequence", required=True)
    p.add_argument("--name")
    p.add_argument("--pipeline-config")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_pipeline)
    p = sub.add_parser("eval-nuscenes", help="score prepared nuScenes scenes against annotations")
    p.add_argument("--name", default="nuscenes_mini")
    p.add_argument("--scene", action="append")
    p.add_argument("--split", choices=("dev", "held_out", "mini_val"), help="named scene split from configs/nuscenes_splits.yaml")
    p.add_argument("--pipeline-config", default="configs/pipeline_nuscenes.yaml")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_eval_nuscenes)
    p = sub.add_parser("validate-nuscenes", help="validate the nuScenes tables, files and calibration")
    p.add_argument("--root")
    p.set_defaults(func=cmd_validate_nuscenes)
    p = sub.add_parser("detect-streampetr", help="run pretrained StreamPETR and store its detections")
    p.add_argument("--scene", action="append")
    p.set_defaults(func=cmd_detect_streampetr)
    p = sub.add_parser("showcase", help="process real scenes and store runs for the application")
    p.add_argument("--scene", action="append", required=True)
    p.add_argument("--pipeline-config", default="configs/pipeline_showcase.yaml")
    p.add_argument("--fault", choices=("front_camera_loss",), help="inject a declared camera fault into real frames")
    p.set_defaults(func=cmd_showcase)
    p = sub.add_parser("check", help="validate configuration and report available assets")
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("report", help="rebuild REPORT.md from stored results")
    p.add_argument("--name", default="default")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("serve", help="start the Stage 8 backend")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    args = parser.parse_args(argv)
    from foresightbev.errors import ForesightError

    try:
        return args.func(args)
    except ForesightError as error:
        print(f"error [{error.code}]: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
