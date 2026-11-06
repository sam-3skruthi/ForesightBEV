import { useState } from "react";
import { layer } from "../api";
import { Lines } from "../charts";
import { useSelection } from "../context";
import { useApi } from "../hooks";
import { Scene } from "../Scene";
import type { Json } from "../types";
import { Loading, Panel, Segmented, Stat, Tag, Unavailable, fmt } from "../ui";

export function Future() {
  const { run, frame, record, detail, recordError } = useSelection();
  const evaluation = useApi<Json>("/evaluation");
  const [horizon, setHorizon] = useState<"1" | "2" | "3">("3");
  if (recordError) return <Unavailable what="Frame unavailable" why={recordError} />;
  if (!record || !detail || !run) return <Loading what="frame" />;
  const showcase = evaluation.data?.real?.showcase, forecast = evaluation.data?.real?.forecast, sim = evaluation.data?.synthetic?.forecaster;
  const moving = record.forecast_paths.filter((p) => !p.stationary);
  return (
    <div className="split-layout">
      <section className="stage">
        <div className="toolbar"><Segmented label="Forecast horizon" value={horizon} onChange={setHorizon} options={[["1", "+1 s"], ["2", "+2 s"], ["3", "+3 s"]] as const} /></div>
        <Scene run={run} frame={frame} record={record} grid={detail.grid} overlays={{ tracks: true, trails: true, forecast: Number(horizon) }} />
        <div className="state-key"><span><i className="k-trail" />Observed path</span><span><i className="k-future" />Predicted path</span><span><i className="k-spread" />Model spread (1σ)</span></div>
        <figure className="labelled"><figcaption>Forecast occupancy at +1, +2, +3 s. Grey is unobserved.</figcaption><img className="forecast-strip" src={layer(run, frame, "forecast")} alt="Forecast occupancy at three horizons" /></figure>
      </section>
      <div className="side">
        <Panel title="Prediction method" note="This is tracking extrapolation, not a trained 4D occupancy network. Circles show the Kalman spread propagated by constant velocity. It describes the model, is uncalibrated, and is not a confidence interval.">
          <ul className="kv">
            <li><span>Method</span><b>Constant velocity</b></li><li><span>Provenance</span><b><Tag kind="classical">{record.forecast.provenance}</Tag></b></li>
            <li><span>Joint samples</span><b>{record.forecast.n_samples}</b></li><li><span>Horizon</span><b>{fmt(record.forecast.horizons.at(-1), 1)} s</b></li>
            <li><span>Reference time</span><b>t = {fmt(record.forecast.t_ref, 2)}</b></li><li><span>Last input</span><b>t = {fmt(record.forecast.last_input, 2)}, causal</b></li>
          </ul>
        </Panel>
        <div className="stats"><Stat label="Predicted agents" value={record.forecast_paths.length} /><Stat label="Moving" value={moving.length} /><Stat label="Held still" value={record.forecast_paths.length - moving.length} /></div>
        {showcase ? <Panel title="Position error on real tracks" note={`mini_val, ${showcase.pooled.forecast.pairs[1]} pairs at 1 s and ${showcase.pooled.forecast.pairs[5]} at 3 s. Keyframe gaps vary from 0.4 to 0.6 s.`}>
          <Lines label="Forecast position error by horizon" x={showcase.pooled.forecast.horizons_s} xLabel="Horizon (s)" yLabel="Error (m)" series={[{ name: "Constant velocity", tone: 1, values: showcase.pooled.forecast.constant_velocity_error_m }, { name: "Frozen", tone: 2, dash: true, values: showcase.pooled.forecast.frozen_error_m }]} />
        </Panel> : <Unavailable what="Real forecast metrics not stored" why="Run the mini_val evaluation." />}
        {forecast ? <Panel title="Methods on real occupancy" note={`${forecast.frames} frames at ${forecast.grid_m} m cells. The U-Net was trained only on simulator data and is not used for risk. It does not transfer.`}>
          <table><thead><tr><th>Method</th><th>Kind</th><th className="n">IoU 1 s</th><th className="n">IoU 3 s</th></tr></thead><tbody>
            <tr><td>Constant velocity</td><td><Tag kind="classical">classical</Tag></td><td className="n">{fmt(forecast.models.constant_velocity_monte_carlo.iou[1], 3)}</td><td className="n">{fmt(forecast.models.constant_velocity_monte_carlo.iou[5], 3)}</td></tr>
            <tr><td>Frozen tracks</td><td><Tag kind="classical">baseline</Tag></td><td className="n">{fmt(forecast.models.frozen_tracks.iou[1], 3)}</td><td className="n">{fmt(forecast.models.frozen_tracks.iou[5], 3)}</td></tr>
            <tr><td>U-Net</td><td><Tag kind="trained">simulator-trained</Tag></td><td className="n">{fmt(forecast.models.simulator_trained_unet.iou[1], 3)}</td><td className="n">{fmt(forecast.models.simulator_trained_unet.iou[5], 3)}</td></tr>
          </tbody></table>
        </Panel> : <Unavailable what="Real occupancy forecast not stored" why="Run foresightbev.evaluation.forecast_real." />}
        {sim && <Panel title="Same U-Net on simulator data" note="Synthetic data. Shown only to explain the gap above."><ul className="kv"><li><span>IoU at 1 s, 3 s</span><b>{fmt(sim.models.learned_unet.iou[1], 3)}, {fmt(sim.models.learned_unet.iou[5], 3)}</b></li></ul></Panel>}
      </div>
    </div>
  );
}
