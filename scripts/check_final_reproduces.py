import json,sys
from pathlib import Path
import cv2; cv2.setNumThreads(1)
from foresightbev.evaluation.experiments import _episode_job, load_experiment
root=Path('.').resolve()
cfg=json.load(open('experiments/end_to_end/results/final/experiment.json'))
stored=json.load(open('experiments/end_to_end/results/final/episodes.json'))
ok=True
for mode,scen,seed in (('filter','occluded_pedestrian',401),('idm_no_filter','cut_in',403),('filter_calibrated','dense_traffic',402),('filter_front_dropout','lead_brake',404)):
    s=next(e for e in stored if (e['mode'],e['scenario'],e['seed'])==(mode,scen,seed))
    n=_episode_job((cfg,str(root),scen,seed,mode,str(root/'experiments/end_to_end/results/final/risk_calibrator.json')))
    keys=[k for k in s if k not in('pipeline_ms','occupancy_stats','tracking','forecast')]
    d=[k for k in keys if s[k]!=n[k]]
    print(mode,scen,seed,'identical' if not d else ('DIFF',{k:(s[k],n[k]) for k in d})); ok&=not d
print('ALL IDENTICAL' if ok else 'MISMATCH')
