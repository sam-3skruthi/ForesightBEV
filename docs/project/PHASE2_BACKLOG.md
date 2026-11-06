# Phase 2 backlog

| Priority | Work | Why | Depends on | Acceptance |
|---|---|---|---|---|
| P0 | Validate StreamPETR on the full nuScenes val split, including night | Two mini_val scenes are preliminary | Full dataset, storage | Metrics on official val, night reported separately |
| P0 | Confirm or replace StreamPETR weights licence | Unverified, treated as non-commercial | Author or legal answer | Written status recorded |
| P0 | Better range: metric depth or ground estimation | Flat ground causes most real range error | Approved model or method | Lower false-free rate on held-out scenes |
| P1 | Reduce ID switches, use model velocity in the tracker | 9 per 100 matched remains high | None | Lower switches on the same protocol |
| P0 | Full nuScenes, with Occ3D labels if licensed | Ten scenes verify only; static occupancy is unscored | Storage, licence | Occ3D protocol metrics on its official split |
| P0 | Run the CARLA adapter | Never executed | Linux host with a GPU | One closed-loop run; scenario suite repeated with results stored |
| P1 | Reactive agents in the simulator | Current agents ignore the ego | None | Filter ablations repeated with reactive agents |
| P1 | Controller and vehicle model | Plans are followed perfectly | CARLA or dynamics model | Tracking error reported; safety margins re-evaluated |
| P1 | Pretrained occupancy baseline | Current occupancy is observational | CUDA machine, checkpoints | Reproduce a published metric on its own protocol before comparing |
| P1 | Learned forecast with joint samples | U-Net marginals cannot feed Stage 6 | Scene-level sampling model | Beats CV on held-out data; event calibration measured |
| P1 | OOD and uncertainty per OCCUQ, ReliOcc | No such output exists | Trained feature model | AUROC and ECE reported separately, under degradation |
| P1 | Use 12 Hz sweeps | 2 Hz keyframes limit velocity estimates | None | Lower velocity error on held-out scenes |
| P1 | Scoring in `filter` on real data | Thresholds were set for simulation | More real data | Alarm rate on human driving falls without missed events |
| P2 | Autoware bridge | Mapping note only | Ubuntu with ROS 2 and Autoware | Decision node runs inside an Autoware launch |
| P2 | REA 05 and planning-risk comparison | Stage 0 gate is open | Reading time | Direct comparison with OccDriver, SURE-Plan and Drive-OccWorld |
| P2 | Formal analysis of the filter | Empirical rules only | Explicit assumptions | Written proof obligations, or a stated decision not to claim them |
