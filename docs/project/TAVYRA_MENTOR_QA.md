# Tavyra mentor questions and answers

Figures come from `docs/project/TAVYRA_FINAL_RELEASE_REPORT.md`. Real figures are from mini_val: 81 frames, two previously
seen daytime scenes, preliminary.

## Scope and motivation

**1. What does Tavyra do?**
It takes six recorded camera views and produces a bird's-eye occupancy map, 3D detections, tracks, a 3-second motion
forecast, risk scores for candidate plans, and an explained pass, modify or fallback decision. On real data it runs as
open-loop replay. It is a research prototype and controls nothing.

**2. Why camera-centric perception?**
Cameras are cheap, dense and carry semantics, so they are a strong base for mass-market systems. Their weakness is depth,
and that weakness makes the free-versus-unknown question central, which is the part this project studies. It does not claim
cameras are better than LiDAR.

**3. Why bird's-eye view?**
Planning happens on the ground plane, and BEV puts all cameras, objects and paths in one metric frame. Distance and
overlap are directly computable. The cost is that height is reduced, and projection depends on a ground assumption.

**4. What is the research question?**
Whether uncertainty in future camera-derived occupancy can become calibrated trajectory-event probabilities that improve
decisions without excessive conservatism. It is a candidate question. Tavyra built the chain to test it, but did not answer
it on real driving.

**5. What is your contribution, and how does it differ from prior work?**
An integrated, causal, inspectable chain that keeps observed-free, unknown and no-coverage separate through to trajectory
checks, uses joint forecast samples in an explainable filter, and openly tests interventions against annotations. Its
parts are established. My comparison against OccDriver, SURE-Plan, Drive-OccWorld and PORA is incomplete, so I call this an
engineering and evaluation contribution, not proven novelty.

## Geometry and occupancy

**6. How are intrinsics and extrinsics used?**
Extrinsics move a point from ego to camera: X = Rᵀ(p − t). Intrinsics project it: u = fx·X/Z + cx, v = fy·Y/Z + cy. The point
counts only if Z > 0 and the pixel is inside the image. These are precomputed for each ground cell and voxel. When images
are scaled to 800×450, intrinsics are scaled too.

**7. How is 3D occupancy constructed?**
By semantic space carving, not a neural occupancy network. Each camera's labels are read along rays. Road or sky means the
ray passed through, so voxels are free. An obstacle gives a contact range from the ground: free before it, occupied in a
0.6 m band at it, and unknown behind it. Cameras are fused, and columns up to 2 m collapse into a BEV state.

**8. Where does depth come from?**
Two places. Occupancy uses ground geometry: where an object's base meets the ground, under a planar assumption. StreamPETR
learns depth implicitly from image features, camera geometry and temporal memory. There is no depth sensor and no
dedicated depth network.

**9. How is unknown different from free?**
Free needs positive evidence that a ray passed through. Unknown means a camera covers the place but offers no such evidence,
such as behind a car. No coverage means no healthy camera sees it. A wrong free is the dangerous error, so missing data
never becomes free.

**10. What happens when a camera fails?**
It gives no evidence, so space only it saw falls to unknown or no coverage. The filter records `CAMERA_COVERAGE_LOST`, and the
showcase configuration forbids speeding up during the loss. I tested this by blanking the front camera for frames 14–25 of
scene-0103, a declared artificial fault.

**11. What assumptions does BEV projection make?**
That visible cells lie on the ground, and that calibration and ego pose are right. Objects above the ground smear away from the
camera. Slopes, kerbs and pitch create range error. A plane fitted to detected object bottoms reduces this, but it is only a
coarse tilt and offset.

**12. What was the false-free problem and how did you reduce it?**
About 30% of annotated dynamic-box cells were marked free. Flat ground was .550, height tolerance reached .303, and pose tilt
did not help (.312). A local plane gave .199, and adding model boxes in a separate layer gave .066. Free area fell from 1240 to
1113 m² per frame, so the gain has a cost. It is a proxy on dynamic objects, not voxel accuracy, and that layer now depends on
the detector.

## LiDAR and leakage

**13. What is the role of LiDAR?**
No LiDAR point enters inference. StreamPETR's output frame is `LIDAR_TOP`, so that sensor's fixed extrinsic is used as a
transform. A subtler dependency is that nuScenes ego poses came from localisation that used LiDAR with a map. So perception
reads cameras, but recorded ego pose is not camera-derived. Annotations were also made with LiDAR support, and those are used
only for scoring.

**14. How do you avoid LiDAR leakage?**
The inference table list excludes annotations and LiDAR data, a test parses the inference modules and fails if they import the annotation classes, and only calibration metadata
of the LiDAR sensor is used. Evaluation code loads annotations separately. The filter sees nothing from the future, and
forecast records confirm the last input is not later than the reference time.

**15. How do you handle train/test leakage?**
Classical tuning used only dev scenes, and held-out scenes were scored once. StreamPETR's checkpoint is reported on
official mini_val only, whose scenes belong to the official val split, assuming the authors trained on the official train
split. mini_train figures are labelled inflated. Tracker variants were picked on mini_train and scored once on mini_val.
Limitation: I had already seen both mini_val scenes, so results are labelled preliminary. No untouched real test set exists.

## Detection and tracking

**16. How are objects detected?**
StreamPETR R50 with 428 queries. 128 are propagated from a temporal memory, so it remembers objects across frames. It attends to
image features with 3D position embeddings and outputs boxes, velocity, scores and ten classes. Tavyra maps eight classes to
vehicle and person. Barriers and cones are not used. The baseline detectors stay available.

**17. How did you verify the port?**
It is a plain PyTorch inference port. 573 needed tensors load strictly with restricted loading. Verification used controls:
recall .877 with memory, .803 without memory, and .045 with shuffled cameras, which shows the geometry path matters.

**18. Are reported metrics measured or copied?**
Measured here. Published figures appear only in the assessment document as published. Mine use a 2 m match within 40 m, not
nuScenes mAP or NDS, so they are not comparable with paper numbers.

**19. How does tracking work?**
A constant-velocity Kalman filter in the world frame with state [x, y, vx, vy]. Association is Hungarian on Mahalanobis
distance with a 99% gate, same class only. Tracks confirm after 3 vehicle hits or 2 person hits, and die after more than 2
misses.

**20. Why use the model's velocity?**
At 2 Hz, position differences give poor velocity. Feeding StreamPETR's velocity reduced error from .50 to .197 m/s on mini_val
and ID switches from 9.8 to 7.7 per 100 matched. Recall dropped slightly from .797 to .786. NEES of 3.4 against the ideal 4
suggests a sensible filter on this small sample.

**21. What is NEES?**
Normalised estimation error squared: (x − x̂)ᵀ P⁻¹ (x − x̂). For a consistent filter it averages the state dimension, here 4.
Much higher means overconfident, much lower means underconfident.

## Forecasting

**22. How does forecasting work?**
Agents are sampled from the Kalman posterior 32 times and rolled out at constant velocity with small acceleration noise, to 3 s
at 0.25 s steps. Stationary agents stay still and static occupied cells persist.

**23. Why is this baseline appropriate?**
Constant velocity is hard to beat at short range and easy to verify. Its joint samples let risk ask about one coherent future. A
learned model would have to beat it on the same protocol. On position error it beat frozen tracks: .78 and 1.66 m against .94 and
1.86 m at 1 and 3 s.

**24. Is this 4D occupancy forecasting?**
No. It is tracking extrapolation rendered on a grid. A U-Net was trained, but only on simulator data. On real occupancy its IoU
is .154 at 1 s and .116 at 3 s, against .362 and .297 for constant velocity, so it is excluded from the risk path.

**25. Why did frozen tracks score slightly better on occupancy IoU?**
It scored .378 and .317. Most agents in these scenes are parked, likely favouring stillness. Position error, measured only on tracked agents,
favours constant velocity. Two metrics measure different things, and two scenes cannot settle it.

## Risk and safety

**26. How is trajectory risk calculated?**
For each candidate, sweep the ego footprint against each forecast sample. The event score is the share of samples with an overlap
within 3 s. Touching static occupied cells sets it to 1 from then on. It also computes unknown exposure, stopping margin, clearance
and TTC under the mean model.

**27. Is that score a collision probability?**
No. It is a frequency within an uncalibrated model. Isotonic calibration exists only on simulator data, so it does not apply to
real driving.

**28. What does the safety filter guarantee?**
Only that the chosen plan satisfied its own checks given its inputs, or that it declared fallback, with a recorded reason. It does
not guarantee collision freedom, because perception can miss objects and agents can act differently than forecast. There is no
formal proof, no controller model, and no safety case.

**29. How do you interpret the real-log intervention results?**
Of 81 frames, the nominal plan conflicted with annotated objects 3 times. There were 8 interventions: 2 with a conflict, 6
without. One conflict passed. Replay is open loop, so no one reacted to those plans. Three events cannot support a rate.
So the filter is not validated on real driving.

**30. Does treating unknown space conservatively help?**
No benefit is measured. On real logs, unknown-as-free removed 2 interventions and changed no conflict outcome. In simulation,
collision rate stayed at 1.1%. It remains on as a design position, not as a shown improvement.

**31. Give an example of a decision.**
scene-0916, frame 21 in the app: the nominal plan has event score .69 and zero clearance to tracked vehicle 114 within 2.5 s, above the 0.10 limit.
The filter replaces it with holding the lane at 70% of desired speed, score .03. In scene-0103, frame 37, nothing is
admissible, so it declares fallback with emergency stop.

**32. How were thresholds chosen?**
The 0.10 limit was set before mini_val evaluation and is identical across real configurations. I did not tune it on evaluation
results. `hold_speed_on_coverage_loss` came from reviewing a fault's logic, and the simulator configuration keeps it off.

## Systems

**33. Why ROS 2?**
It gives realistic process and message boundaries and a route to robotics stacks. I proved determinism: replay of scene-0916 over
ROS 2 matched in-process decisions and occupancy hashes. A stalled camera triggered the watchdog and the run continued. Real-time
performance is not shown: real-frame replay took about 3.4 s per frame.

**34. Why FastAPI and React?**
FastAPI gives typed contracts, validation and automatic docs for Python outputs. React gives interactive inspection of frames and
evaluation. Both sit outside the decision path, because a dashboard must never affect safety logic.

**35. How do you know the frontend shows real data?**
Pages read `/api/v1` and contain no metrics of their own. A release audit checked every stored run through the API, and a browser
sweep checked each page. Unavailable data shows as unavailable.

## Evaluation and limits

**36. What are the limitations of nuScenes mini?**
Ten scenes, 404 keyframes at 2 Hz, no occupancy labels, a small mini_val with no night scene, and few conflicts. It verifies a
pipeline; it cannot support generalisation.

**37. Replay versus closed-loop simulation?**
In replay, the logged car always drives the same way, so decisions cannot change later frames. In closed loop, the decision moves the
vehicle and the scene responds. Tavyra's closed loop exists only in its simple simulator, with scripted agents that do not react.
Its 1.1% against 23.3% collision comparison is not evidence about real roads.

**38. What are your measured results?**
On mini_val: vehicles .902 recall and .806 precision, people .831 and .734, track recall .786, velocity error .197 m/s, false-free
proxy .066, 73 pass, 5 modify, 3 fallback. Human-path alarms 1 of 68, with no true contact. The pipeline took about 450 ms with cached
perception, and live StreamPETR about 290 ms on MPS.

**39. What failed, and what did you change?**
The classical detector was too weak on real footage, so StreamPETR was integrated. Flat-ground occupancy marked too many objects free, so
I added plane fitting and model boxes. Two-stage association tripled false tracks and was rejected. The U-Net did not transfer and was
excluded. A camera-fault review found the plan could speed up during camera loss, so I added a hold. My timestamp hypothesis for an
earlier error was tested and rejected. A scorer bug was found through a reproducibility failure and fixed for all detectors.

**40. Why is this not deployable?**
It has no live input, no certified perception, no vehicle control, no map or route, unvalidated risk scores, inadequate latency evidence,
no redundancy, and no safety case to ISO 26262 or SOTIF standards. Validation data are tiny.

**41. What are the licence constraints?**
nuScenes is non-commercial. No separate StreamPETR weights licence was found, so it is unverified and treated as non-commercial. The
checkpoint and dataset are not redistributed in the repository.

**42. What comes next?**
Full nuScenes validation with Occ3D labels, confirming the weights licence, running CARLA, adding metric depth or better ground
estimation, a learned forecaster that outputs joint samples, and calibrating risk on real data.
