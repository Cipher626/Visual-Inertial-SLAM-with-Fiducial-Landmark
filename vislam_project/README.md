# Marker-anchored visual–inertial SLAM (phone + laptop)

IE415 course project. An Android phone records video and IMU on a shared clock. A laptop base station then calibrates the sensors, estimates the trajectory and the ArUco marker map, and reports accuracy against a baseline that does not use the markers.

Architecture: phone–laptop hybrid with offline processing (handout §4.3 and §4.5).

**Status.** Every stage has been validated end to end on synthetic data with exact ground truth (results below). It has **not yet been run on real phone recordings**. Expect real-world errors to be larger, because real footage has motion blur, rolling shutter and plain walls.

---

## 1. What you need

| Item | Notes |
|---|---|
| Android phone with full Camera2 support | Must pass the "sync video IMU recording" check in OpenCamera Sensors (section 3) |
| Laptop (Windows, macOS or Linux) | Python 3.12 (tested); 3.11 should also work |
| Printer, tape measure (≥ 5 m), masking tape | For markers, survey and floor checkpoints |
| Optional: USB cable + Android platform-tools (`adb`) | One-click copy of recordings from the phone |

No ROS, no GTSAM and no compilation are needed.

## 2. Install (laptop)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py               # opens the base-station UI in the browser
```

Everything in the UI is also available from the command line (`python run.py -h`).

Before touching real data, check the pipeline on the synthetic room (about 2 minutes):

```bash
python run.py simulate --out data/sim_loop
python run.py process data/sim_loop --mode slam --marker-size-mm 160
```

You should get a trajectory error around 1 cm with markers and around 4–5 cm without.

## 3. Phone setup: OpenCamera Sensors

Install it from F-Droid (`com.opencamera_sensors.app`) or from the GitHub releases of `prime-slam/OpenCamera-Sensors`. Then set the following, and **keep these settings identical for calibration and all experiments**:

1. Settings → enable **Camera2 API**.
2. Settings → IMU settings… → **Enable sync video IMU recording**. If your phone cannot do synchronised timestamping, the app reports it here. That is the single most important check.
3. Settings → **Enable remote recording control**, then restart the app. This lets the laptop UI start and stop recordings over Wi-Fi.
4. Video settings:
   - resolution 1280×720 (1920×1080 also works but is slower), 30 fps
   - **video stabilisation off**, OIS off if offered
   - audio stays off (the app disables it by default because it adds frames)
5. Switch to video mode and fix the camera:
   - **lock the focus** (manual, roughly 1.5 m)
   - set a **short exposure**: 8 ms or less, raising ISO as needed
   - auto-focus "breathing" changes the intrinsics, and long exposure blurs while walking

Files land in `DCIM/OpenCamera/`:
- `VID_<date>.mp4`
- folder `<date>/` containing `VID_<date>gyro.csv`, `VID_<date>accel.csv`, `VID_<date>magnetic.csv` and `VID_<date>_imu_timestamps.csv` (the app's code writes no underscore before the sensor names, unlike its README; the importer accepts both)
- turn **off** "save frames" in the IMU settings: it only writes preview JPGs you do not need

Copy both the video and the folder to the laptop. You can use a USB cable, the **Pull newest over USB (adb)** button, or any file-sharing app.

## 4. Markers and room

```bash
python run.py targets --size-mm 150 --ids 0-9     # -> targets/markers.pdf, targets/charuco_board.pdf
```

**Printing**
- Print at **100 % / actual size**, never "fit to page".
- Measure the printed black square and enter that value as the marker size (sidebar, or `--marker-size-mm`).
- Glue the markers flat. Curled paper is a real error source.

**Placement**
- Use 6–10 markers on walls or fixed furniture, at 1.2–1.8 m height.
- Keep them away from windows that cause glare.
- Leave **one stretch of wall with no markers** for the marker-loss experiment.
- Keep the room textured (posters, furniture). Plain walls starve the visual odometry.

**Survey (ground truth)**
1. Pick a room frame: origin at a floor corner, x along one wall, y along the other, z up.
2. Tape-measure each marker centre (x, y, z) and note which way it faces.
3. Copy `templates/markers_surveyed.json` to `room/markers_surveyed.json` and fill it in. `normal` is the direction from the wall into the room, for example `[-1, 0, 0]` for a marker on the wall at x = max.
4. Record marker size, height and orientation in your report (the handout requires this).

## 5. Calibration (once per phone and settings)

Tape the ChArUco board flat, then record a 40–60 s video:

1. Hold still for 2 s.
2. Move slowly so the board fills different parts of the image and is seen at different tilts (this gives the intrinsics).
3. Then rotate the phone briskly about all three axes while keeping the board in view (this gives the camera–IMU calibration).

Import it as experiment "calibration board", then in the **Calibrate** tab:

| Step | Output | Accept if |
|---|---|---|
| Calibrate camera | `calib/camera.json` (K, distortion) | RMS < 1 px (ideally < 0.5) |
| Calibrate camera–IMU | `calib/camera_imu.json` (time offset `td`, rotation `R_CI`) | correlation > 0.9, residual < ~0.2 rad/s, a single sharp peak in the td curve |

CLI equivalent:
`python run.py calib-camera data/calib --square-mm <measured>` then `python run.py calib-imu data/calib --square-mm <measured>`

If the camera–IMU correlation is below 0.9, run **Diagnose camera–IMU sync** (or `python run.py sync-diag data/calib --square-mm <measured>`). It estimates the offset in 6 s windows and names the cause:
- an offset that jumps between windows means frames and timestamps are mis-paired part-way through
- a stable offset of several frames means the video started late, so the offset must be estimated per recording
- low correlation in every window means the visual rotation is too noisy (blur, board too small)

On synthetic data a healthy result reads about 0.999 correlation and 0.07 rad/s residual.

## 6. Recording protocol (every experiment)

1. Laptop UI → Record → **Connect** to the phone IP → **Start recording**.
2. **Hold still for 2 s facing a marker** that appears at least ~40 px wide. This gives the gyro bias, gravity direction and bootstrap.
3. **Side-step about 20 cm slowly, without turning the phone**, keeping that marker in view. Monocular vision needs this parallax, and the marker supplies the metric scale.
4. Walk the path at normal pace, keeping the phone at chest height. Look left and right at corners: these turns are what each recording uses to measure its own camera–IMU time offset.
5. For closed loops, return to the exact start spot (tape mark) and **hold still for 2 s**.
6. **Stop recording**, copy the files, then **Import** (choose the experiment type and add notes).

**Optional checkpoints** for position ground truth along the path:
- Tape marks on the floor, with the phone held over each one.
- Stand still for about 2 s at each, in order.
- Put a `checkpoints.csv` (columns `id,x,y` in the room frame; template in `templates/`) into that dataset's folder.
- The i-th still period is matched to the i-th row.

## 7. The required experiments (handout §8)

| Experiment | How to record | Process with | Report |
|---|---|---|---|
| 1. Simple path | L or rectangle, checkpoints at the corners | `slam` and `anchor` | checkpoint XY error, marker map error |
| 2. Marker-loss path | Walk past the marker-free wall for ≥ 5 s (measure the duration from the UI timeline) | `slam` | error growth during the gap (Results → diagnostics), baseline vs markers |
| 3. Closed loop | Start and end on the same tape mark, re-observe the first markers | `slam` | start-to-end drift, drift per metre, loop-closure residual, marker re-observation spread |
| 4. Repeat run | New recording in the same room, starting elsewhere | `reloc` with `runs/<loop run>/slam/map.json` (or the anchor map) | time to first fix, checkpoint error, marker map consistency |
| 5. Difficult condition | Pick one: dim light, fast motion, oblique views, occlusion, or remove half the markers | `slam` | same metrics plus the visual-tracking % and detection rate |

The **baseline** (VO + gyro, with no marker constraints after the bootstrap) is computed automatically in every run and shown next to the marker-constrained result. Repeat each experiment 2–3 times for repeatability; `python run.py compare --csv results.csv` tabulates all runs.

To stress-test the pipeline before field work:
`python tools/degrade.py data/sim_loop data/sim_dark --gain 0.35 --noise 6`
(also accepts `--blackout T0 T1` and `--blur N`)

## 8. Method

**Frames and conventions.** `T_AB` maps points from frame B to frame A.
- Camera C is OpenCV (x right, y down, z forward).
- Phone IMU I is the Android sensor frame.
- World W is gravity-aligned with z up.
- Marker M is the ArUco frame (x right, y up, z out of the marker).

**State.** Camera pose `T_WC,k` at every processed frame (10–15 Hz), marker poses `T_WM,i`, and one log-scale per visual-odometry segment after a re-initialisation. The gyro bias is estimated from the initial still period; the accelerometer bias is not estimated. This is a deliberate simplification of the handout's full state `[p v q b_a b_w]`. Velocity is not optimised; it is used only in the front-end motion model.

**Front-end** (`vislam/frontend.py`)
- KLT feature tracks with a forward–backward check.
- Pose from PnP-RANSAC on triangulated points. The gyro-predicted rotation `R_CI ΔR_gyro R_CI^T` is used as the initial guess and as a 8° consistency gate.
- **Bootstrap**: the first frame's pose is `T_WC0` (gravity from the accelerometer). During the side-step, poses come from `W_T_C = W_T_C0 · C0_T_M · (Ck_T_M)^-1`, which fixes the metric scale.
- When vision fails: rotation from the gyro, position from a decaying constant-velocity model (or accelerometer strapdown, selectable).
- Recovery from loss: essential-matrix re-initialisation with the gyro consistency check. The scale is guessed from the previous walking speed and is corrected later by the back-end.

**Markers** (`vislam/markers.py`)
- `ArucoDetector` with sub-pixel corners.
- `C_T_M` from `solvePnP(IPPE_SQUARE)`.
- Gates on size, reprojection error and range.
- Only IDs you placed are accepted, and an ID seen fewer than 5 times is discarded. Both protect against false detections; OpenCV 4.10 produced a phantom ID from wall texture in testing.

**Back-end** (`vislam/backend.py`) minimises the handout's cost:

```
Σ_k ‖r_odo(X_k, X_k+1, s_seg)‖²_Σ  +  Σ_k ‖r_gyro(X_k, X_k+1)‖²  +  Σ_(k,i) ρ_Huber(‖r_corner(X_k, L_i, z_ki)‖²_Λ)  +  priors
```

- `r_corner` reprojects the four marker corners (known size) and compares them with the detected pixels. This uses the raw measurement and avoids the planar-pose (IPPE) orientation ambiguity of using `C_T_M` directly.
- Markers are initialised with `W_T_M = W_T_C · C_T_M`.
- Solver: Levenberg–Marquardt on the sparse normal equations, with Huber weights by IRLS and a grouped finite-difference Jacobian. Two rounds; observations above 6 px after round 1 are rejected.
- Re-observing a marker after a loop is what closes the loop.
- Modes:
  - `slam`: markers are unknown landmarks; the first pose is fixed.
  - `anchor`: tape-measured marker poses are priors, so the result is in the room frame (`W_T_C = W_T_M (C_T_M)^-1` initialises the run).
  - `reloc`: a saved map is used as priors.

**Synchronisation.** `td` (the frame at camera time t was captured at IMU time t + td) is chosen to maximise the correlation of the visual and gyro rotation-rate magnitudes. `R_CI` is then the Kabsch fit of the angular-velocity vectors.
- The board calibration supplies `R_CI`, which is fixed by how the camera and IMU are mounted.
- `td` is **measured again on every recording**: after a first front-end pass, the pipeline compares visual-odometry rotation with the gyro. If the result is reliable (correlation ≥ 0.85) and differs from the value in use by more than 10 ms, the front-end runs again with it. On a real phone the offset varied between recordings, apparently because the video can start a few frames after the timestamp file (issue log ISS-04).
- `config.json → sync` controls this (`per_recording_td`, `min_corr`, `rerun_if_diff_ms`).
- The offset is measured from rotation while visual tracking runs, which starts only after the bootstrap side-step. Turns during the walk provide the signal; rotations before the side-step do not help (an earlier version of this README asked for wrist twists there; they only delayed the start).

## 9. Requirement checklist (handout §6)

| # | Requirement | Where |
|---|---|---|
| 1 | Moving camera sequence | OpenCamera Sensors video → `dataset.py` |
| 2 | IMU acquisition | `_gyro.csv`, `_accel.csv` → `dataset.py`, `imu.py` |
| 3 | Timestamping and sync | shared clock from the app, `td` estimation (`calibration.py`), per-run sync check, Sensors tab checks (rates, gaps, frame-count match, coverage) |
| 4 | Camera calibration | ChArUco → `calibrate_camera_charuco` |
| 5 | Detect and identify ArUco | `markers.py` |
| 6 | Marker pose relative to camera | `IPPE_SQUARE` PnP → `C_T_M` |
| 7 | Continuous trajectory | `frontend.py` (VO + gyro + inertial fallback), `backend.py` |
| 8 | Build or load a marker map | `runs/.../map.json` (built); `--map` / `room/markers_surveyed.json` (loaded) |
| 9 | Trajectory shown relative to the map | Results tab, top-down plot |
| 10 | Repeated observations used | corner factors for loop closure, consistency metrics, `reloc` mode |
| 11 | Baseline comparison | `trajectory_baseline.csv` vs `trajectory_optimized.csv` in every run |
| 12 | Quantitative evaluation | `metrics.json`: drift, drift per metre, loop residual, marker position and normal error, checkpoint error, detection rate, relocalisation time, runtime, sync check |

**User-facing build (§7)**
- Start/stop recording: Record tab, via Wi-Fi remote control.
- Camera and sensor status: fps and exposure on start; rates, gaps and still periods in the Sensors tab.
- Save and load datasets: Import, plus the sidebar dataset list.
- Processing: Process tab.
- Detected markers: frame preview with axes, and the marker table.
- Trajectory and map: Results tab.
- Diagnostics: tracking status over time, detections per marker, timings, optimiser stats.

## 10. What is reused vs written (handout §13)

**Reused**
- OpenCamera Sensors (data logging).
- OpenCV: ArUco detection, `solvePnP` / IPPE, KLT, `goodFeaturesToTrack`, `findEssentialMat`, `triangulatePoints`, `calibrateCamera`.
- SciPy: rotations, sparse linear solve.
- Streamlit and Plotly (UI).

**Written for this project**
- Data import and synchronisation analysis.
- Gyro integration.
- Camera–IMU time-offset and rotation estimation.
- The VO front-end logic (bootstrap, gyro gating, keyframes, inertial fallback, re-initialisation).
- The factor graph and LM solver (corner reprojection, gyro and scale factors).
- Relocalisation mode, all metrics, the synthetic room simulator and the UI.

ARCore is not used.

## 11. Synthetic validation (reproduce with `run.py simulate` + `run.py process`)

Synthetic room: 6 × 5 m, 7 markers of 16 cm, 640×480 at 30 fps. Simulated phone IMU at 200 Hz with bias and noise, a 15 ms clock offset and a perturbed `R_CI`. Results are from OpenCV 5.0.

| Run | Trajectory RMSE (baseline → with markers) | Marker map RMSE | Loop-closure residual |
|---|---|---|---|
| Closed loop, `slam` | 4.4 → 0.9 cm | 4.0 → 0.5 cm | 12.1 → 0.2 cm |
| Closed loop, `anchor` (room frame, no alignment) | 18.0 → 1.5 cm | 21.0 → 0.3 cm | 40.2 → 0.2 cm |
| Closed loop with a 2 s camera blackout | 41.6 → 2.0 cm | 25.8 → 0.9 cm | 97.7 → 0.2 cm |
| Repeat run, `reloc` in the `slam` map | 12.6 → 2.4 cm | 9.0 → 0.4 cm | – |
| Repeat run, `reloc` in the room-frame map | 34.1 → 4.1 cm | 31.6 → 0.7 cm | – |
| Closed loop with frames paired 4 frames late (mimics the real phone) | 6.9 → 1.5 cm | 0.8 cm with markers (2.2 cm without per-recording offset) | – |

The baseline is sensitive to small input changes: a 1 ms change in time offset or a different OpenCV build moves it by about 2 cm (6.1 cm was measured for the first row with the offset fixed at 15 ms). The with-marker numbers stay around 1 cm.

Calibration checks:
- ChArUco intrinsics within about 1.2 px of truth (0.14 px RMS).
- Board-based camera–IMU calibration: `td` 19.9 ms vs 22.0 ms true, `R_CI` error 0.04°.

Speed was about 30 fps on a laptop-class CPU and the back-end took 1–5 s. On this test, OpenCV 4.10 gives similar accuracy once the ID filter is on.

The synthetic room is easy: sharp texture, no blur, no rolling shutter, a global shutter and perfect markers. Use it to check the pipeline works, **not** as your results.

## 12. Known limitations (state these in your report)

- No pose exists before the first marker fix: the bootstrap needs a marker for metric scale. Relocalisation time is therefore "time until a known marker is first seen".
- No local bundle adjustment. Map points are triangulated from two views and never refined, so plain walls and fast turns will drop tracking.
- After a tracking loss the scale is guessed from walking speed. Without a marker afterwards, that segment's scale stays wrong.
- Constant-velocity propagation through a long gap drifts quickly. Accelerometer propagation with an un-estimated bias drifts quadratically.
- Rolling shutter and the camera–IMU lever arm (about 1 cm) are not modelled.
- The default `R_CI` (used if you skip camera–IMU calibration) assumes a back camera with 90° sensor orientation and raw, unrotated frames. Calibrate instead of trusting it.
- The timestamp file and the video can disagree by a few frames (the encoder starts late or skips frames). The importer matches them using the video's own frame times; dropped frames leave gaps that make the match exact. With no drops at all the match is ambiguous: the importer then pairs from the first frame and warns, and the per-run sync check catches any resulting shift (a multiple of 33 ms).

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| "Bootstrap never succeeded: …" | The message names the cause (no marker / wrong dictionary or IDs / too dark / marker too small / never moved 10 cm with the marker in view / too little texture) and writes `bootstrap_diagnostics.json` into the run folder |
| A processed run is missing from Results | Fixed: run names with `/` used to create an extra folder level. Older runs saved that way are now listed too |
| "marker map is N % larger/smaller than the tape survey" | The marker size entered does not match the printed size (it sets the metric scale), or the survey has an error |
| "video frames are WxH but camera.json was calibrated at …" | Calibration and experiments must use the same video resolution and orientation |
| Need help debugging a recording | `python run.py share data/<name>` (or the Sensors-tab button) writes a zip of the first 20 s at half resolution plus calibration, usually a few MB |
| Visual tracking % low | Add texture, shorten exposure, slow down, use `stride` 1 |
| Detection rate low | Bigger markers, more light, a lower `min_side_px`; check the dictionary and the ID list in the sidebar |
| td curve flat or correlation < 0.8 | Timestamps are not on one clock (the device lacks support), or the calibration video has too little rotation |
| Connect fails | Same Wi-Fi (a phone hotspot works), remote control enabled, app restarted and in video mode |
| Results in the wrong place in the room | `slam` mode is in its own frame; tick "Show in room frame" (needs `room/markers_surveyed.json`) or use `anchor` mode |

## 14. Layout

```
app.py              Streamlit base station (Record / Sensors / Calibrate / Process / Results)
run.py              CLI with the same functions
config.json         all thresholds and noise parameters
vislam/             dataset, imu, markers, calibration, frontend, backend, pipeline, metrics, phone_remote, geometry
tools/              make_targets.py, simulate.py, degrade.py
templates/          markers_surveyed.json, checkpoints.csv
data/  calib/  room/  runs/   created at run time
```
