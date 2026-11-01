"""Laptop base station UI.   Run:  streamlit run app.py
Record (phone remote control + import) -> Sensors (status, sync) -> Calibrate -> Process -> Results."""
import json
import re
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vislam import pipeline  # noqa: E402
from vislam.calibration import (calibrate_camera_charuco, board_orientations, estimate_cam_imu,  # noqa: E402
                                save_json, load_json)
from vislam.dataset import load_dataset, dataset_report, ingest_opencamera, list_datasets, read_frame  # noqa: E402
from vislam.imu import ImuData  # noqa: E402
from vislam.markers import MarkerDetector, draw_detections, get_dictionary  # noqa: E402
from vislam.phone_remote import OpenCameraRemote, adb_available, adb_pull_latest  # noqa: E402

DATA, CALIB, RUNS, ROOM = ROOT / "data", ROOT / "calib", ROOT / "runs", ROOT / "room"
for d in (DATA, CALIB, RUNS, ROOM):
    d.mkdir(exist_ok=True)
EXPERIMENTS = ["simple path", "marker-loss path", "closed loop", "repeat run (relocalisation)",
               "difficult: low light", "difficult: fast motion", "difficult: oblique view",
               "difficult: occlusion", "difficult: sparse markers", "calibration board", "other"]
COLORS = {"baseline": "#8a8f98", "optimized": "#1f6feb", "surveyed": "#d29922", "vo": "#2da44e",
          "imu": "#cf222e", "init": "#8250df"}

st.set_page_config(page_title="Marker-anchored VI-SLAM", layout="wide")


# ------------------------------------------------------------------ helpers
def load_cfg():
    return json.loads((ROOT / "config.json").read_text())


def camera_file(ds_root=None):
    return pipeline.find_file("camera.json", ds_root or ROOT)


def camimu_file(ds_root=None):
    return pipeline.find_file("camera_imu.json", ds_root or ROOT)


def map_files():
    files = sorted(ROOM.glob("*.json")) + sorted(DATA.glob("*/markers_surveyed.json")) + sorted(RUNS.glob("*/*/map.json"))
    return [str(p.relative_to(ROOT)) for p in files]


def show_image(bgr, caption=None):
    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption=caption, width="stretch")


def ok_badge(ok, good="ok", bad="check"):
    return f"✅ {good}" if ok else f"⚠️ {bad}"


def parse_ids(text):
    """'0-8', '0,1,2,5', '0-3, 7' -> sorted ids, plus any tokens that could not be read."""
    ids, bad = set(), []
    for tok in text.replace(" ", "").split(","):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", tok)
        if not m:
            bad.append(tok)
            continue
        a_, b_ = int(m.group(1)), int(m.group(2) or m.group(1))
        ids.update(range(min(a_, b_), max(a_, b_) + 1))
    return sorted(ids), bad


def show_error(e):
    st.error(str(e))
    with st.expander("Technical details"):
        st.code(traceback.format_exc(), language=None)


@st.cache_data(show_spinner="Reading dataset...")
def cached_report(path, stamp):
    return dataset_report(load_dataset(path))


def list_runs():
    return sorted(str(p.parent.relative_to(RUNS)) for p in RUNS.rglob("metrics.json"))


def top_down_figure(run_dir, room_frame_T=None, show_baseline=True, surveyed=None):
    fig = go.Figure()

    def tf(P):
        if room_frame_T is None:
            return P
        return P @ room_frame_T[:3, :3].T + room_frame_T[:3, 3]

    for name, dash in (("baseline", "dot"), ("optimized", "solid")):
        if name == "baseline" and not show_baseline:
            continue
        df, _ = pipeline.load_traj(run_dir / f"trajectory_{name}.csv")
        P = tf(df[["x", "y", "z"]].values)
        label = "VO + gyro only (baseline)" if name == "baseline" else "with marker constraints"
        fig.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode="lines", name=label,
                                 line=dict(color=COLORS[name], dash=dash, width=2)))
        fig.add_trace(go.Scatter(x=P[:1, 0], y=P[:1, 1], mode="markers", showlegend=False,
                                 marker=dict(color=COLORS[name], size=10, symbol="circle-open"), hoverinfo="skip"))
    for name, sym in (("map_baseline", "x"), ("map", "square")):
        if name == "map_baseline" and not show_baseline:
            continue
        mk, _ = pipeline.load_map(run_dir / f"{name}.json")
        if not mk:
            continue
        ids = sorted(mk)
        P = tf(np.array([mk[i][:3, 3] for i in ids]))
        N = np.array([mk[i][:3, 2] for i in ids])
        if room_frame_T is not None:
            N = N @ room_frame_T[:3, :3].T
        col = COLORS["optimized"] if name == "map" else COLORS["baseline"]
        fig.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode="markers+text", text=[str(i) for i in ids],
                                 textposition="top center", name="markers (estimated)" if name == "map" else "markers (first sighting)",
                                 marker=dict(symbol=sym, size=11, color=col)))
        if name == "map":
            for p, n in zip(P, N):
                fig.add_trace(go.Scatter(x=[p[0], p[0] + 0.3 * n[0]], y=[p[1], p[1] + 0.3 * n[1]], mode="lines",
                                         line=dict(color=col, width=1), showlegend=False, hoverinfo="skip"))
    if surveyed:
        ids = sorted(surveyed)
        P = np.array([surveyed[i][:3, 3] for i in ids])
        fig.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode="markers", name="markers (tape-measured)",
                                 marker=dict(symbol="circle-open", size=16, color=COLORS["surveyed"], line=dict(width=2))))
    fig.update_layout(height=560, xaxis_title="x [m]", yaxis_title="y [m]", legend=dict(orientation="h", y=-0.15),
                      margin=dict(l=10, r=10, t=30, b=10))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# ------------------------------------------------------------------ sidebar
cfg = load_cfg()
with st.sidebar:
    st.header("Session")
    datasets = list_datasets(DATA)
    ds_name = st.selectbox("Dataset", datasets, index=len(datasets) - 1 if datasets else None,
                           placeholder="Import a recording first")
    st.subheader("Markers")
    cfg["markers"]["marker_size_m"] = st.number_input("Printed marker size (black square) [mm]", 20.0, 500.0,
                                                      cfg["markers"]["marker_size_m"] * 1000, 1.0) / 1000
    cfg["markers"]["dictionary"] = st.selectbox("Dictionary", ["DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250", "DICT_APRILTAG_36h11"],
                                                index=0)
    ids_txt = st.text_input("IDs placed in the room", ",".join(map(str, cfg["markers"].get("valid_ids", []))),
                            help="Detections of other IDs are ignored (protects against false detections). Leave empty to accept all.")
    cfg["markers"]["valid_ids"], bad_ids = parse_ids(ids_txt)
    if bad_ids:
        st.warning(f"Could not read {bad_ids}; use numbers and ranges like 0-8 or 0,1,2,5.")
    if not cfg["markers"]["valid_ids"]:
        st.warning("No IDs given: every detected ID will be accepted, including false detections.")
    st.subheader("Processing")
    cfg["processing"]["stride"] = st.slider("Use every n-th video frame", 1, 6, cfg["processing"]["stride"])
    cfg["frontend"]["lost_motion_model"] = st.radio("When vision fails, propagate position with",
                                                    ["const_vel", "accel"], horizontal=True,
                                                    format_func=lambda v: "constant velocity" if v == "const_vel" else "accelerometer")
    st.subheader("Calibration in use")
    cf = camera_file(DATA / ds_name if ds_name else None)
    cif = camimu_file(DATA / ds_name if ds_name else None)
    if cf:
        c = json.loads(cf.read_text())
        st.caption(f"Camera: {cf.relative_to(ROOT)} · RMS {c.get('rms_px', 0):.2f} px · {c['image_size'][0]}×{c['image_size'][1]}")
    else:
        st.warning("No camera calibration yet (Calibrate tab).")
    if cif:
        c = json.loads(cif.read_text())
        st.caption(f"Camera–IMU: {cif.relative_to(ROOT)} · td {1e3 * c['td_s']:.1f} ms")
    else:
        st.warning("No camera–IMU calibration yet; a default rotation and td = 0 will be used.")

tab_rec, tab_sens, tab_cal, tab_proc, tab_res = st.tabs(["Record", "Sensors", "Calibrate", "Process", "Results"])

# ------------------------------------------------------------------ Record
with tab_rec:
    left, right = st.columns(2)
    with left:
        st.subheader("Phone remote control")
        st.caption("OpenCamera Sensors in video mode, 'Enable remote recording control' switched on, same Wi-Fi as this laptop.")
        ip = st.text_input("Phone IP address", st.session_state.get("ip", "192.168.1.50"))
        st.session_state["ip"] = ip
        c1, c2, c3 = st.columns(3)
        if c1.button("Connect", width="stretch"):
            try:
                st.session_state["remote"] = OpenCameraRemote(ip, timeout=4)
                st.success("Connected")
            except OSError as e:
                st.error(f"Could not reach {ip}:6969 ({e}). Check the IP, Wi-Fi and the remote-control setting.")
        remote = st.session_state.get("remote")
        if c2.button("Start recording", disabled=remote is None, width="stretch", type="primary"):
            try:
                info = remote.start()
                st.session_state["rec_info"], st.session_state["rec_t0"] = info, time.time()
            except Exception as e:
                st.error(f"Start failed: {e}")
        if c3.button("Stop recording", disabled=remote is None, width="stretch"):
            try:
                remote.stop()
                st.session_state["rec_t1"] = time.time()
                st.success("Stopped. Copy the recording to the laptop and import it below.")
            except Exception as e:
                st.error(f"Stop failed: {e}")
        info = st.session_state.get("rec_info")
        if info:
            m1, m2, m3 = st.columns(3)
            m1.metric("Frame rate", f"{info['fps']:.1f} fps" if info["fps"] else "-")
            m2.metric("Exposure", f"{info['exposure_ns'] / 1e6:.1f} ms")
            t_end = st.session_state.get("rec_t1") or time.time()
            m3.metric("Recording time", f"{t_end - st.session_state['rec_t0']:.0f} s")
            if info["exposure_ns"] > 8e6:
                st.warning("Exposure above 8 ms blurs when walking. Lower it (raise ISO / add light).")
        st.markdown("**Protocol:** hold still 2 s facing a marker → side-step ~20 cm with it in view → walk the path → "
                    "for loops, return to the start spot → hold still 2 s → stop.")
    with right:
        st.subheader("Import a recording")
        exp = st.selectbox("Experiment", EXPERIMENTS, index=2)
        name = st.text_input("Dataset name", f"{exp.split(':')[0].split(' ')[0]}_{time.strftime('%m%d_%H%M')}".replace(" ", "_"))
        notes = st.text_area("Notes (lighting, marker layout, anything unusual)", height=80)
        src = st.text_input("Folder or .mp4 copied from the phone's DCIM/OpenCamera",
                            placeholder=r"e.g. C:\Users\me\Downloads\OpenCamera  or  /home/me/phone/OpenCamera")
        b1, b2 = st.columns(2)
        if b1.button("Import from folder", width="stretch", disabled=not src):
            try:
                d = ingest_opencamera(src, DATA, name, notes, extra_meta={"experiment": exp})
                st.success(f"Imported to {d.relative_to(ROOT)}. Select it in the sidebar.")
            except Exception as e:
                show_error(e)
        if b2.button("Pull newest over USB (adb)", width="stretch", disabled=not adb_available(),
                     help=None if adb_available() else "Install Android platform-tools and enable USB debugging"):
            try:
                with st.spinner("Copying from phone..."):
                    vid = adb_pull_latest(ROOT / "incoming" / name)
                    d = ingest_opencamera(vid.parent, DATA, name, notes, video=vid, extra_meta={"experiment": exp})
                st.success(f"Imported to {d.relative_to(ROOT)}.")
            except Exception as e:
                show_error(e)

# ------------------------------------------------------------------ Sensors
with tab_sens:
    if not ds_name:
        st.info("Import a recording in the Record tab.")
    else:
        ds = load_dataset(DATA / ds_name)
        st.caption(f"{ds.meta.get('experiment', '')}  {ds.meta.get('notes', '')}")
        rep = cached_report(str(ds.root), max(p.stat().st_mtime for p in ds.root.iterdir()))
        c = st.columns(5)
        c[0].metric("Duration", f"{rep['duration_s']:.1f} s")
        c[1].metric("Camera", f"{rep['camera_rate_hz']:.1f} Hz", f"max gap {rep['camera_max_gap_ms']:.0f} ms", delta_color="off")
        c[2].metric("Gyro", f"{rep['gyro_rate_hz']:.0f} Hz", f"max gap {rep['gyro_max_gap_ms']:.0f} ms", delta_color="off")
        c[3].metric("Accelerometer", f"{rep['accel_rate_hz']:.0f} Hz")
        c[4].metric("Frames (video / timestamps)", f"{rep['n_video_frames']} / {rep['n_frame_timestamps']}")
        st.write(ok_badge(rep["frame_count_match"], "every video frame has a timestamp",
                          "video and timestamp counts differ; frames are matched in order"), " · ",
                 ok_badge(rep["imu_covers_video"], "IMU covers the whole video",
                          f"IMU starts {rep['imu_start_offset_s']:.2f} s / ends {rep['imu_end_offset_s']:.2f} s relative to video"),
                 " · ", ok_badge(rep["camera_max_gap_ms"] < 2.5e3 / max(rep["camera_rate_hz"], 1), "no dropped frames",
                                 "frame gaps found"))
        al = ds.meta.get("frame_alignment")
        if al and al.get("method") == "container frame times":
            st.caption(f"Frame alignment: video starts at timestamp #{al['first_timestamp_index']}, "
                       f"{100 * al['matched_fraction']:.1f} % of {al['n_video']} video frames matched "
                       f"({al['n_timestamps']} timestamps, median residual {al['median_residual_ms']:.2f} ms)")
        for w in ds.meta.get("import_warnings", []):
            st.warning(w)
        imu = ImuData(ds.gyro, ds.accel)
        static = imu.static_intervals()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ds.gyro[:, 0], y=np.linalg.norm(ds.gyro[:, 1:], axis=1), name="|gyro| [rad/s]", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=ds.accel[:, 0], y=np.linalg.norm(ds.accel[:, 1:], axis=1) - 9.81, name="|accel| − g [m/s²]",
                                 line=dict(width=1), yaxis="y2"))
        for a, b in static:
            fig.add_vrect(x0=a, x1=b, fillcolor="#2da44e", opacity=0.12, line_width=0)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="time [s] (green = held still)",
                          yaxis2=dict(overlaying="y", side="right"), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
        if not static or static[0][0] > 3:
            st.warning("No still period at the start: the gyro bias cannot be estimated. Hold the phone still for 2 s next time.")

        if st.button("Make a small shareable copy (first 20 s, half resolution)"):
            from vislam.dataset import make_excerpt
            with st.spinner("Writing excerpt..."):
                z = make_excerpt(ds, ROOT / "share", 20.0, 0.5, [CALIB / "camera.json", CALIB / "camera_imu.json",
                                                                 ds.root / "camera.json", ds.root / "camera_imu.json"])
            st.success(f"Wrote {z.relative_to(ROOT)} ({z.stat().st_size / 1e6:.1f} MB)")

        st.subheader("Frame preview and marker detection")
        idx = st.slider("Frame", 0, max(rep["n_frame_timestamps"] - 1, 0), 0)
        img = read_frame(ds, idx)
        if img is not None:
            cf = camera_file(ds.root)
            if cf:
                cam = json.loads(cf.read_text())
                det = MarkerDetector(cfg["markers"]["dictionary"], cfg["markers"]["marker_size_m"], cam["K"], cam["dist"],
                                     valid_ids=cfg["markers"].get("valid_ids"),
                                     error_correction_rate=cfg["markers"].get("error_correction_rate", 0.6))
                obs = det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                show_image(draw_detections(img, obs, np.array(cam["K"]), np.array(cam["dist"]), cfg["markers"]["marker_size_m"]),
                           f"t = {ds.frame_t[idx]:.2f} s · {len(obs)} marker(s): " +
                           ", ".join(f"id {o.marker_id} ({o.side_px:.0f} px, {np.linalg.norm(o.T_CM[:3, 3]):.2f} m)" for o in obs))
            else:
                corners, ids, _ = cv2.aruco.ArucoDetector(get_dictionary(cfg["markers"]["dictionary"])).detectMarkers(img)
                vis = img.copy()
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                show_image(vis, "Detection only (calibrate the camera to see marker poses)")

        st.subheader("Camera–IMU sync check on this recording")
        st.caption("Each processing run re-estimates the time offset from visual-odometry rotation vs gyro and compares "
                   "it with the calibrated value. A gap of several tens of ms, or a correlation below ~0.8, points to timestamps "
                   "that are not on a common clock (on short runs with little rotation this estimate is noisy).")
        checks = []
        for p in sorted((RUNS / ds_name).glob("*/metrics.json")):
            s = json.loads(p.read_text()).get("sync_check", {})
            if "td_s" in s:
                checks.append({"run": p.parent.name, "estimated td [ms]": 1e3 * s["td_s"], "calibrated td [ms]": 1e3 * s["td_used_s"],
                               "correlation": s["corr_peak"], "rotation difference [deg]": s["R_CI_diff_deg"]})
        if checks:
            st.dataframe(pd.DataFrame(checks), hide_index=True, width="stretch")
        else:
            st.info("Process this dataset once to see the sync check.")

# ------------------------------------------------------------------ Calibrate
with tab_cal:
    st.caption("Record the printed ChArUco board with the same resolution, focus and stabilisation settings used for "
               "experiments. Move slowly for intrinsics, then rotate briskly about all three axes for the camera–IMU step.")
    c = st.columns(5)
    board = {"squares_x": c[0].number_input("Squares x", 3, 20, 5), "squares_y": c[1].number_input("Squares y", 3, 20, 7),
             "square_m": c[2].number_input("Square [mm] (measure it)", 5.0, 200.0, 35.0) / 1000,
             "marker_m": c[3].number_input("Board marker [mm]", 3.0, 200.0, 26.0) / 1000,
             "dictionary": c[4].selectbox("Board dictionary", ["DICT_5X5_100", "DICT_4X4_50", "DICT_6X6_250"])}
    cal_ds = st.selectbox("Calibration recording", datasets, key="cal_ds")
    a, b = st.columns(2)
    with a:
        if st.button("Calibrate camera", disabled=not cal_ds, type="primary"):
            bar = st.progress(0.0)
            try:
                res = calibrate_camera_charuco(load_dataset(DATA / cal_ds), board, stride=5, progress=bar.progress)
                save_json(res, CALIB / "camera.json")
                st.success(f"Saved calib/camera.json · RMS {res['rms_px']:.3f} px over {res['n_views']} views")
                if res["rms_px"] > 1.0:
                    st.warning("RMS above 1 px: lock focus, avoid blur, keep the board flat, and re-record.")
                st.code(np.array2string(np.array(res["K"]), precision=1), language=None)
                st.bar_chart(pd.DataFrame({"per-view RMS [px]": res["per_view_rms_px"]}))
            except Exception as e:
                show_error(e)
    with b:
        if st.button("Calibrate camera–IMU", disabled=not cal_ds or not (CALIB / "camera.json").exists()):
            bar = st.progress(0.0)
            try:
                dsc = load_dataset(DATA / cal_ds)
                imu = ImuData(dsc.gyro, dsc.accel)
                st_int = imu.static_intervals()
                if st_int:
                    imu.estimate_gyro_bias(st_int[0])
                t, R = board_orientations(dsc, load_json(CALIB / "camera.json"), board, progress=bar.progress)
                res = estimate_cam_imu(t, R, imu, min_dt=0.1, max_dt=0.3)
                res["gyro_bias"], res["source"] = imu.bg.tolist(), str(dsc.root)
                save_json(res, CALIB / "camera_imu.json")
                st.success(f"Saved calib/camera_imu.json · td {1e3 * res['td_s']:.1f} ms · correlation {res['corr_peak']:.3f} · "
                           f"residual {res['omega_rms_resid_rad_s']:.3f} rad/s")
                st.code(np.array2string(np.array(res["R_CI"]), precision=3), language=None)
                f2 = go.Figure(go.Scatter(x=np.array(res["td_curve"]["td"]) * 1e3, y=res["td_curve"]["corr"]))
                f2.update_layout(height=250, xaxis_title="candidate time offset [ms]", yaxis_title="correlation",
                                 margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(f2, width="stretch")
                if res["corr_peak"] < 0.9:
                    st.warning("Correlation below 0.9: do not use this file yet. Run 'Diagnose camera–IMU sync'.")
            except Exception as e:
                show_error(e)
    st.subheader("Diagnose camera–IMU sync")
    st.caption("Splits the calibration recording into 6 s windows and estimates the time offset in each. A stable offset "
               "with high correlation is healthy; an offset that jumps means frames and timestamps are mis-paired; low "
               "correlation everywhere means the visual rotation is too noisy (blur, board too small).")
    if st.button("Diagnose camera–IMU sync", disabled=not cal_ds or not (CALIB / "camera.json").exists()):
        from vislam.calibration import sync_diagnose
        bar = st.progress(0.0)
        try:
            dsc = load_dataset(DATA / cal_ds)
            imu = ImuData(dsc.gyro, dsc.accel)
            st_int = imu.static_intervals()
            if st_int:
                imu.estimate_gyro_bias(st_int[0])
            t, R = board_orientations(dsc, load_json(CALIB / "camera.json"), board, progress=bar.progress)
            d = sync_diagnose(t, R, imu, n_frames=len(dsc.frame_t))
            for v in d["verdicts"]:
                (st.success if v.startswith("consistent") else st.warning)(v)
            g = d["global"]
            st.write(f"Whole recording: td {1e3 * g['td_s']:.1f} ms · correlation {g['corr_peak']:.3f} · residual "
                     f"{g['omega_rms_resid_rad_s']:.3f} rad/s · board in {100 * d.get('board_detected_fraction', 0):.0f} % of frames")
            if d["windows"]:
                st.dataframe(pd.DataFrame(d["windows"]), hide_index=True, width="stretch")
            f3 = go.Figure()
            f3.add_trace(go.Scatter(x=d["series"]["t"], y=d["series"]["vision"], name="vision (board)"))
            f3.add_trace(go.Scatter(x=d["series"]["t"], y=d["series"]["gyro"], name=f"gyro shifted {1e3 * g['td_s']:.0f} ms"))
            f3.update_layout(height=300, xaxis_title="time [s]", yaxis_title="rotation rate [rad/s]",
                             margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
            st.plotly_chart(f3, width="stretch")
            save_json({k: v for k, v in d.items() if k != "series"}, CALIB / f"sync_diag_{cal_ds}.json")
        except Exception as e:
            show_error(e)

# ------------------------------------------------------------------ Process
with tab_proc:
    if not ds_name:
        st.info("Import a recording first.")
    else:
        mode = st.radio("Mode", ["slam", "anchor", "reloc"], horizontal=True, format_func={
            "slam": "Build map (markers unknown)", "anchor": "Anchor to tape-measured markers",
            "reloc": "Relocalise in a saved map"}.get)
        mfile = None
        if mode != "slam":
            opts = [m for m in map_files() if (mode == "anchor") == ("markers_surveyed" in m or m.startswith("room"))] or map_files()
            mfile = st.selectbox("Marker map", opts)
        tag = st.text_input("Run name", mode, help="Becomes a folder name: slashes and spaces are replaced by _")
        if st.button("Process dataset", type="primary", disabled=(mode != "slam" and not mfile)):
            bar, msg = st.progress(0.0), st.empty()
            try:
                out, met = pipeline.run(DATA / ds_name, cfg, mode=mode, map_file=(ROOT / mfile) if mfile else None, tag=tag,
                                        progress=lambda p, s: bar.progress(min(1.0, p), text=s), log=msg.write)
                st.session_state["last_run"] = str(out)
                st.success(f"Done in {met['timing']['total_s']:.1f} s. Open the Results tab.")
                for w in met["warnings"]:
                    st.warning(w)
            except Exception as e:
                show_error(e)

# ------------------------------------------------------------------ Results
with tab_res:
    runs = list_runs()
    if not runs:
        st.info("No processed runs yet.")
    else:
        last = st.session_state.get("last_run")
        default = runs.index(str(Path(last).relative_to(RUNS))) if last and str(Path(last).relative_to(RUNS)) in runs else len(runs) - 1
        run_name = st.selectbox("Run", runs, index=default)
        rd = RUNS / run_name
        met = json.loads((rd / "metrics.json").read_text())

        def g(d, *k):
            for kk in k:
                d = d.get(kk) if isinstance(d, dict) else None
            return d

        rows = [("Start-to-end distance [m] (= drift on a closed loop)", ("closed_loop", "endpoint_drift_m")),
                ("Start-to-end distance per metre travelled", ("closed_loop", "drift_per_m")),
                ("Path length [m]", ("closed_loop", "path_length_m")),
                ("Loop-closure residual, mean [m]", ("loop_closure_residual", "summary", "mean")),
                ("Marker re-observation spread [m]", ("observation_consistency", "summary", "mean")),
                ("Marker position RMSE vs tape [m]", ("marker_map", "position", "rmse")),
                ("Marker normal error, mean [deg]", ("marker_map", "normal_angle", "mean")),
                ("Checkpoint XY RMSE [m]", ("checkpoints", "summary", "rmse")),
                ("Trajectory RMSE vs ground truth [m]", ("ate", "rmse")),
                ("Orientation RMSE [deg]", ("orientation", "orientation", "rmse")),
                ("Heading RMSE [deg]", ("orientation", "heading_abs", "rmse"))]
        table = pd.DataFrame([(n, g(met["baseline"], *k), g(met["optimized"], *k)) for n, k in rows],
                             columns=["Metric", "VO + gyro only (baseline)", "With marker constraints"]).dropna(
            subset=["VO + gyro only (baseline)", "With marker constraints"], how="all")
        left, right = st.columns([3, 2])
        with right:
            st.dataframe(table, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.3f") for c in table.columns[1:]})
            fe = met["frontend"]
            if met.get("marker_size_m"):
                st.write(f"Marker size used: {1e3 * met['marker_size_m']:.0f} mm (sets the metric scale of everything)")
            st.write(f"Frames {fe['n_frames']} · visual tracking {100 * fe['frac_vo']:.0f} % · inertial fallback "
                     f"{100 * fe['frac_imu']:.0f} % · waiting/bootstrap {100 * fe['frac_wait']:.0f} %")
            st.write(f"Marker detection rate {met['detection']['rate']:.2f} ({met['detection']['detected']}/{met['detection']['expected']} "
                     f"expected sightings) · frames with a marker {100 * met['detection']['frames_with_marker']:.0f} %")
            st.write(f"Processing {met['timing']['processing_fps']:.1f} fps · detection {met['timing']['detect_ms_per_frame']:.1f} ms/frame · "
                     f"VO {met['timing']['vo_ms_per_frame']:.1f} ms/frame · optimisation {met['timing']['backend_s']:.1f} s")
            b = met["backend"]
            st.write(f"Optimisation: {b['round1']['n_marker_obs']} marker observations, {b['rejected_obs']} rejected, "
                     f"corner reprojection median {b['corner_reproj_px'].get('median', float('nan')):.2f} px")
            if met.get("relocalization"):
                st.write(f"Relocalisation: first fix after {met['relocalization']['time_to_first_fix_s']:.2f} s on marker "
                         f"{met['relocalization']['marker_id']}")
            sy = met.get("sync", {})
            if sy:
                st.write(f"Time offset used: {1e3 * sy['td_used_s']:.1f} ms from {sy['source']}"
                         + (f" (correlation {sy['corr']:.2f})" if sy.get("source") == "this recording" else "")
                         + f" · calibration {1e3 * sy['td_calibration_s']:.1f} ms"
                         + (" · front-end re-run" if sy.get("rerun") else ""))
            if "td_s" in met.get("sync_check", {}):
                s = met["sync_check"]
                st.write(f"Sync check: td {1e3 * s['td_s']:.1f} ms (calibrated {1e3 * s['td_used_s']:.1f} ms), "
                         f"camera–IMU rotation differs by {s['R_CI_diff_deg']:.2f}°")
            for w in met.get("warnings", []):
                st.warning(w)
            st.download_button("Download map.json", (rd / "map.json").read_text(), "map.json")
            st.download_button("Download metrics.json", (rd / "metrics.json").read_text(), "metrics.json")
        with left:
            surveyed_p = pipeline.find_file("markers_surveyed.json", DATA / met["dataset"])
            surveyed = pipeline.load_map(surveyed_p)[0] if surveyed_p else None
            T_room = None
            al = g(met["optimized"], "marker_map", "alignment")
            if met.get("frame") != "room" and al and surveyed:
                if st.checkbox("Show in room frame (aligned on the tape-measured markers)", True):
                    T_room = np.eye(4)
                    T_room[:3, :3], T_room[:3, 3] = np.array(al["R"]), np.array(al["t"])
            show_base = st.checkbox("Show baseline", True)
            st.plotly_chart(top_down_figure(rd, T_room, show_base, surveyed if (T_room is not None or met.get("frame") == "room") else None),
                            width="stretch")

        st.subheader("Diagnostics over time")
        fe_df = pd.read_csv(rd / "frontend.csv")
        obs_df = pd.read_csv(rd / "observations.csv")
        fig = go.Figure()
        for stt in ("vo", "imu", "init"):
            m = fe_df["status"] == stt
            fig.add_trace(go.Scatter(x=fe_df.t[m], y=fe_df.n_map[m], mode="markers", marker=dict(size=4, color=COLORS[stt]),
                                     name={"vo": "visual tracking", "imu": "inertial fallback", "init": "bootstrap"}[stt]))
        fig.add_trace(go.Scatter(x=obs_df.t, y=obs_df.marker_id, mode="markers", marker=dict(size=5, symbol="line-ns-open",
                                 color="#444"), name="marker id seen", yaxis="y2"))
        fig.update_layout(height=320, xaxis_title="time [s]", yaxis_title="3D points tracked",
                          yaxis2=dict(overlaying="y", side="right", title="marker id"), legend=dict(orientation="h"),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        errs = [p for p in (rd / "error_baseline.csv", rd / "error_optimized.csv") if p.exists()]
        if errs:
            fig = go.Figure()
            for p in errs:
                e = pd.read_csv(p)
                col = e.columns[1]
                fig.add_trace(go.Scatter(x=e.t, y=e[col], name=col.replace("err_", "position error, "),
                                         line=dict(color=COLORS[col.replace("err_", "")])))
            fig.update_layout(height=260, xaxis_title="time [s]", yaxis_title="error vs ground truth [m]",
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
        st.subheader("Markers")
        summ = obs_df.groupby("marker_id").agg(sightings=("t", "size"), first_seen_s=("t", "min"), last_seen_s=("t", "max"),
                                               mean_range_m=("range_m", "mean"), mean_size_px=("side_px", "mean"),
                                               pnp_reproj_px=("pnp_reproj_px", "mean")).reset_index()
        cons = g(met["optimized"], "observation_consistency", "per_marker_rms_m") or {}
        summ["spread_after_m"] = summ.marker_id.astype(str).map(cons)
        mm = g(met["optimized"], "marker_map", "per_marker") or {}
        if mm:
            summ["error_vs_tape_m"] = summ.marker_id.astype(str).map(lambda i: mm.get(i, {}).get("pos_err_m"))
        st.dataframe(summ, hide_index=True, width="stretch")

        st.subheader("All runs")
        allrows = []
        for p in sorted(RUNS.rglob("metrics.json")):
            m = json.loads(p.read_text())
            allrows.append({"run": str(p.parent.relative_to(RUNS)), "mode": m["mode"],
                            "drift base [m]": m["baseline"]["closed_loop"]["endpoint_drift_m"],
                            "drift opt [m]": m["optimized"]["closed_loop"]["endpoint_drift_m"],
                            "loop res base [m]": g(m["baseline"], "loop_closure_residual", "summary", "mean"),
                            "loop res opt [m]": g(m["optimized"], "loop_closure_residual", "summary", "mean"),
                            "map RMSE base [m]": g(m["baseline"], "marker_map", "position", "rmse"),
                            "map RMSE opt [m]": g(m["optimized"], "marker_map", "position", "rmse"),
                            "VO %": 100 * m["frontend"]["frac_vo"], "detection rate": m["detection"]["rate"]})
        st.dataframe(pd.DataFrame(allrows), hide_index=True, width="stretch")
