"""Command-line entry point (the Streamlit app in app.py calls the same functions).

    python run.py targets --size-mm 150 --ids 0-9
    python run.py ingest  <folder copied from phone DCIM/OpenCamera> --name loop1 --experiment closed_loop
    python run.py report  data/loop1
    python run.py calib-camera data/calib_board --square-mm 35 --marker-mm 26
    python run.py calib-imu    data/calib_board
    python run.py sync-diag    data/calib_board      (why a camera-IMU calibration looks bad)
    python run.py process data/loop1 --mode slam
    python run.py process data/loop1 --mode anchor --map room/markers_surveyed.json
    python run.py process data/repeat1 --mode reloc --map runs/loop1/slam/map.json
    python run.py compare
    python run.py share   data/loop1 --seconds 20      (small zip to send for debugging)
    python run.py simulate --out data/sim_loop
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vislam.dataset import ingest_opencamera, load_dataset, dataset_report  # noqa: E402
from vislam.calibration import (calibrate_camera_charuco, board_orientations, estimate_cam_imu,  # noqa: E402
                                save_json, load_json)
from vislam.imu import ImuData  # noqa: E402
from vislam import pipeline  # noqa: E402


def load_config(path=None):
    return json.loads(Path(path or ROOT / "config.json").read_text())


def board_cfg_from(a):
    return {"squares_x": a.squares_x, "squares_y": a.squares_y, "square_m": a.square_mm / 1000,
            "marker_m": a.marker_mm / 1000, "dictionary": a.board_dict}


def cmd_calib_camera(a):
    ds = load_dataset(a.dataset)
    res = calibrate_camera_charuco(ds, board_cfg_from(a), stride=a.stride,
                                   progress=lambda p: print(f"\r  scanning {100 * p:5.1f}%", end=""))
    print()
    out = Path(a.out or ROOT / "calib" / "camera.json")
    save_json(res, out)
    print(f"RMS reprojection {res['rms_px']:.3f} px over {res['n_views']} views -> {out}")
    if res["rms_px"] > 1.0:
        print("WARNING: RMS > 1 px. Re-record with focus locked, less motion blur, board flat.")


def cmd_calib_imu(a):
    ds = load_dataset(a.dataset)
    cam = load_json(a.camera or ROOT / "calib" / "camera.json")
    imu = ImuData(ds.gyro, ds.accel)
    st = imu.static_intervals()
    if st:
        imu.estimate_gyro_bias(st[0])
    t, R = board_orientations(ds, cam, board_cfg_from(a), stride=1,
                              progress=lambda p: print(f"\r  board poses {100 * p:5.1f}%", end=""))
    print()
    res = estimate_cam_imu(t, R, imu, td_range=a.td_range, min_dt=0.1, max_dt=0.3)
    res["gyro_bias"] = imu.bg.tolist()
    res["source"] = str(ds.root)
    out = Path(a.out or ROOT / "calib" / "camera_imu.json")
    save_json(res, out)
    print(f"td = {1e3 * res['td_s']:.1f} ms, correlation {res['corr_peak']:.3f}, "
          f"omega residual {res['omega_rms_resid_rad_s']:.3f} rad/s -> {out}")
    if res["corr_peak"] < 0.9:
        print("WARNING: correlation < 0.9 - run  python run.py sync-diag <dataset>  before trusting this file")


def cmd_sync_diag(a):
    import plotly.graph_objects as go
    from vislam.calibration import sync_diagnose
    ds = load_dataset(a.dataset)
    cam = load_json(a.camera or ROOT / "calib" / "camera.json")
    imu = ImuData(ds.gyro, ds.accel)
    st = imu.static_intervals()
    if st:
        imu.estimate_gyro_bias(st[0])
    t, R = board_orientations(ds, cam, board_cfg_from(a), stride=1,
                              progress=lambda p: print(f"\r  board poses {100 * p:5.1f}%", end=""))
    print()
    d = sync_diagnose(t, R, imu, n_frames=len(ds.frame_t))
    print(json.dumps({k: v for k, v in d.items() if k != "series"}, indent=2))
    out = ROOT / "calib" / f"sync_diag_{ds.root.name}.html"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["series"]["t"], y=d["series"]["vision"], name="|omega| from the board (vision)"))
    fig.add_trace(go.Scatter(x=d["series"]["t"], y=d["series"]["gyro"], name=f"|omega| gyro, shifted by {1e3 * d['global']['td_s']:.0f} ms"))
    fig.update_layout(xaxis_title="time [s]", yaxis_title="rotation rate [rad/s]", title="Vision vs gyro rotation rate")
    out.parent.mkdir(exist_ok=True)
    fig.write_html(out)
    print("plot:", out)


def cmd_process(a):
    cfg = load_config(a.config)
    if a.marker_size_mm:
        cfg["markers"]["marker_size_m"] = a.marker_size_mm / 1000
    if a.stride:
        cfg["processing"]["stride"] = a.stride
    out, met = pipeline.run(a.dataset, cfg, mode=a.mode, map_file=a.map, tag=a.tag)
    print_summary(met)


def print_summary(met):
    rows = [("start-to-end distance [m] (= drift if closed loop)", ("closed_loop", "endpoint_drift_m")),
            ("  ... per metre travelled", ("closed_loop", "drift_per_m")),
            ("loop-closure residual mean [m]", ("loop_closure_residual", "summary", "mean")),
            ("marker re-observation spread [m]", ("observation_consistency", "summary", "mean")),
            ("marker position RMSE [m]", ("marker_map", "position", "rmse")),
            ("checkpoint XY RMSE [m]", ("checkpoints", "summary", "rmse")),
            ("ATE RMSE [m] (ground truth)", ("ate", "rmse"))]

    def get(d, keys):
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d
    print(f"\n{'metric':52s}{'baseline (VO+gyro)':>20s}{'with markers':>16s}")
    for name, keys in rows:
        b, o = get(met["baseline"], keys), get(met["optimized"], keys)
        if b is None and o is None:
            continue
        fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "-"
        print(f"{name:52s}{fmt(b):>20s}{fmt(o):>16s}")
    fe = met["frontend"]
    if met.get("marker_size_m"):
        print(f"\nmarker size used: {1e3 * met['marker_size_m']:.0f} mm (sets the metric scale)", end="")
    print(f"\nframes {fe['n_frames']}, VO {100 * fe['frac_vo']:.0f} %, inertial fallback {100 * fe['frac_imu']:.0f} %, "
          f"detection rate {met['detection']['rate']:.2f}, processing {met['timing']['processing_fps']:.1f} fps")
    sy = met.get("sync", {})
    if sy:
        extra = f", measured on this recording (corr {sy['corr']:.2f})" if sy.get("source") == "this recording" else ""
        print(f"time offset used: {1e3 * sy['td_used_s']:.1f} ms{extra}; calibration {1e3 * sy['td_calibration_s']:.1f} ms"
              + ("; front-end re-run with it" if sy.get("rerun") else ""))
    if met.get("sync_check") and "td_s" in met["sync_check"]:
        s = met["sync_check"]
        print(f"sync check on the final trajectory: td {1e3 * s['td_s']:.1f} ms (used {1e3 * s['td_used_s']:.1f} ms), "
              f"R_CI differs by {s['R_CI_diff_deg']:.2f} deg")
    for w in met.get("warnings", []):
        print("WARNING:", w)


def cmd_compare(a):
    import pandas as pd
    rows = []
    for p in sorted((ROOT / "runs").rglob("metrics.json")):
        m = json.loads(p.read_text())
        r = {"dataset": m["dataset"], "run": str(p.parent.relative_to(ROOT / "runs")), "mode": m["mode"]}
        for v in ("baseline", "optimized"):
            d = m[v]
            r[f"{v}_drift_m"] = d["closed_loop"]["endpoint_drift_m"]
            r[f"{v}_loopres_m"] = d["loop_closure_residual"]["summary"].get("mean")
            r[f"{v}_map_rmse_m"] = d.get("marker_map", {}).get("position", {}).get("rmse")
            r[f"{v}_ate_m"] = d.get("ate", {}).get("rmse")
        r["vo_frac"] = m["frontend"]["frac_vo"]
        r["det_rate"] = m["detection"]["rate"]
        rows.append(r)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}") if len(df) else "no runs yet")
    if a.csv and len(df):
        df.to_csv(a.csv, index=False)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "simulate":   # forward everything to tools/simulate.py
        subprocess.run([sys.executable, str(ROOT / "tools" / "simulate.py")] + sys.argv[2:], check=True)
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("targets")
    s.add_argument("--out", default=str(ROOT / "targets"))
    s.add_argument("--ids", default="0-9")
    s.add_argument("--size-mm", default="150")

    sub.add_parser("simulate", help="synthetic dataset with ground truth; options as in tools/simulate.py")

    s = sub.add_parser("share", help="small zip (first N seconds, downscaled video, calibration) for debugging help")
    s.add_argument("dataset")
    s.add_argument("--seconds", type=float, default=20.0)
    s.add_argument("--scale", type=float, default=0.5)

    s = sub.add_parser("ingest")
    s.add_argument("src")
    s.add_argument("--name")
    s.add_argument("--experiment", default="")
    s.add_argument("--notes", default="")

    s = sub.add_parser("report")
    s.add_argument("dataset")

    for name in ("calib-camera", "calib-imu", "sync-diag"):
        s = sub.add_parser(name)
        s.add_argument("dataset")
        s.add_argument("--squares-x", type=int, default=5)
        s.add_argument("--squares-y", type=int, default=7)
        s.add_argument("--square-mm", type=float, default=35.0)
        s.add_argument("--marker-mm", type=float, default=26.0)
        s.add_argument("--board-dict", default="DICT_5X5_100")
        s.add_argument("--stride", type=int, default=5)
        s.add_argument("--camera")
        s.add_argument("--td-range", type=float, default=0.3)
        s.add_argument("--out")

    s = sub.add_parser("process")
    s.add_argument("dataset")
    s.add_argument("--mode", choices=["slam", "anchor", "reloc"], default="slam")
    s.add_argument("--map")
    s.add_argument("--config")
    s.add_argument("--marker-size-mm", type=float)
    s.add_argument("--stride", type=int)
    s.add_argument("--tag", help="output sub-folder name (default: mode)")

    s = sub.add_parser("compare")
    s.add_argument("--csv")

    a = ap.parse_args()
    if a.cmd == "targets":
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_targets.py"), "--out", a.out, "--ids", a.ids,
                        "--size-mm", a.size_mm], check=True)
    elif a.cmd == "ingest":
        d = ingest_opencamera(a.src, ROOT / "data", a.name, a.notes, extra_meta={"experiment": a.experiment})
        print("imported to", d)
        print(json.dumps(dataset_report(load_dataset(d)), indent=2))
    elif a.cmd == "share":
        from vislam.dataset import make_excerpt
        z = make_excerpt(load_dataset(a.dataset), ROOT / "share", a.seconds, a.scale,
                         [ROOT / "calib" / "camera.json", ROOT / "calib" / "camera_imu.json",
                          Path(a.dataset) / "camera.json", Path(a.dataset) / "camera_imu.json"])
        print(f"wrote {z} ({z.stat().st_size / 1e6:.1f} MB)")
    elif a.cmd == "report":
        print(json.dumps(dataset_report(load_dataset(a.dataset)), indent=2))
    elif a.cmd == "calib-camera":
        cmd_calib_camera(a)
    elif a.cmd == "calib-imu":
        cmd_calib_imu(a)
    elif a.cmd == "sync-diag":
        cmd_sync_diag(a)
    elif a.cmd == "process":
        cmd_process(a)
    elif a.cmd == "compare":
        cmd_compare(a)


if __name__ == "__main__":
    main()
