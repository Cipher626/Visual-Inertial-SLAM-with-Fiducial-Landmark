"""Printable targets at true physical size (print at 100 % / 'actual size', never 'fit to page').

    python tools/make_targets.py --out targets --ids 0-9 --size-mm 150
Produces markers.pdf (one marker per A4 page, two if <= 110 mm, with ID and a 10 cm check line) and charuco_board.pdf."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vislam.calibration import make_board  # noqa: E402
from vislam.markers import get_dictionary  # noqa: E402

DPI = 300
A4 = (2480, 3508)  # px at 300 dpi


def mm(v):
    return int(round(v / 25.4 * DPI))


def parse_ids(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def marker_pages(ids, size_mm, dictionary):
    d = get_dictionary(dictionary)
    pages = []
    per_page = 2 if size_mm <= 110 else 1
    if size_mm > 180:
        raise ValueError("markers larger than 180 mm do not fit on A4 with a label")
    for k in range(0, len(ids), per_page):
        page = np.full(A4[::-1], 255, np.uint8)
        for slot, mid in enumerate(ids[k:k + per_page]):
            img = cv2.aruco.generateImageMarker(d, mid, mm(size_mm))
            x0 = (A4[0] - img.shape[1]) // 2
            y0 = mm(25) + slot * (A4[1] // 2)
            page[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
            ytxt = y0 + img.shape[0] + mm(10)
            cv2.putText(page, f"{dictionary}  id {mid}  black square = {size_mm:.0f} mm", (x0, ytxt),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
            cv2.line(page, (x0, ytxt + mm(8)), (x0 + mm(100), ytxt + mm(8)), 0, 4)
            cv2.putText(page, "100 mm check", (x0 + mm(102), ytxt + mm(10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
        pages.append(Image.fromarray(page))
    return pages


def board_page(board_cfg=None):
    board, c = make_board(board_cfg)
    w_mm, h_mm = c["squares_x"] * c["square_m"] * 1000, c["squares_y"] * c["square_m"] * 1000
    img = board.generateImage((mm(w_mm), mm(h_mm)), marginSize=0, borderBits=1)
    page = np.full(A4[::-1], 255, np.uint8)
    x0, y0 = (A4[0] - img.shape[1]) // 2, mm(15)
    page[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    cv2.putText(page, f"ChArUco {c['squares_x']}x{c['squares_y']}  square {c['square_m']*1000:.0f} mm  marker "
                f"{c['marker_m']*1000:.0f} mm  {c['dictionary']}", (mm(15), y0 + img.shape[0] + mm(12)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
    cv2.putText(page, "Measure one printed square with a ruler and enter the real size in the calibration step.",
                (mm(15), y0 + img.shape[0] + mm(22)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2)
    return Image.fromarray(page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="targets")
    ap.add_argument("--ids", default="0-9")
    ap.add_argument("--size-mm", type=float, default=150)
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pages = marker_pages(parse_ids(a.ids), a.size_mm, a.dictionary)
    pages[0].save(out / "markers.pdf", save_all=True, append_images=pages[1:], resolution=DPI)
    board_page().save(out / "charuco_board.pdf", resolution=DPI)
    print("wrote", out / "markers.pdf", "and", out / "charuco_board.pdf")


if __name__ == "__main__":
    main()
