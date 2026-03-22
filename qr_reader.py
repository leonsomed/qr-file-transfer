#!/usr/bin/env python3
"""Live QR scanner: write the first valid JSON QR payload to a new file and exit."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read QR codes from the camera and check JSON.")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="PATH",
        help="Path to write the JSON string (must not exist)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.exists():
        logger.error("Output path already exists: %s", out_path)
        return 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Could not open camera %s", args.camera)
        return 1

    detector = cv2.QRCodeDetector()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.error("Frame grab failed")
                return 1

            try:
                data, bbox, _ = detector.detectAndDecode(frame)
            except cv2.error:
                data, bbox = "", None

            if bbox is not None and len(bbox) > 0:
                pts = bbox.astype(int).reshape(-1, 2)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            if data:
                out_path.write_text(data, encoding="utf-8")
                print(data)
                break

            cv2.imshow("QR reader (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
