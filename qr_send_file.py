#!/usr/bin/env python3
"""Send a file as a repeating sequence of QR codes (metadata + hex chunks)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import qrcode
from qrcode.exceptions import DataOverflowError

from qr_transfer_constants import (
    CHUNK_HEX_CHARS_DEFAULT,
    CHUNK_HEX_CHARS_MAX,
    CHUNK_HEX_CHARS_MIN,
    CHUNK_TYPE,
    DWELL_SECONDS_DEFAULT,
    DWELL_SECONDS_MIN,
    MAX_FILE_BYTES,
    METADATA_TYPE,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Tunable default; overridden by CLI --chunk-chars when provided.
CHUNK_HEX_CHARS = CHUNK_HEX_CHARS_DEFAULT


def _json_compact(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _pil_to_bgr(img) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _make_qr_bgr(payload: str) -> np.ndarray:
    qr = qrcode.QRCode(
        version=40,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=8,
        border=2,
    )
    qr.add_data(payload)
    try:
        qr.make(fit=True)
    except DataOverflowError as e:
        logger.error("QR payload too large; try a smaller --chunk-chars: %s", e)
        raise SystemExit(1) from e
    return _pil_to_bgr(qr.make_image(fill_color="black", back_color="white"))


def _show_code(window: str, bgr: np.ndarray, dwell: float) -> bool:
    """Show QR until dwell elapses. Returns False if user pressed 'q'."""
    deadline = time.perf_counter() + dwell
    while time.perf_counter() < deadline:
        cv2.imshow(window, bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
    return True


def main() -> int:
    global CHUNK_HEX_CHARS

    parser = argparse.ArgumentParser(description="Send a file via looping QR codes.")
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the file to send",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=CHUNK_HEX_CHARS_DEFAULT,
        metavar="N",
        help=(
            f"Max hex characters per chunk data field "
            f"(default: {CHUNK_HEX_CHARS_DEFAULT}, allowed: {CHUNK_HEX_CHARS_MIN}–{CHUNK_HEX_CHARS_MAX})"
        ),
    )
    parser.add_argument(
        "--dwell",
        "--seconds-per-code",
        type=float,
        default=DWELL_SECONDS_DEFAULT,
        dest="dwell",
        metavar="SEC",
        help=f"Seconds each QR is shown (default: {DWELL_SECONDS_DEFAULT}, min: {DWELL_SECONDS_MIN})",
    )
    args = parser.parse_args()

    path = args.file.resolve()
    if not path.is_file():
        logger.error("Not a file: %s", path)
        return 1

    if args.chunk_chars < CHUNK_HEX_CHARS_MIN or args.chunk_chars > CHUNK_HEX_CHARS_MAX:
        logger.error(
            "--chunk-chars must be between %s and %s (got %s)",
            CHUNK_HEX_CHARS_MIN,
            CHUNK_HEX_CHARS_MAX,
            args.chunk_chars,
        )
        return 1

    if args.dwell < DWELL_SECONDS_MIN:
        logger.error("--dwell must be at least %s (got %s)", DWELL_SECONDS_MIN, args.dwell)
        return 1

    CHUNK_HEX_CHARS = args.chunk_chars

    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        logger.error("File too large (%s bytes); max is %s bytes", size, MAX_FILE_BYTES)
        return 1

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    hex_body = raw.hex()

    chunks: list[str] = []
    step = CHUNK_HEX_CHARS
    for i in range(0, len(hex_body), step):
        chunks.append(hex_body[i : i + step])

    metadata = {
        "type": METADATA_TYPE,
        "filename": path.name,
        "sha256": digest,
        "chunks": len(chunks),
    }
    metadata_json = _json_compact(metadata)
    try:
        meta_bgr = _make_qr_bgr(metadata_json)
    except SystemExit:
        return 1

    chunk_bgrs: list[np.ndarray] = []
    for order, piece in enumerate(chunks):
        payload = _json_compact({"type": CHUNK_TYPE, "order": order, "data": piece})
        try:
            chunk_bgrs.append(_make_qr_bgr(payload))
        except SystemExit:
            return 1

    window = "QR send (q to quit)"
    logger.info(
        "Sending %s (%s bytes), %s data chunks, chunk_chars=%s, dwell=%ss",
        path.name,
        size,
        len(chunks),
        CHUNK_HEX_CHARS,
        args.dwell,
    )

    try:
        while True:
            if not _show_code(window, meta_bgr, args.dwell):
                break
            for bgr in chunk_bgrs:
                if not _show_code(window, bgr, args.dwell):
                    return 0
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
