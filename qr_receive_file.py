#!/usr/bin/env python3
"""Receive a file from QR codes (metadata + ordered hex chunks)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import cv2

from qr_transfer_constants import CHUNK_TYPE, METADATA_TYPE

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def _normalize_metadata(obj: dict) -> dict | None:
    if obj.get("type") != METADATA_TYPE:
        return None
    fn = obj.get("filename")
    sha = obj.get("sha256")
    n = obj.get("chunks")
    if not isinstance(fn, str) or not fn.strip():
        return None
    if not isinstance(sha, str) or not _SHA256_RE.match(sha):
        return None
    if not isinstance(n, int) or n < 0:
        return None
    return {
        "type": METADATA_TYPE,
        "filename": fn,
        "sha256": sha.lower(),
        "chunks": n,
    }


def _parse_chunk(obj: dict) -> tuple[int, str] | None:
    if obj.get("type") != CHUNK_TYPE:
        return None
    order = obj.get("order")
    data = obj.get("data")
    if not isinstance(order, int) or order < 0:
        return None
    if not isinstance(data, str) or not data:
        return None
    if not _HEX_RE.match(data):
        return None
    return order, data


def _log_data_chunk_progress(metadata: dict | None, chunks: dict[int, str]) -> None:
    received = len(chunks)
    if metadata is not None:
        total = metadata["chunks"]
        remaining = max(0, total - received)
        logger.info(
            "Data chunks received: %s/%s (%s remaining)",
            received,
            total,
            remaining,
        )
    else:
        logger.info(
            "Data chunks received: %s (metadata not read yet; total unknown)",
            received,
        )


def _finalize(metadata: dict, chunks: dict[int, str], out_dir: Path) -> int:
    n = metadata["chunks"]
    parts: list[str] = []
    for i in range(n):
        if i not in chunks:
            logger.error("Internal error: missing chunk %s", i)
            return 1
        parts.append(chunks[i])
    hex_concat = "".join(parts)
    if len(hex_concat) % 2 != 0:
        logger.error("Assembled hex has odd length; data may be corrupt")
        return 1
    try:
        raw = bytes.fromhex(hex_concat)
    except ValueError as e:
        logger.error("Invalid assembled hex: %s", e)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(metadata["filename"]).name
    if out_path.exists():
        logger.error("Output file already exists: %s", out_path)
        return 1
    out_path.write_bytes(raw)

    actual = hashlib.sha256(raw).hexdigest()
    expected = metadata["sha256"]
    if actual == expected:
        logger.info("Success: SHA-256 matches metadata (%s).", expected)
    else:
        logger.warning(
            "SHA-256 does not match metadata. Expected %s, got %s",
            expected,
            actual,
        )
    logger.info("Wrote %s (%s bytes)", out_path, len(raw))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Receive a file from QR codes via camera.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write the reconstructed file (default: current directory)",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Could not open camera %s", args.camera)
        return 1

    detector = cv2.QRCodeDetector()
    metadata: dict | None = None
    chunks: dict[int, str] = {}
    window = "QR receive (q to quit)"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.error("Frame grab failed")
                return 1

            try:
                data, bbox, _ = detector.detectAndDecode(frame)
            except cv2.error:
                # OpenCV can throw on degenerate/false-positive contours (e.g. contourArea == 0).
                data, bbox = "", None

            if bbox is not None and len(bbox) > 0:
                pts = bbox.astype(int).reshape(-1, 2)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            if data:
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    obj = None
                if isinstance(obj, dict):
                    norm = _normalize_metadata(obj)
                    if norm is not None:
                        if metadata is None:
                            metadata = norm
                            logger.info(
                                "Metadata: %s, %s chunks, sha256=%s…",
                                metadata["filename"],
                                metadata["chunks"],
                                metadata["sha256"][:16],
                            )
                            _log_data_chunk_progress(metadata, chunks)
                        elif metadata != norm:
                            metadata = norm
                            chunks = {}
                            logger.info(
                                "Metadata changed: %s, %s chunks, sha256=%s…",
                                metadata["filename"],
                                metadata["chunks"],
                                metadata["sha256"][:16],
                            )
                            _log_data_chunk_progress(metadata, chunks)
                    parsed = _parse_chunk(obj)
                    if parsed is not None:
                        order, chunk_data = parsed
                        new_index = order not in chunks
                        chunks[order] = chunk_data
                        if new_index:
                            _log_data_chunk_progress(metadata, chunks)

            if metadata is not None:
                n = metadata["chunks"]
                if n == 0:
                    code = _finalize(metadata, {}, args.output_dir.resolve())
                    return code
                if len(chunks) == n and set(chunks) == set(range(n)):
                    code = _finalize(metadata, chunks, args.output_dir.resolve())
                    return code

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit without completing transfer")
                return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
