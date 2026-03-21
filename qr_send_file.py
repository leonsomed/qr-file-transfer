#!/usr/bin/env python3
"""Send a file as a repeating sequence of QR codes (metadata + hex chunks)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# Use worker processes when chunk count is high enough to amortize pool startup.
MIN_CHUNKS_FOR_PROCESS_POOL = 8


class QuitRequest(Exception):
    """User pressed 'q' while waiting for a chunk to finish encoding."""


class EncodingFailed(Exception):
    """Background encoder failed; original exception is __cause__."""


def _json_compact(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _matrix_to_bgr(qr: qrcode.QRCode) -> np.ndarray:
    """Rasterize QR modules to a BGR uint8 image (white background, black modules)."""
    mat = np.asarray(qr.get_matrix(), dtype=np.uint8)
    box = qr.box_size
    scaled = np.kron(mat, np.ones((box, box), dtype=np.uint8))
    bgr = np.full((*scaled.shape, 3), 255, dtype=np.uint8)
    bgr[scaled.astype(bool)] = (0, 0, 0)
    return bgr


def _encode_qr_bgr(payload: str) -> np.ndarray:
    """Encode payload to a QR image. Top-level for ProcessPoolExecutor pickling."""
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=6,
        border=2,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    return _matrix_to_bgr(qr)


def _encode_qr_bgr_indexed(item: tuple[int, str]) -> tuple[int, np.ndarray]:
    """Same as _encode_qr_bgr but returns index for ordered reassembly (picklable)."""
    idx, payload = item
    return idx, _encode_qr_bgr(payload)


def _make_qr_bgr(payload: str) -> np.ndarray:
    try:
        return _encode_qr_bgr(payload)
    except DataOverflowError as e:
        logger.error("QR payload too large; try a smaller --chunk-chars: %s", e)
        raise SystemExit(1) from e


def _chunk_encoding_worker(
    payloads: list[str],
    chunk_results: dict[int, np.ndarray],
    result_lock: threading.Lock,
    error_box: list[BaseException | None],
) -> None:
    """Encode chunk QRs in a background thread; store by index as each finishes."""
    try:
        if not payloads:
            return
        if len(payloads) >= MIN_CHUNKS_FOR_PROCESS_POOL:
            workers = max(1, min(len(payloads), os.cpu_count() or 4))
            total = len(payloads)
            logger.info(
                "Encoding %s chunk QRs in background (%s worker processes)",
                total,
                workers,
            )
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(_encode_qr_bgr_indexed, (i, p)) for i, p in enumerate(payloads)
                ]
                completed = 0
                for fut in as_completed(futures):
                    idx, bgr = fut.result()
                    with result_lock:
                        chunk_results[idx] = bgr
                    completed += 1
                    logger.info(
                        "Chunk QR workers progress: %s/%s complete",
                        completed,
                        total,
                    )
        else:
            total = len(payloads)
            for i, p in enumerate(payloads):
                bgr = _encode_qr_bgr(p)
                with result_lock:
                    chunk_results[i] = bgr
                logger.info("Chunk QR encoding progress: %s/%s complete", i + 1, total)
    except BaseException as e:
        with result_lock:
            if error_box[0] is None:
                error_box[0] = e


def _wait_for_chunk_ready(
    index: int,
    chunk_results: dict[int, np.ndarray],
    result_lock: threading.Lock,
    error_box: list[BaseException | None],
    window: str,
    idle_bgr: np.ndarray,
) -> np.ndarray:
    """Block until chunk ``index`` is encoded, encoding error, or user presses 'q'."""
    while True:
        with result_lock:
            err = error_box[0]
            if err is not None:
                raise EncodingFailed(err) from err
            bgr = chunk_results.get(index)
            if bgr is not None:
                return bgr
        cv2.imshow(window, idle_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise QuitRequest()
        time.sleep(0.001)


def _log_encoding_failure(exc: EncodingFailed) -> None:
    cause = exc.__cause__
    if isinstance(cause, DataOverflowError):
        logger.error("QR payload too large; try a smaller --chunk-chars: %s", cause)
    else:
        logger.error("Chunk encoding failed: %s", cause)


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

    chunk_payloads = [
        _json_compact({"type": CHUNK_TYPE, "order": order, "data": piece})
        for order, piece in enumerate(chunks)
    ]

    chunk_results: dict[int, np.ndarray] = {}
    result_lock = threading.Lock()
    error_box: list[BaseException | None] = [None]
    encode_thread = threading.Thread(
        target=_chunk_encoding_worker,
        args=(chunk_payloads, chunk_results, result_lock, error_box),
        name="qr-chunk-encode",
        daemon=True,
    )
    encode_thread.start()

    window = "QR send (q to quit)"
    logger.info(
        "Sending %s (%s bytes), %s data chunks, chunk_chars=%s, dwell=%ss",
        path.name,
        size,
        len(chunks),
        CHUNK_HEX_CHARS,
        args.dwell,
    )

    n_chunks = len(chunk_payloads)
    try:
        while True:
            if not _show_code(window, meta_bgr, args.dwell):
                break
            for i in range(n_chunks):
                try:
                    bgr = _wait_for_chunk_ready(
                        i, chunk_results, result_lock, error_box, window, meta_bgr
                    )
                except QuitRequest:
                    return 0
                except EncodingFailed as e:
                    _log_encoding_failure(e)
                    return 1
                if not _show_code(window, bgr, args.dwell):
                    return 0
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
