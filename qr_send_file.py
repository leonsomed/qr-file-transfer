#!/usr/bin/env python3
"""Send a file as a repeating sequence of QR codes (metadata + hex chunks)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
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
from qrcode.exceptions import DataOverflowError

from qr_transfer_codec import json_compact, normalize_missing_ranges_payload, ranges_to_indices
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
from qr_transfer_qr import encode_qr_bgr, encode_qr_bgr_indexed, make_qr_bgr, resize_to_height

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CHUNK_HEX_CHARS = CHUNK_HEX_CHARS_DEFAULT
MIN_CHUNKS_FOR_PROCESS_POOL = 8


class QuitRequest(Exception):
    """User pressed 'q' while waiting for a chunk to finish encoding."""


class EncodingFailed(Exception):
    """Background encoder failed; original exception is __cause__."""


def _chunk_payload(
    order: int,
    piece: str,
    digest: str,
    bidirectional: bool,
) -> str:
    obj: dict = {"type": CHUNK_TYPE, "order": order, "data": piece}
    if bidirectional:
        obj["sha256"] = digest
    return json_compact(obj)


def _chunk_encoding_worker(
    payloads: list[str],
    chunk_results: dict[int, np.ndarray],
    result_lock: threading.Lock,
    error_box: list[BaseException | None],
) -> None:
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
                    pool.submit(encode_qr_bgr_indexed, (i, p)) for i, p in enumerate(payloads)
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
                bgr = encode_qr_bgr(p)
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
    cap: cv2.VideoCapture | None = None,
) -> np.ndarray:
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    last_cam = placeholder
    while True:
        with result_lock:
            err = error_box[0]
            if err is not None:
                raise EncodingFailed(err) from err
            bgr = chunk_results.get(index)
            if bgr is not None:
                return bgr
        if cap is not None:
            ok, frame = cap.read()
            if ok:
                last_cam = frame
            cv2.imshow(window, _compose_bidir_display(last_cam, idle_bgr))
        else:
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


def _compose_bidir_display(left_bgr: np.ndarray, right_qr_bgr: np.ndarray, target_h: int = 480) -> np.ndarray:
    """Camera / preview on the left, outbound QR on the right (matches receiver layout)."""
    lh = resize_to_height(left_bgr, target_h)
    rh = resize_to_height(right_qr_bgr, target_h)
    h = max(lh.shape[0], rh.shape[0])
    lh = resize_to_height(lh, h)
    rh = resize_to_height(rh, h)
    w = lh.shape[1] + rh.shape[1]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, : lh.shape[1]] = lh
    out[:, lh.shape[1] :] = rh
    return out


def _show_code(window: str, bgr: np.ndarray, dwell: float) -> bool:
    deadline = time.perf_counter() + dwell
    while time.perf_counter() < deadline:
        cv2.imshow(window, bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
    return True


def _show_code_bidirectional(
    window: str,
    bgr: np.ndarray,
    dwell: float,
    cap: cv2.VideoCapture,
    detector: cv2.QRCodeDetector,
    digest_lower: str,
    chunk_count: int,
    on_missing_ranges: Callable[[list[list[int]]], None],
) -> bool:
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    last_cam = placeholder
    deadline = time.perf_counter() + dwell
    while time.perf_counter() < deadline:
        ok, frame = cap.read()
        if ok:
            last_cam = frame
            try:
                data, _, _ = detector.detectAndDecode(frame)
            except cv2.error:
                data = ""
            if data:
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    obj = None
                if isinstance(obj, dict):
                    norm = normalize_missing_ranges_payload(obj, digest_lower, chunk_count)
                    if norm is not None:
                        on_missing_ranges(norm)
        cv2.imshow(window, _compose_bidir_display(last_cam, bgr))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
    return True


def _run_oneway_loop(
    window: str,
    meta_bgr: np.ndarray,
    args: argparse.Namespace,
    n_chunks: int,
    chunk_results: dict[int, np.ndarray],
    result_lock: threading.Lock,
    error_box: list[BaseException | None],
) -> int:
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


def _run_bidirectional_loop(
    window: str,
    cap: cv2.VideoCapture,
    meta_bgr: np.ndarray,
    metadata_json: str,
    digest_lower: str,
    chunk_count: int,
    args: argparse.Namespace,
    chunk_results: dict[int, np.ndarray],
    result_lock: threading.Lock,
    error_box: list[BaseException | None],
) -> int:
    detector = cv2.QRCodeDetector()
    pending: set[int] = set(range(chunk_count))
    pending_lock = threading.Lock()

    def apply_ranges(norm_ranges: list[list[int]]) -> None:
        idx = ranges_to_indices(norm_ranges)
        idx = {i for i in idx if 0 <= i < chunk_count}
        if not idx:
            logger.warning("Ignoring empty missing_ranges from peer")
            return
        with pending_lock:
            pending.clear()
            pending.update(idx)
        logger.info("Peer requested re-send of chunk indices: %s (count %s)", sorted(pending)[:20], len(pending))

    try:
        logger.info(
            "Handshake step 1/2: Waiting to scan the peer's file_metadata QR "
            "(decoded string must match ours). Our metadata QR is on the right; "
            "point the camera at the receiver's screen."
        )
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        last_cam = placeholder
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.error("Camera frame grab failed")
                return 1
            last_cam = frame
            try:
                data, _, _ = detector.detectAndDecode(frame)
            except cv2.error:
                data = ""
            if data == metadata_json:
                logger.info(
                    "Handshake step 2/2: Scanned matching file_metadata from peer — "
                    "handshake OK; starting metadata + chunk send loop."
                )
                break
            cv2.imshow(window, _compose_bidir_display(last_cam, meta_bgr))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return 0

        logger.info(
            "Handshake finished — cycling outbound metadata and data_chunk QRs; "
            "watching camera for peer missing_ranges."
        )
        while True:
            if not _show_code_bidirectional(
                window,
                meta_bgr,
                args.dwell,
                cap,
                detector,
                digest_lower,
                chunk_count,
                apply_ranges,
            ):
                break
            with pending_lock:
                todo = sorted(pending)
            if not todo and chunk_count > 0:
                logger.warning("Pending chunks empty; showing full set until peer sends ranges")
                with pending_lock:
                    pending = set(range(chunk_count))
                todo = list(range(chunk_count))
            for i in todo:
                try:
                    bgr = _wait_for_chunk_ready(
                        i,
                        chunk_results,
                        result_lock,
                        error_box,
                        window,
                        meta_bgr,
                        cap,
                    )
                except QuitRequest:
                    return 0
                except EncodingFailed as e:
                    _log_encoding_failure(e)
                    return 1
                if not _show_code_bidirectional(
                    window,
                    bgr,
                    args.dwell,
                    cap,
                    detector,
                    digest_lower,
                    chunk_count,
                    apply_ranges,
                ):
                    return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    global CHUNK_HEX_CHARS

    parser = argparse.ArgumentParser(description="Send a file via looping QR codes.")
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the file to send",
    )
    parser.add_argument(
        "--mode",
        choices=("one-way", "bidirectional"),
        default="one-way",
        help="one-way: display only. bidirectional: camera + handshake + peer missing_ranges",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for bidirectional mode (default: 0)",
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
    digest_lower = digest.lower()
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
    metadata_json = json_compact(metadata)
    bidir = args.mode == "bidirectional"
    try:
        meta_bgr = make_qr_bgr(metadata_json)
    except DataOverflowError:
        return 1

    chunk_payloads = [
        _chunk_payload(order, piece, digest_lower, bidir)
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
        "Sending %s (%s bytes), %s data chunks, mode=%s, chunk_chars=%s, dwell=%ss",
        path.name,
        size,
        len(chunks),
        args.mode,
        CHUNK_HEX_CHARS,
        args.dwell,
    )

    n_chunks = len(chunk_payloads)

    if args.mode == "one-way":
        return _run_oneway_loop(
            window, meta_bgr, args, n_chunks, chunk_results, result_lock, error_box
        )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Could not open camera %s", args.camera)
        return 1

    return _run_bidirectional_loop(
        window,
        cap,
        meta_bgr,
        metadata_json,
        digest_lower,
        n_chunks,
        args,
        chunk_results,
        result_lock,
        error_box,
    )


if __name__ == "__main__":
    sys.exit(main())
