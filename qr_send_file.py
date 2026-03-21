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

from qr_transfer_codec import (
    format_inclusive_chunk_ranges,
    full_file_chunk_ranges,
    json_compact,
    normalize_missing_ranges_payload,
    ranges_to_indices,
)
from qr_transfer_display import compose_bidirectional_layout
from qr_transfer_constants import (
    CHUNK_HEX_CHARS_DEFAULT,
    CHUNK_HEX_CHARS_MAX,
    CHUNK_HEX_CHARS_MIN,
    CHUNK_TYPE,
    DWELL_SECONDS_DEFAULT,
    DWELL_SECONDS_MAX,
    DWELL_SECONDS_MIN,
    MAX_FILE_BYTES,
    METADATA_TYPE,
)
from qr_transfer_qr import encode_qr_bgr, encode_qr_bgr_indexed, make_qr_bgr

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CHUNK_HEX_CHARS = CHUNK_HEX_CHARS_DEFAULT
MIN_CHUNKS_FOR_PROCESS_POOL = 8

# OpenCV trackbar is integer steps; map 0..N to [DWELL_SECONDS_MIN, DWELL_SECONDS_MAX].
_DWELL_TRACKBAR_MAX = 480


def _dwell_from_trackbar_pos(pos: int) -> float:
    pos = max(0, min(_DWELL_TRACKBAR_MAX, int(pos)))
    span = DWELL_SECONDS_MAX - DWELL_SECONDS_MIN
    return DWELL_SECONDS_MIN + (pos / _DWELL_TRACKBAR_MAX) * span


def _trackbar_pos_from_dwell(dwell: float) -> int:
    dwell = max(DWELL_SECONDS_MIN, min(DWELL_SECONDS_MAX, float(dwell)))
    span = DWELL_SECONDS_MAX - DWELL_SECONDS_MIN
    return int(round((dwell - DWELL_SECONDS_MIN) / span * _DWELL_TRACKBAR_MAX))


def _init_sender_dwell_trackbar(window: str, initial_dwell: float) -> list[float]:
    """Create send window + dwell slider (seconds per QR). Returns mutable [current_dwell]."""
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    dwell_ref = [
        max(DWELL_SECONDS_MIN, min(DWELL_SECONDS_MAX, float(initial_dwell))),
    ]

    def _on_trackbar(pos: int) -> None:
        dwell_ref[0] = _dwell_from_trackbar_pos(pos)

    cv2.createTrackbar(
        "seconds / QR",
        window,
        _trackbar_pos_from_dwell(dwell_ref[0]),
        _DWELL_TRACKBAR_MAX,
        _on_trackbar,
    )
    return dwell_ref


def _qr_strings_from_frame(detector: cv2.QRCodeDetector, frame: np.ndarray) -> list[str]:
    """All decoded QR strings in the frame (peer may show metadata, control, or both)."""
    out: list[str] = []
    if frame is None or frame.size == 0:
        return out
    decode_multi = getattr(detector, "detectAndDecodeMulti", None)
    ret, infos = False, None
    if decode_multi is not None:
        try:
            ret, infos, *_ = decode_multi(frame)
        except cv2.error:
            ret, infos = False, None
    if ret and infos is not None:
        seq = infos if isinstance(infos, (list, tuple)) else (infos,)
        for item in seq:
            if isinstance(item, str) and item:
                out.append(item)
    if not out:
        try:
            s, _, _ = detector.detectAndDecode(frame)
        except cv2.error:
            return out
        if s:
            out.append(s)
    return out


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
    idle_caption: str | None = None,
    get_bidir_footer: Callable[[], tuple[str, str, str | None]] | None = None,
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
            layout_kw: dict = {}
            if get_bidir_footer is not None:
                ft, fb, fh = get_bidir_footer()
                layout_kw["footer_title"] = ft
                layout_kw["footer_body"] = fb
                layout_kw["footer_highlight"] = fh
            cv2.imshow(
                window,
                compose_bidirectional_layout(last_cam, idle_bgr, **layout_kw),
            )
        else:
            vis = (
                _oneway_qr_with_caption_bar(idle_bgr, idle_caption)
                if idle_caption
                else idle_bgr
            )
            cv2.imshow(window, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise QuitRequest()
        time.sleep(0.001)


def _log_encoding_failure(exc: EncodingFailed) -> None:
    cause = exc.__cause__
    if isinstance(cause, DataOverflowError):
        logger.error("QR payload too large; try a smaller --chunk-chars: %s", cause)
    else:
        logger.error("Chunk encoding failed: %s", cause)


def _oneway_qr_with_caption_bar(bgr: np.ndarray, caption: str) -> np.ndarray:
    """One-way mode has no camera feed; draw caption on a bar above the QR (not on the code)."""
    h_qr, w = bgr.shape[:2]
    band_h = max(36, int(h_qr * 0.09))
    bar = np.full((band_h, w, 3), 28, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(1.05, w / 480.0))
    thickness = max(1, int(round(scale * 2)))
    (tw, th), baseline = cv2.getTextSize(caption, font, scale, thickness)
    x = max(4, (w - tw) // 2)
    y = min(band_h - 4, (band_h + th) // 2 + baseline // 2)
    cv2.putText(bar, caption, (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(bar, caption, (x, y), font, scale, (220, 220, 220), thickness, cv2.LINE_AA)
    return np.vstack([bar, bgr])


def _show_code(
    window: str,
    bgr: np.ndarray,
    dwell_ref: list[float],
    caption: str | None = None,
) -> bool:
    deadline = time.perf_counter() + dwell_ref[0]
    vis = _oneway_qr_with_caption_bar(bgr, caption) if caption else bgr
    while time.perf_counter() < deadline:
        cv2.imshow(window, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
    return True


def _show_code_bidirectional(
    window: str,
    bgr: np.ndarray,
    dwell_ref: list[float],
    cap: cv2.VideoCapture,
    detector: cv2.QRCodeDetector,
    digest_lower: str,
    chunk_count: int,
    on_missing_ranges: Callable[[list[list[int]]], bool],
    get_bidir_footer: Callable[[], tuple[str, str, str | None]],
) -> bool:
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    last_cam = placeholder
    deadline = time.perf_counter() + dwell_ref[0]
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
        ft, fb, fh = get_bidir_footer()
        cv2.imshow(
            window,
            compose_bidirectional_layout(
                last_cam,
                bgr,
                footer_title=ft,
                footer_body=fb,
                footer_highlight=fh,
            ),
        )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
    return True


def _run_oneway_loop(
    window: str,
    meta_bgr: np.ndarray,
    dwell_ref: list[float],
    n_chunks: int,
    chunk_results: dict[int, np.ndarray],
    result_lock: threading.Lock,
    error_box: list[BaseException | None],
) -> int:
    try:
        while True:
            if not _show_code(window, meta_bgr, dwell_ref, caption="metadata"):
                break
            for i in range(n_chunks):
                try:
                    bgr = _wait_for_chunk_ready(
                        i,
                        chunk_results,
                        result_lock,
                        error_box,
                        window,
                        meta_bgr,
                        idle_caption="metadata",
                    )
                except QuitRequest:
                    return 0
                except EncodingFailed as e:
                    _log_encoding_failure(e)
                    return 1
                if not _show_code(window, bgr, dwell_ref, caption=f"order {i}"):
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
    dwell_ref: list[float],
    chunk_results: dict[int, np.ndarray],
    result_lock: threading.Lock,
    error_box: list[BaseException | None],
) -> int:
    detector = cv2.QRCodeDetector()
    pending: set[int] = set(range(chunk_count))
    pending_lock = threading.Lock()
    peer_ranges_norm: list[list[int]] | None = None
    outbound_highlight_cell: list[str | None] = [None]

    def sending_footer() -> tuple[str, str, str | None]:
        title = "Sending (inclusive ranges)"
        if chunk_count <= 0:
            body = "—"
        elif peer_ranges_norm is None:
            body = format_inclusive_chunk_ranges(full_file_chunk_ranges(chunk_count))
        else:
            body = format_inclusive_chunk_ranges(peer_ranges_norm)
        return title, body, outbound_highlight_cell[0]

    def apply_ranges(norm_ranges: list[list[int]]) -> bool:
        nonlocal peer_ranges_norm
        idx = ranges_to_indices(norm_ranges)
        idx = {i for i in idx if 0 <= i < chunk_count}
        if not idx:
            logger.warning("Ignoring empty missing_ranges from peer")
            return False
        with pending_lock:
            pending.clear()
            pending.update(idx)
        peer_ranges_norm = [[int(a), int(b)] for a, b in norm_ranges]
        logger.info("Peer requested re-send of chunk indices: %s (count %s)", sorted(pending)[:20], len(pending))
        return True

    try:
        logger.info(
            "Handshake step 1/2: Point the camera at the receiver — scan either their "
            "file_metadata QR (exact JSON match) or a missing_ranges control QR for our file "
            "(sha256 match). Either completes the handshake."
        )
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        last_cam = placeholder
        outbound_highlight_cell[0] = "metadata"
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.error("Camera frame grab failed")
                return 1
            last_cam = frame
            payloads = _qr_strings_from_frame(detector, frame)
            if metadata_json in payloads:
                logger.info(
                    "Handshake step 2/2: Scanned matching file_metadata from peer — "
                    "handshake OK; starting metadata + chunk send loop."
                )
                break
            handshake_via_control = False
            for data in payloads:
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                norm = normalize_missing_ranges_payload(obj, digest_lower, chunk_count)
                if norm is not None and apply_ranges(norm):
                    logger.info(
                        "Handshake step 2/2: Scanned peer missing_ranges for our file first — "
                        "handshake OK without metadata echo; starting send loop."
                    )
                    handshake_via_control = True
                    break
            if handshake_via_control:
                break
            ft, fb, fh = sending_footer()
            cv2.imshow(
                window,
                compose_bidirectional_layout(
                    last_cam,
                    meta_bgr,
                    footer_title=ft,
                    footer_body=fb,
                    footer_highlight=fh,
                ),
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return 0

        logger.info(
            "Handshake finished — showing file_metadata QR once, then data_chunk loop only "
            "(metadata is not repeated between chunk passes); watching camera for missing_ranges."
        )
        outbound_highlight_cell[0] = "metadata"
        if not _show_code_bidirectional(
            window,
            meta_bgr,
            dwell_ref,
            cap,
            detector,
            digest_lower,
            chunk_count,
            apply_ranges,
            sending_footer,
        ):
            return 0
        if chunk_count <= 0:
            return 0
        # Re-fetch pending after each chunk: apply_ranges() may run anytime during dwell /
        # display and must not be ignored until the current `for` loop finishes.
        while True:
            with pending_lock:
                todo = sorted(pending)
            if not todo and chunk_count > 0:
                logger.warning("Pending chunks empty; showing full set until peer sends ranges")
                peer_ranges_norm = None
                with pending_lock:
                    pending = set(range(chunk_count))
                todo = list(range(chunk_count))
            if not todo:
                continue
            snap = tuple(todo)
            peer_updated_pending = False
            for i in todo:
                with pending_lock:
                    if i not in pending:
                        continue
                outbound_highlight_cell[0] = f"order {i}"
                try:
                    bgr = _wait_for_chunk_ready(
                        i,
                        chunk_results,
                        result_lock,
                        error_box,
                        window,
                        meta_bgr,
                        cap,
                        get_bidir_footer=sending_footer,
                    )
                except QuitRequest:
                    return 0
                except EncodingFailed as e:
                    _log_encoding_failure(e)
                    return 1
                with pending_lock:
                    cur = tuple(sorted(pending))
                    still_want = i in pending
                if cur != snap or not still_want:
                    logger.info(
                        "Peer missing_ranges updated — resuming with new indices "
                        "(not waiting to finish the previous list)."
                    )
                    peer_updated_pending = True
                    break
                outbound_highlight_cell[0] = f"order {i}"
                if not _show_code_bidirectional(
                    window,
                    bgr,
                    dwell_ref,
                    cap,
                    detector,
                    digest_lower,
                    chunk_count,
                    apply_ranges,
                    sending_footer,
                ):
                    return 0
                with pending_lock:
                    cur = tuple(sorted(pending))
                if cur != snap:
                    logger.info(
                        "Peer missing_ranges updated — resuming with new indices "
                        "(not waiting to finish the previous list)."
                    )
                    peer_updated_pending = True
                    break
            if peer_updated_pending:
                continue
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
        help=(
            f"Seconds each QR is shown (default: {DWELL_SECONDS_DEFAULT}; "
            f"range: {DWELL_SECONDS_MIN}–{DWELL_SECONDS_MAX}, adjustable live via slider)"
        ),
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
    if args.dwell > DWELL_SECONDS_MAX:
        logger.error("--dwell must be at most %s (got %s)", DWELL_SECONDS_MAX, args.dwell)
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
    dwell_ref = _init_sender_dwell_trackbar(window, args.dwell)
    logger.info(
        "Sending %s (%s bytes), %s data chunks, mode=%s, chunk_chars=%s, dwell=%ss "
        "(use trackbar to change seconds per QR while running)",
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
            window,
            meta_bgr,
            dwell_ref,
            n_chunks,
            chunk_results,
            result_lock,
            error_box,
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
        dwell_ref,
        chunk_results,
        result_lock,
        error_box,
    )


if __name__ == "__main__":
    sys.exit(main())
