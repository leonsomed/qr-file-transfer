#!/usr/bin/env python3
"""Receive a file from QR codes (metadata + ordered hex chunks)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import json
import logging
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from qr_transfer_codec import (
    build_missing_ranges_object,
    format_inclusive_chunk_ranges,
    indices_to_ranges,
    json_compact,
    missing_indices_for_file,
)
from qr_transfer_display import compose_bidirectional_layout
from qr_transfer_constants import (
    CHUNK_TYPE,
    CONTROL_METADATA_CONSECUTIVE_REPEATS,
    MAX_MISSING_RANGE_ENTRIES,
    METADATA_TYPE,
    PROGRESS_CACHE_INTERVAL_DEFAULT,
    PROGRESS_CACHE_VERSION,
)
from qr_transfer_qr import make_qr_bgr

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


def _parse_chunk(obj: dict, metadata: dict | None, bidirectional: bool) -> tuple[int, str] | None:
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
    if bidirectional and metadata is not None:
        sha = obj.get("sha256")
        if isinstance(sha, str) and _SHA256_RE.match(sha):
            if sha.lower() != metadata["sha256"]:
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


def _finalize(
    metadata: dict,
    chunks: dict[int, str],
    out_dir: Path,
    cache_path: Path | None,
) -> int:
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
    if cache_path is not None:
        try:
            cache_path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Could not remove progress cache: %s", e)
    return 0


def _decode_frame_safe(detector: cv2.QRCodeDetector, frame: np.ndarray) -> str:
    """Decode QR from the full camera frame (same coverage as one-way mode)."""
    try:
        data, _, _ = detector.detectAndDecode(frame)
    except cv2.error:
        return ""
    return data or ""


def _load_progress_cache(path: Path) -> tuple[dict | None, dict[int, str], str | None]:
    if not path.is_file():
        return None, {}, None
    try:
        raw = path.read_text(encoding="utf-8")
        doc = json.loads(raw)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Progress cache unreadable (%s); starting fresh", e)
        return None, {}, None
    if doc.get("version") != PROGRESS_CACHE_VERSION:
        logger.error("Progress cache version mismatch; starting fresh")
        return None, {}, None
    meta_obj = doc.get("metadata")
    if not isinstance(meta_obj, dict):
        return None, {}, None
    norm = _normalize_metadata(meta_obj)
    if norm is None:
        logger.error("Progress cache has invalid metadata; starting fresh")
        return None, {}, None
    chunks_raw = doc.get("chunks")
    if not isinstance(chunks_raw, dict):
        return None, {}, None
    chunks: dict[int, str] = {}
    for k, v in chunks_raw.items():
        try:
            ik = int(k)
        except (TypeError, ValueError):
            continue
        if not isinstance(v, str) or not _HEX_RE.match(v):
            continue
        if ik < 0 or ik >= norm["chunks"]:
            continue
        chunks[ik] = v
    mj = json_compact(norm)
    logger.info(
        "Loaded progress cache: %s chunks / %s for %s",
        len(chunks),
        norm["chunks"],
        norm["filename"],
    )
    return norm, chunks, mj


def _save_progress_cache(path: Path, metadata: dict | None, chunks: dict[int, str]) -> None:
    if metadata is None:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    doc = {
        "version": PROGRESS_CACHE_VERSION,
        "metadata": metadata,
        "chunks": {str(k): chunks[k] for k in sorted(chunks)},
    }
    try:
        tmp.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        logger.warning("Progress cache write failed: %s", e)


def _cache_writer_loop(
    path: Path,
    interval: float,
    stop: threading.Event,
    state_lock: threading.Lock,
    get_snapshot: Callable[[], tuple[dict | None, dict[int, str]]],
) -> None:
    while not stop.wait(timeout=interval):
        with state_lock:
            metadata, chunks = get_snapshot()
        _save_progress_cache(path, metadata, chunks)


def _run_oneway(
    cap: cv2.VideoCapture,
    detector: cv2.QRCodeDetector,
    args: argparse.Namespace,
) -> int:
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
                    parsed = _parse_chunk(obj, metadata, False)
                    if parsed is not None:
                        order, chunk_data = parsed
                        new_index = order not in chunks
                        chunks[order] = chunk_data
                        if new_index:
                            _log_data_chunk_progress(metadata, chunks)
            if metadata is not None:
                n = metadata["chunks"]
                if n == 0:
                    return _finalize(metadata, {}, args.output_dir.resolve(), None)
                if len(chunks) == n and set(chunks) == set(range(n)):
                    return _finalize(metadata, chunks, args.output_dir.resolve(), None)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit without completing transfer")
                return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def _run_bidirectional(
    cap: cv2.VideoCapture,
    detector: cv2.QRCodeDetector,
    args: argparse.Namespace,
) -> int:
    state_lock = threading.Lock()
    metadata: dict | None = None
    chunks: dict[int, str] = {}
    metadata_json: str | None = None
    handshake_done = False
    last_max_order: int | None = None
    consecutive_repeat_streak = 0
    last_counted_repeat_order: int | None = None
    echo_bgr: np.ndarray | None = None
    control_bgr: np.ndarray | None = None
    cache_path: Path | None = Path(args.progress_cache).resolve() if args.progress_cache else None

    def snapshot() -> tuple[dict | None, dict[int, str]]:
        return metadata, dict(chunks)

    stop_writer = threading.Event()
    writer_thread: threading.Thread | None = None
    if cache_path is not None:
        writer_thread = threading.Thread(
            target=_cache_writer_loop,
            args=(cache_path, args.progress_interval, stop_writer, state_lock, snapshot),
            name="qr-progress-cache",
            daemon=True,
        )
        writer_thread.start()

    try:
        if cache_path is not None:
            loaded_meta, loaded_chunks, loaded_mj = _load_progress_cache(cache_path)
            if loaded_meta is not None and loaded_mj is not None:
                metadata = loaded_meta
                chunks = loaded_chunks
                metadata_json = loaded_mj
                try:
                    echo_bgr = make_qr_bgr(metadata_json)
                except Exception as e:
                    logger.error("Could not build echo QR from cache: %s", e)
                    return 1
                ntot = metadata["chunks"]
                if len(chunks) >= ntot and ntot > 0 and set(chunks) == set(range(ntot)):
                    logger.info("Cache already complete; finalizing.")
                    stop_writer.set()
                    if writer_thread is not None:
                        writer_thread.join(timeout=2.0)
                    return _finalize(metadata, chunks, args.output_dir.resolve(), cache_path)
                miss = missing_indices_for_file(ntot, set(chunks))
                ranges = indices_to_ranges(miss, MAX_MISSING_RANGE_ENTRIES)
                ctrl = json_compact(build_missing_ranges_object(metadata["sha256"], ranges))
                try:
                    control_bgr = make_qr_bgr(ctrl)
                except Exception as e:
                    logger.error("Could not build control QR from cache: %s", e)
                    return 1
                handshake_done = True
                logger.info("Resume from cache: showing missing_ranges for peer immediately.")
                logger.info(
                    "Handshake (cache resume): metadata and partial chunks loaded — "
                    "showing missing_ranges; still accepting sender QRs from camera."
                )

        window = "QR receive bidirectional (q to quit)"

        if metadata is None:
            logger.info(
                "Handshake step 1/3: Waiting for sender's file_metadata QR "
                "(decoded from the full camera frame; left panel is only the preview)."
            )
        elif not handshake_done:
            logger.info(
                "Handshake step 2/3: file_metadata received — echo QR on the right; "
                "waiting for sender to scan it and start sending data_chunk QRs."
            )
        else:
            logger.info(
                "Handshake already complete at startup — receiving data and updating control QRs."
            )

        def recompute_control_bgr() -> None:
            nonlocal control_bgr
            with state_lock:
                if metadata is None:
                    return
                ntot = metadata["chunks"]
                miss = missing_indices_for_file(ntot, set(chunks))
                sha = metadata["sha256"]
            if not miss:
                with state_lock:
                    control_bgr = None
                return
            ranges = indices_to_ranges(miss, MAX_MISSING_RANGE_ENTRIES)
            payload = json_compact(build_missing_ranges_object(sha, ranges))
            try:
                img = make_qr_bgr(payload)
            except Exception as e:
                logger.warning("Control QR encode failed: %s", e)
                return
            with state_lock:
                control_bgr = img

        def emit_control_immediate(reason: str) -> None:
            recompute_control_bgr()
            if control_bgr is not None:
                logger.info("Control QR updated (%s)", reason)

        def manual_refresh_control() -> None:
            """Rebuild missing_ranges QR from current metadata + chunks (same as automatic updates)."""
            recompute_control_bgr()
            with state_lock:
                has_qr = control_bgr is not None
            logger.info(
                "Manual control QR refresh — %s",
                "missing_ranges QR regenerated from current state"
                if has_qr
                else "no control QR (awaiting metadata or all chunks received)",
            )

        def missing_footer() -> tuple[str, str]:
            title = "Missing (inclusive ranges)"
            with state_lock:
                m = metadata
                have = set(chunks.keys())
            if m is None:
                return title, "Waiting for file metadata"
            ntot = m["chunks"]
            if ntot <= 0:
                return title, "—"
            miss = missing_indices_for_file(ntot, have)
            if not miss:
                return title, "None (complete)"
            r = indices_to_ranges(miss, MAX_MISSING_RANGE_ENTRIES)
            return title, format_inclusive_chunk_ranges(r)

        btn_bounds: list[tuple[int, int, int, int] | None] = [None]
        mouse_setup = [False]

        def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            b = btn_bounds[0]
            if b is None:
                return
            x0, y0, bw, bh = b
            if x0 <= x < x0 + bw and y0 <= y < y0 + bh:
                manual_refresh_control()

        exit_code: int | None = None
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.error("Frame grab failed")
                exit_code = 1
                break

            data = _decode_frame_safe(detector, frame)
            if data:
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    obj = None
                if isinstance(obj, dict):
                    with state_lock:
                        norm = _normalize_metadata(obj)
                        if norm is not None:
                            if metadata is None:
                                metadata = norm
                                metadata_json = json_compact(norm)
                                try:
                                    echo_bgr = make_qr_bgr(metadata_json)
                                except Exception as e:
                                    logger.error("Echo QR failed: %s", e)
                                    return 1
                                logger.info(
                                    "Metadata: %s, %s chunks, sha256=%s…",
                                    metadata["filename"],
                                    metadata["chunks"],
                                    metadata["sha256"][:16],
                                )
                                logger.info(
                                    "Handshake step 2/3: file_metadata received — "
                                    "showing matching echo QR on the right for the sender to scan."
                                )
                                _log_data_chunk_progress(metadata, chunks)
                            elif metadata != norm:
                                metadata = norm
                                chunks = {}
                                metadata_json = json_compact(norm)
                                try:
                                    echo_bgr = make_qr_bgr(metadata_json)
                                except Exception as e:
                                    logger.error("Echo QR failed: %s", e)
                                    return 1
                                handshake_done = False
                                control_bgr = None
                                consecutive_repeat_streak = 0
                                last_counted_repeat_order = None
                                last_max_order = None
                                logger.info(
                                    "Metadata changed: %s, %s chunks, sha256=%s…",
                                    metadata["filename"],
                                    metadata["chunks"],
                                    metadata["sha256"][:16],
                                )
                                logger.info(
                                    "Handshake step 2/3: new file_metadata — echo QR updated; "
                                    "waiting for sender to scan and send data_chunks."
                                )
                                _log_data_chunk_progress(metadata, chunks)
                        meta_ref = metadata
                    parsed = _parse_chunk(obj, meta_ref, True)
                    if parsed is not None and meta_ref is not None:
                        order, chunk_data = parsed
                        ntot = meta_ref["chunks"]
                        # Do not `continue` here: a bad/stale chunk QR in view must not skip
                        # the completion check and finalize path at the bottom of the loop.
                        if order < ntot:
                            with state_lock:
                                had = order in chunks
                                dup = had
                                chunks[order] = chunk_data
                                if last_max_order is None or order > last_max_order:
                                    last_max_order = order
                                new_index = not had
                            if new_index:
                                _log_data_chunk_progress(meta_ref, chunks)
                            if not handshake_done and new_index:
                                handshake_done = True
                                logger.info(
                                    "Handshake step 3/3: Scanned sender data_chunk — "
                                    "handshake complete; receiving remaining chunks / control flow."
                                )
                                emit_control_immediate("after handshake")
                            elif handshake_done:
                                if dup:
                                    if order != last_counted_repeat_order:
                                        consecutive_repeat_streak += 1
                                        last_counted_repeat_order = order
                                    if (
                                        consecutive_repeat_streak
                                        >= CONTROL_METADATA_CONSECUTIVE_REPEATS
                                    ):
                                        emit_control_immediate(
                                            "%s repeated chunks with distinct order numbers "
                                            "(same order in a row counts once)"
                                            % CONTROL_METADATA_CONSECUTIVE_REPEATS
                                        )
                                        consecutive_repeat_streak = 0
                                        last_counted_repeat_order = None
                                else:
                                    consecutive_repeat_streak = 0
                                    last_counted_repeat_order = None

            with state_lock:
                meta_ref = metadata
                if meta_ref is not None:
                    ntot = meta_ref["chunks"]
                    if ntot == 0:
                        logger.info("All data chunks received (0 chunks); assembling file…")
                        exit_code = _finalize(
                            meta_ref, {}, args.output_dir.resolve(), cache_path
                        )
                        break
                    miss = missing_indices_for_file(ntot, set(chunks))
                    if not miss:
                        logger.info(
                            "All %s data chunks received (indices 0..%s); assembling file…",
                            ntot,
                            ntot - 1,
                        )
                        exit_code = _finalize(
                            meta_ref, chunks, args.output_dir.resolve(), cache_path
                        )
                        break

            right = control_bgr
            if right is None:
                right = echo_bgr
            if right is None:
                right = np.full((480, 480, 3), 240, dtype=np.uint8)
            ft, fb = missing_footer()
            display = compose_bidirectional_layout(
                frame,
                right,
                target_h=480,
                footer_title=ft,
                footer_body=fb,
            )
            if not mouse_setup[0]:
                cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback(window, on_mouse)
                mouse_setup[0] = True
            disp = display.copy()
            dh, dw = disp.shape[:2]
            btn_w, btn_h = 158, 28
            bx = max(0, dw - btn_w - 8)
            by = max(0, dh - btn_h - 6)
            btn_bounds[0] = (bx, by, btn_w, btn_h)
            cv2.rectangle(disp, (bx, by), (bx + btn_w - 1, by + btn_h - 1), (55, 75, 105), -1)
            cv2.rectangle(disp, (bx, by), (bx + btn_w - 1, by + btn_h - 1), (200, 210, 230), 1)
            cv2.putText(
                disp,
                "Refresh control",
                (bx + 8, by + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (245, 245, 250),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window, disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit without completing transfer")
                exit_code = 1
                break
        if exit_code is not None:
            return exit_code
        return 0
    finally:
        stop_writer.set()
        if writer_thread is not None:
            writer_thread.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Receive a file from QR codes via camera.")
    parser.add_argument(
        "--mode",
        choices=("one-way", "bidirectional"),
        default="one-way",
        help="one-way: camera only. bidirectional: split view + control QRs",
    )
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
    parser.add_argument(
        "--progress-cache",
        type=Path,
        default=None,
        metavar="PATH",
        help="Bidirectional only: JSON file for resume (periodic save)",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=PROGRESS_CACHE_INTERVAL_DEFAULT,
        metavar="SEC",
        help=f"Seconds between cache writes (default: {PROGRESS_CACHE_INTERVAL_DEFAULT})",
    )
    args = parser.parse_args()

    if args.progress_cache is not None and args.mode != "bidirectional":
        logger.error("--progress-cache is only valid with --mode bidirectional")
        return 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Could not open camera %s", args.camera)
        return 1

    detector = cv2.QRCodeDetector()

    if args.mode == "one-way":
        return _run_oneway(cap, detector, args)
    return _run_bidirectional(cap, detector, args)


if __name__ == "__main__":
    sys.exit(main())
