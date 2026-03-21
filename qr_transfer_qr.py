"""Encode text payloads to BGR QR images (shared by send and receive)."""

from __future__ import annotations

import logging

import cv2
import numpy as np
import qrcode
from qrcode.exceptions import DataOverflowError

logger = logging.getLogger(__name__)

# Match sender defaults for scannable output
QR_BOX_SIZE = 8
QR_BORDER = 2


def matrix_to_bgr(qr: qrcode.QRCode) -> np.ndarray:
    mat = np.asarray(qr.get_matrix(), dtype=np.uint8)
    box = qr.box_size
    scaled = np.kron(mat, np.ones((box, box), dtype=np.uint8))
    bgr = np.full((*scaled.shape, 3), 255, dtype=np.uint8)
    bgr[scaled.astype(bool)] = (0, 0, 0)
    return bgr


def encode_qr_bgr(payload: str) -> np.ndarray:
    """Top-level for ProcessPoolExecutor pickling."""
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=QR_BOX_SIZE,
        border=QR_BORDER,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    return matrix_to_bgr(qr)


def encode_qr_bgr_indexed(item: tuple[int, str]) -> tuple[int, np.ndarray]:
    idx, payload = item
    return idx, encode_qr_bgr(payload)


def make_qr_bgr(payload: str) -> np.ndarray:
    """Encode payload; on overflow log and re-raise DataOverflowError."""
    try:
        return encode_qr_bgr(payload)
    except DataOverflowError as e:
        logger.error("QR payload too large; try a smaller --chunk-chars: %s", e)
        raise


def resize_to_height(bgr: np.ndarray, target_h: int) -> np.ndarray:
    if bgr.size == 0 or target_h < 1:
        return bgr
    h, w = bgr.shape[:2]
    if h == target_h:
        return bgr
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(bgr, (new_w, target_h), interpolation=cv2.INTER_NEAREST)
