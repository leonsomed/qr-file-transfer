"""Bidirectional UI: camera + QR row and range summary footer below."""

from __future__ import annotations

import cv2
import numpy as np

from qr_transfer_qr import resize_to_height


def _left_caption_overlay(bgr: np.ndarray, caption: str) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(1.1, h / 420.0))
    thickness = max(1, int(round(scale * 2)))
    (_tw, _th), baseline = cv2.getTextSize(caption, font, scale, thickness)
    x = max(4, (w - _tw) // 2)
    y = min(h - 6, h - baseline - 4)
    cv2.putText(out, caption, (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(out, caption, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def _wrap_footer_lines(
    text: str,
    font: int,
    scale: float,
    thickness: int,
    max_width: int,
) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        cur = ""
        for w in words:
            trial = f"{cur} {w}".strip()
            tw, _ = cv2.getTextSize(trial, font, scale, thickness)[0]
            if tw <= max_width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                wtw, _ = cv2.getTextSize(w, font, scale, thickness)[0]
                if wtw <= max_width:
                    cur = w
                else:
                    chunk = ""
                    for ch in w:
                        t2 = chunk + ch
                        t2w, _ = cv2.getTextSize(t2, font, scale, thickness)[0]
                        if t2w <= max_width:
                            chunk = t2
                        else:
                            if chunk:
                                lines.append(chunk)
                            chunk = ch
                    cur = chunk
        if cur:
            lines.append(cur)
    return lines if lines else [""]


def _put_text_outline(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x, y = org
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def append_bidirectional_footer(
    top_row_bgr: np.ndarray,
    footer_title: str,
    footer_body: str,
    footer_highlight: str | None = None,
) -> np.ndarray:
    _h, w = top_row_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.52
    body_scale = 0.48
    highlight_scale = 0.5
    title_th = 1
    body_th = 1
    highlight_th = 1
    margin = 8
    max_text_w = max(40, w - 2 * margin)
    body_lines = _wrap_footer_lines(footer_body, font, body_scale, body_th, max_text_w)
    (_w_title, title_h), title_bl = cv2.getTextSize(footer_title, font, title_scale, title_th)
    line_gap = 4
    gap_after_title = 6
    gap_highlight_to_body = 5
    hl_lines: list[str] = []
    hl_heights_sum = 0
    if footer_highlight:
        hl_lines = _wrap_footer_lines(
            footer_highlight, font, highlight_scale, highlight_th, max_text_w
        )
        for hl in hl_lines:
            (_hw, hh), hbl = cv2.getTextSize(hl, font, highlight_scale, highlight_th)
            hl_heights_sum += hh + hbl + line_gap
    body_line_heights: list[int] = []
    for line in body_lines:
        (_bw, bh), bbl = cv2.getTextSize(line, font, body_scale, body_th)
        body_line_heights.append(bh + bbl + line_gap)
    gap_before_body = gap_after_title + hl_heights_sum
    if footer_highlight:
        gap_before_body += gap_highlight_to_body
    footer_h = (
        margin
        + title_h
        + title_bl
        + gap_before_body
        + sum(body_line_heights)
        + margin
    )
    footer_h = max(int(footer_h), 52)
    strip = np.full((footer_h, w, 3), 32, dtype=np.uint8)
    y = margin + title_h
    _put_text_outline(
        strip, footer_title, (margin, y), font, title_scale, (220, 220, 230), title_th
    )
    y += title_bl + gap_after_title
    if footer_highlight and hl_lines:
        for hl in hl_lines:
            (_hw, hh), hbl = cv2.getTextSize(hl, font, highlight_scale, highlight_th)
            y += hh
            _put_text_outline(
                strip,
                hl,
                (margin, y),
                font,
                highlight_scale,
                (190, 230, 190),
                highlight_th,
            )
            y += hbl + line_gap
        y += gap_highlight_to_body - line_gap
    for line in body_lines:
        (_bw, bh), bbl = cv2.getTextSize(line, font, body_scale, body_th)
        y += bh
        _put_text_outline(strip, line, (margin, y), font, body_scale, (200, 200, 210), body_th)
        y += bbl + line_gap
    return np.vstack([top_row_bgr, strip])


def compose_bidirectional_layout(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    *,
    target_h: int = 480,
    left_caption: str | None = None,
    footer_title: str | None = None,
    footer_body: str | None = None,
    footer_highlight: str | None = None,
) -> np.ndarray:
    lh = resize_to_height(left_bgr, target_h)
    rh = resize_to_height(right_bgr, target_h)
    if left_caption:
        lh = _left_caption_overlay(lh, left_caption)
    h = max(lh.shape[0], rh.shape[0])
    lh = resize_to_height(lh, h)
    rh = resize_to_height(rh, h)
    w_top = lh.shape[1] + rh.shape[1]
    top = np.zeros((h, w_top, 3), dtype=np.uint8)
    top[:, : lh.shape[1]] = lh
    top[:, lh.shape[1] :] = rh
    if footer_title is not None and footer_body is not None:
        return append_bidirectional_footer(
            top, footer_title, footer_body, footer_highlight=footer_highlight
        )
    return top
