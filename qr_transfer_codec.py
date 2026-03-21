"""Range helpers and missing_ranges payload parsing for QR file transfer."""

from __future__ import annotations

import json

from qr_transfer_constants import MAX_MISSING_RANGE_ENTRIES, MISSING_RANGES_TYPE


def json_compact(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def ranges_to_indices(ranges: list) -> set[int]:
    out: set[int] = set()
    for pair in ranges:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            a, b = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            continue
        if a > b:
            continue
        for i in range(a, b + 1):
            out.add(i)
    return out


def _merge_adjacent_ranges(ranges: list[list[int]]) -> list[list[int]]:
    if not ranges:
        return []
    s = sorted(ranges, key=lambda x: (x[0], x[1]))
    merged: list[list[int]] = [s[0][:]]
    for start, end in s[1:]:
        last = merged[-1]
        if start <= last[1] + 1:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return merged


def indices_to_ranges(missing: set[int], max_entries: int) -> list[list[int]]:
    """Turn missing chunk indices into inclusive [start, end] ranges, capped at max_entries."""
    if not missing or max_entries < 1:
        return []
    sorted_indices = sorted(missing)
    raw: list[list[int]] = []
    start = prev = sorted_indices[0]
    for x in sorted_indices[1:]:
        if x == prev + 1:
            prev = x
        else:
            raw.append([start, prev])
            start = prev = x
    raw.append([start, prev])
    ranges = _merge_adjacent_ranges(raw)
    while len(ranges) > max_entries:
        best_j = 0
        best_gap = 10**18
        for j in range(len(ranges) - 1):
            gap_between = ranges[j + 1][0] - ranges[j][1] - 1
            if gap_between < best_gap:
                best_gap = gap_between
                best_j = j
        ranges = (
            ranges[:best_j]
            + [[ranges[best_j][0], ranges[best_j + 1][1]]]
            + ranges[best_j + 2 :]
        )
    return ranges


def missing_indices_for_file(metadata_chunks: int, have: set[int]) -> set[int]:
    n = metadata_chunks
    return set(range(n)) - have


def normalize_missing_ranges_payload(
    obj: dict,
    expected_sha256_lower: str,
    chunk_count: int,
) -> list[list[int]] | None:
    """Validate missing_ranges JSON; return normalized list of [start,end] inclusive, or None."""
    if obj.get("type") != MISSING_RANGES_TYPE:
        return None
    sha = obj.get("sha256")
    ranges = obj.get("ranges")
    if not isinstance(sha, str) or sha.lower() != expected_sha256_lower:
        return None
    if not isinstance(ranges, list):
        return None
    normalized: list[list[int]] = []
    for r in ranges:
        if not isinstance(r, (list, tuple)) or len(r) != 2:
            return None
        try:
            a, b = int(r[0]), int(r[1])
        except (TypeError, ValueError):
            return None
        if a > b or a < 0 or b >= chunk_count:
            return None
        normalized.append([a, b])
    merged = _merge_adjacent_ranges(normalized)
    if len(merged) > MAX_MISSING_RANGE_ENTRIES:
        return None
    return merged


def build_missing_ranges_object(sha256_lower: str, ranges: list[list[int]]) -> dict:
    return {
        "type": MISSING_RANGES_TYPE,
        "sha256": sha256_lower,
        "ranges": ranges,
    }
