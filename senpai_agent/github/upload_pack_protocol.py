"""Bounded packet-line framing for GitHub upload-pack."""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from typing import BinaryIO


_MAX_PACKET_BYTES = 65_520


class GitHubUploadPackError(RuntimeError):
    """GitHub rejected or returned an invalid upload-pack exchange."""


def packet(payload: bytes) -> bytes:
    return f"{len(payload) + 4:04x}".encode() + payload


def packets(
    stream: BinaryIO,
    *,
    deadline: float | None = None,
) -> Iterator[bytes | int]:
    while True:
        header = _read_exact(stream, 4, allow_eof=True, deadline=deadline)
        if not header:
            return
        try:
            size = int(header, 16)
        except ValueError as error:
            raise GitHubUploadPackError(
                "GitHub returned an invalid packet header"
            ) from error
        if size in (0, 1, 2):
            yield size
            continue
        if size < 4 or size > _MAX_PACKET_BYTES:
            raise GitHubUploadPackError("GitHub returned an invalid packet size")
        yield _read_exact(stream, size - 4, deadline=deadline)


def read_bounded(stream: BinaryIO, limit: int, *, deadline: float) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        _set_stream_timeout(stream, deadline)
        read = getattr(stream, "read1", stream.read)
        chunk = read(min(64 * 1024, limit + 1 - size))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        size += len(chunk)
        if size > limit:
            raise GitHubUploadPackError("GitHub returned an oversized Git response")


def remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise GitHubUploadPackError("GitHub upload-pack timed out")
    return value


def single_section(
    framed: Sequence[bytes | int],
    name: str,
) -> tuple[bytes, ...]:
    if not framed or framed[-1] not in (0, 2):
        raise GitHubUploadPackError(
            f"GitHub returned an incomplete {name} response"
        )
    end = len(framed) - 1
    if framed[-1] == 2:
        if end == 0 or framed[end - 1] != 0:
            raise GitHubUploadPackError(f"GitHub returned invalid {name} framing")
        end -= 1
    payloads = framed[:end]
    if any(not isinstance(payload, bytes) for payload in payloads):
        raise GitHubUploadPackError(f"GitHub returned invalid {name} framing")
    return tuple(payloads)  # type: ignore[return-value]


def _read_exact(
    stream: BinaryIO,
    size: int,
    *,
    allow_eof: bool = False,
    deadline: float | None = None,
) -> bytes:
    chunks: list[bytes] = []
    remaining_bytes = size
    while remaining_bytes:
        if deadline is not None:
            _set_stream_timeout(stream, deadline)
        read = getattr(stream, "read1", stream.read)
        chunk = read(min(remaining_bytes, 64 * 1024))
        if not chunk:
            if allow_eof and remaining_bytes == size:
                return b""
            raise GitHubUploadPackError("GitHub truncated an upload-pack packet")
        chunks.append(chunk)
        remaining_bytes -= len(chunk)
    return b"".join(chunks)


def _set_stream_timeout(stream: BinaryIO, deadline: float) -> None:
    timeout = remaining(deadline)
    raw = getattr(getattr(stream, "fp", None), "raw", None)
    socket = getattr(raw, "_sock", None)
    if socket is not None:
        socket.settimeout(timeout)
