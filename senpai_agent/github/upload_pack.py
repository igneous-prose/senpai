"""Credential-contained Git protocol v2 reads from GitHub."""

from __future__ import annotations

import base64
import http.client
import io
import re
import ssl
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

from pydantic import SecretStr

from senpai_agent.github.upload_pack_protocol import (
    GitHubUploadPackError,
    packet as _packet,
    packets as _packets,
    read_bounded as _read_bounded,
    remaining as _remaining,
    single_section as _single_section,
)


_GITHUB_HOST = "github.com"
_MAX_DISCOVERY_BYTES = 16 * 1024 * 1024
_MAX_PACK_BYTES = 8 * 1024 * 1024 * 1024
_MAX_PROGRESS_BYTES = 1024 * 1024
_OBJECT_ID = re.compile(rb"[0-9a-fA-F]{40}\Z")
_REPOSITORY = re.compile(
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))"
    r"/(?P<name>[A-Za-z0-9_.-]{1,100})\Z"
)
_TLS_CONTEXT = ssl.create_default_context()


def download_github_pack(
    *,
    repo: str,
    token: SecretStr,
    sources: Sequence[str],
    destination: Path,
    object_exists: Callable[[str], bool],
    timeout: int = 300,
) -> dict[str, str]:
    """Resolve sources and download their missing objects without credentialed Git."""

    repository_path = _repository_path(repo)
    authorization = _basic_authorization(token)
    deadline = time.monotonic() + timeout
    _advertise(
        repository_path,
        authorization,
        deadline=deadline,
    )
    branch_sources = tuple(
        source for source in sources if _OBJECT_ID.fullmatch(source.encode()) is None
    )
    advertised = _list_refs(
        repository_path,
        authorization,
        branch_sources,
        deadline=deadline,
    )
    resolved = {
        source: source.lower() if source not in advertised else advertised[source]
        for source in sources
    }
    wanted = tuple(dict.fromkeys(resolved.values()))
    if all(object_exists(object_id) for object_id in wanted):
        return resolved
    _download_pack(
        repository_path,
        authorization,
        wanted,
        Path(destination),
        deadline=deadline,
    )
    return resolved


def _advertise(
    repository_path: str,
    authorization: str,
    *,
    deadline: float,
) -> frozenset[str]:
    with _request(
        "GET",
        f"{repository_path}/info/refs?service=git-upload-pack",
        authorization,
        accept="application/x-git-upload-pack-advertisement",
        deadline=deadline,
    ) as response:
        body = _read_bounded(response, _MAX_DISCOVERY_BYTES, deadline=deadline)

    packets = list(_packets(io.BytesIO(body)))
    if len(packets) < 5 or packets[:3] != [
        b"# service=git-upload-pack\n",
        0,
        b"version 2\n",
    ]:
        raise GitHubUploadPackError("GitHub returned an invalid protocol v2 prelude")
    if packets[-1] != 0 or any(not isinstance(item, bytes) for item in packets[3:-1]):
        raise GitHubUploadPackError(
            "GitHub returned invalid protocol v2 capabilities"
        )
    try:
        capabilities = frozenset(
            item.decode().removesuffix("\n") for item in packets[3:-1]
        )
    except UnicodeDecodeError as error:
        raise GitHubUploadPackError(
            "GitHub returned invalid protocol v2 capabilities"
        ) from error
    names = {capability.split("=", 1)[0] for capability in capabilities}
    if not {"ls-refs", "fetch"}.issubset(names):
        raise GitHubUploadPackError(
            "GitHub did not offer the required protocol v2 commands"
        )
    if "object-format=sha1" not in capabilities:
        raise GitHubUploadPackError("GitHub did not offer SHA-1 Git objects")
    return capabilities


def _list_refs(
    repository_path: str,
    authorization: str,
    sources: Sequence[str],
    *,
    deadline: float,
) -> dict[str, str]:
    if not sources:
        return {}
    body = b"".join(
        (
            _packet(b"command=ls-refs\n"),
            _packet(b"object-format=sha1\n"),
            b"0001",
            *(_packet(f"ref-prefix {source}\n".encode()) for source in sources),
            b"0000",
        )
    )
    with _request(
        "POST",
        f"{repository_path}/git-upload-pack",
        authorization,
        accept="application/x-git-upload-pack-result",
        content_type="application/x-git-upload-pack-request",
        body=body,
        deadline=deadline,
    ) as response:
        result = _read_bounded(response, _MAX_DISCOVERY_BYTES, deadline=deadline)

    packets = list(_packets(io.BytesIO(result)))
    payloads = _single_section(packets, "ls-refs")
    requested = set(sources)
    refs: dict[str, str] = {}
    seen: set[str] = set()
    for payload in payloads:
        fields = payload.removesuffix(b"\n").split(b" ")
        if len(fields) < 2:
            raise GitHubUploadPackError("GitHub returned an invalid branch ref")
        object_id, raw_name = fields[:2]
        try:
            name = raw_name.decode()
        except UnicodeDecodeError as error:
            raise GitHubUploadPackError("GitHub returned an invalid branch name") from error
        if name in seen:
            raise GitHubUploadPackError(f"GitHub returned duplicate ref {name}")
        seen.add(name)
        if _OBJECT_ID.fullmatch(object_id) is None:
            raise GitHubUploadPackError("GitHub returned an invalid branch object ID")
        if name in requested:
            refs[name] = object_id.decode()
    missing = requested.difference(refs)
    if missing:
        names = ", ".join(sorted(missing))
        raise GitHubUploadPackError(f"GitHub did not advertise {names}")
    return refs


def _download_pack(
    repository_path: str,
    authorization: str,
    object_ids: Sequence[str],
    destination: Path,
    *,
    deadline: float,
) -> None:
    arguments = [b"no-progress\n", b"ofs-delta\n"]
    arguments.extend(f"want {object_id}\n".encode() for object_id in object_ids)
    arguments.append(b"done\n")
    body = b"".join(
        (
            _packet(b"command=fetch\n"),
            _packet(b"object-format=sha1\n"),
            b"0001",
            *(_packet(argument) for argument in arguments),
            b"0000",
        )
    )
    try:
        with _request(
            "POST",
            f"{repository_path}/git-upload-pack",
            authorization,
            accept="application/x-git-upload-pack-result",
            content_type="application/x-git-upload-pack-request",
            body=body,
            deadline=deadline,
        ) as response:
            _write_pack(response, destination, deadline=deadline)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _write_pack(
    response: http.client.HTTPResponse,
    destination: Path,
    *,
    deadline: float,
) -> None:
    packets = iter(_packets(response, deadline=deadline))
    if next(packets, None) != b"packfile\n":
        raise GitHubUploadPackError("GitHub returned no protocol v2 pack section")

    prefix = bytearray()
    pack_bytes = 0
    progress_bytes = 0
    ended = False
    response_ended = False
    with destination.open("xb") as stream:
        for payload in packets:
            if isinstance(payload, int):
                if payload == 0 and not ended:
                    ended = True
                elif payload == 2 and ended and not response_ended:
                    response_ended = True
                else:
                    raise GitHubUploadPackError(
                        "GitHub returned invalid pack section framing"
                    )
                continue
            if ended:
                raise GitHubUploadPackError(
                    "GitHub returned data after the pack section"
                )
            channel, data = payload[:1], payload[1:]
            if channel == b"\x02":
                progress_bytes += len(data)
                if progress_bytes > _MAX_PROGRESS_BYTES:
                    raise GitHubUploadPackError(
                        "GitHub returned excessive upload-pack progress"
                    )
                continue
            if channel == b"\x03":
                raise GitHubUploadPackError(
                    "GitHub upload-pack reported a remote failure"
                )
            if channel != b"\x01":
                raise GitHubUploadPackError(
                    "GitHub returned an invalid pack side-band channel"
                )
            pack_bytes += len(data)
            if pack_bytes > _MAX_PACK_BYTES:
                raise GitHubUploadPackError("GitHub returned an oversized Git pack")
            if len(prefix) < 4:
                prefix.extend(data[: 4 - len(prefix)])
            stream.write(data)
    if not ended or pack_bytes == 0 or bytes(prefix) != b"PACK":
        raise GitHubUploadPackError("GitHub returned an incomplete Git pack")


@contextmanager
def _request(
    method: str,
    path: str,
    authorization: str,
    *,
    accept: str,
    deadline: float,
    content_type: str | None = None,
    body: bytes | None = None,
) -> Iterator[http.client.HTTPResponse]:
    connection = http.client.HTTPSConnection(
        _GITHUB_HOST,
        443,
        timeout=_remaining(deadline),
        context=_TLS_CONTEXT,
    )
    headers = {
        "Authorization": authorization,
        "Accept": accept,
        "Accept-Encoding": "identity",
        "Git-Protocol": "version=2",
        "User-Agent": "senpai-git-ref-sync/1",
    }
    if content_type is not None:
        headers["Content-Type"] = content_type
    if body is not None:
        headers["Content-Length"] = str(len(body))
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        _check_response(response, expected=accept)
        yield response
    except GitHubUploadPackError:
        raise
    except (OSError, TimeoutError, http.client.HTTPException):
        raise GitHubUploadPackError(
            "GitHub upload-pack failed before a valid response"
        ) from None
    finally:
        connection.close()


def _check_response(response: http.client.HTTPResponse, *, expected: str) -> None:
    if response.status != 200:
        raise GitHubUploadPackError(
            f"GitHub upload-pack returned HTTP {response.status}"
        )
    content_type = (response.getheader("Content-Type") or "").split(";", 1)[0]
    if content_type.lower() != expected:
        raise GitHubUploadPackError(
            "GitHub upload-pack returned an unexpected content type"
        )
    encoding = (response.getheader("Content-Encoding") or "identity").lower()
    if encoding != "identity":
        raise GitHubUploadPackError("GitHub upload-pack returned encoded content")


def _repository_path(repo: str) -> str:
    match = _REPOSITORY.fullmatch(repo)
    if match is None:
        raise ValueError("repo must use a safe owner/name form")
    owner = quote(match.group("owner"), safe="")
    name = quote(match.group("name"), safe="")
    return f"/{owner}/{name}.git"


def _basic_authorization(token: SecretStr) -> str:
    if not isinstance(token, SecretStr):
        raise TypeError("token must be a SecretStr")
    value = token.get_secret_value()
    if not value.strip():
        raise ValueError("token must not be empty")
    authorization = "Basic " + base64.b64encode(
        f"x-access-token:{value}".encode()
    ).decode()
    from senpai_agent.weave_monitoring import register_trace_secret

    register_trace_secret(value)
    register_trace_secret(authorization)
    return authorization
