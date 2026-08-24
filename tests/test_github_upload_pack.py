import base64
import io
from pathlib import Path

import pytest
from pydantic import SecretStr

from senpai_agent.github.upload_pack import (
    GitHubUploadPackError,
    download_github_pack,
)


OID = "1" * 40
SOURCE = "refs/heads/research"
ADVERTISEMENT_TYPE = "application/x-git-upload-pack-advertisement"
RESULT_TYPE = "application/x-git-upload-pack-result"


def packet(payload: bytes) -> bytes:
    return f"{len(payload) + 4:04x}".encode() + payload


def advertisement() -> bytes:
    return b"".join(
        (
            packet(b"# service=git-upload-pack\n"),
            b"0000",
            packet(b"version 2\n"),
            packet(b"agent=git/github-test\n"),
            packet(b"ls-refs=unborn\n"),
            packet(b"fetch=shallow wait-for-done filter\n"),
            packet(b"object-format=sha1\n"),
            b"0000",
        )
    )


def listed_ref(source: str = SOURCE, object_id: str = OID) -> bytes:
    return packet(f"{object_id} {source}\n".encode()) + b"0000"


def pack_result(pack: bytes = b"PACKcontents") -> bytes:
    return b"".join(
        (
            packet(b"packfile\n"),
            packet(b"\x01" + pack[:2]),
            packet(b"\x01" + pack[2:]),
            b"0000",
            b"0002",
        )
    )


class Response(io.BytesIO):
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        content_type: str = RESULT_TYPE,
        content_encoding: str | None = None,
        fragment: int = 3,
    ):
        super().__init__(body)
        self.status = status
        self.fragment = fragment
        self.headers = {"content-type": content_type}
        if content_encoding is not None:
            self.headers["content-encoding"] = content_encoding

    def getheader(self, name: str):
        return self.headers.get(name.lower())

    def read1(self, size: int = -1) -> bytes:
        return super().read(min(size, self.fragment))


def discovery_response(**kwargs) -> Response:
    return Response(
        advertisement(),
        content_type=ADVERTISEMENT_TYPE,
        **kwargs,
    )


def install_connections(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[Response],
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    class Connection:
        def __init__(self, host, port, *, timeout, context):
            assert host == "github.com"
            assert port == 443
            assert timeout > 0
            assert context is not None
            self.call: dict[str, object] = {}

        def request(self, method, path, *, body, headers):
            self.call = {
                "method": method,
                "path": path,
                "body": body,
                "headers": dict(headers),
            }
            calls.append(self.call)

        def getresponse(self):
            return responses.pop(0)

        def close(self):
            return None

    monkeypatch.setattr(
        "senpai_agent.github.upload_pack.http.client.HTTPSConnection",
        Connection,
    )
    return calls


def test_download_uses_fixed_protocol_v2_requests_and_keeps_auth_in_python(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    responses = [
        discovery_response(),
        Response(listed_ref()),
        Response(pack_result()),
    ]
    calls = install_connections(monkeypatch, responses)
    registered: list[str] = []
    monkeypatch.setattr(
        "senpai_agent.weave_monitoring.register_trace_secret",
        registered.append,
    )
    destination = tmp_path / "incoming.pack"

    resolved = download_github_pack(
        repo="acme/widgets",
        token=SecretStr("typed-token"),
        sources=(SOURCE,),
        destination=destination,
        object_exists=lambda _object_id: False,
    )

    authorization = "Basic " + base64.b64encode(
        b"x-access-token:typed-token"
    ).decode()
    assert resolved == {SOURCE: OID}
    assert destination.read_bytes() == b"PACKcontents"
    assert not responses
    assert registered == ["typed-token", authorization]
    assert [call["method"] for call in calls] == ["GET", "POST", "POST"]
    assert calls[0]["path"] == (
        "/acme/widgets.git/info/refs?service=git-upload-pack"
    )
    assert calls[1]["path"] == "/acme/widgets.git/git-upload-pack"
    assert calls[2]["path"] == "/acme/widgets.git/git-upload-pack"
    for call in calls:
        headers = call["headers"]
        assert headers["Authorization"] == authorization
        assert headers["Git-Protocol"] == "version=2"
        assert headers["Accept-Encoding"] == "identity"
        assert "typed-token" not in str(call["path"])
        assert "typed-token" not in str(call["body"])
    assert b"command=ls-refs" in calls[1]["body"]
    assert f"ref-prefix {SOURCE}\n".encode() in calls[1]["body"]
    assert b"command=fetch" in calls[2]["body"]
    assert f"want {OID}\n".encode() in calls[2]["body"]
    assert b"thin-pack" not in calls[2]["body"]
    assert b"have " not in calls[2]["body"]


def test_existing_object_skips_the_pack_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    responses = [discovery_response(), Response(listed_ref())]
    calls = install_connections(monkeypatch, responses)
    destination = tmp_path / "incoming.pack"

    resolved = download_github_pack(
        repo="acme/widgets",
        token=SecretStr("typed-token"),
        sources=(SOURCE,),
        destination=destination,
        object_exists=lambda object_id: object_id == OID,
    )

    assert resolved == {SOURCE: OID}
    assert len(calls) == 2
    assert not destination.exists()


def test_direct_commit_oid_skips_ref_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    responses = [discovery_response(), Response(pack_result())]
    calls = install_connections(monkeypatch, responses)
    destination = tmp_path / "incoming.pack"

    resolved = download_github_pack(
        repo="acme/widgets",
        token=SecretStr("typed-token"),
        sources=(OID.upper(),),
        destination=destination,
        object_exists=lambda _object_id: False,
    )

    assert resolved == {OID.upper(): OID}
    assert destination.read_bytes() == b"PACKcontents"
    assert len(calls) == 2
    assert b"command=fetch" in calls[1]["body"]


@pytest.mark.parametrize(
    "response",
    (
        discovery_response(status=302),
        Response(advertisement(), content_type="text/html"),
        discovery_response(content_encoding="gzip"),
    ),
)
def test_discovery_rejects_untrusted_http_responses_without_leaking_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    response: Response,
):
    install_connections(monkeypatch, [response])
    authorization = base64.b64encode(b"x-access-token:typed-token").decode()

    with pytest.raises(GitHubUploadPackError) as raised:
        download_github_pack(
            repo="acme/widgets",
            token=SecretStr("typed-token"),
            sources=(SOURCE,),
            destination=tmp_path / "incoming.pack",
            object_exists=lambda _object_id: False,
        )

    assert "typed-token" not in str(raised.value)
    assert authorization not in str(raised.value)


@pytest.mark.parametrize(
    "result",
    (
        packet(b"packfile\n") + packet(b"\x03remote failure\n") + b"0000",
        packet(b"packfile\n") + b"0008\x01PA",
        packet(b"packfile\n") + packet(b"\x01NOT-a-pack") + b"0000",
        packet(b"unexpected\n") + b"0000",
    ),
)
def test_invalid_pack_responses_remove_partial_files_and_redact_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    result: bytes,
):
    install_connections(
        monkeypatch,
        [discovery_response(), Response(listed_ref()), Response(result)],
    )
    destination = tmp_path / "incoming.pack"

    with pytest.raises(GitHubUploadPackError) as raised:
        download_github_pack(
            repo="acme/widgets",
            token=SecretStr("typed-token"),
            sources=(SOURCE,),
            destination=destination,
            object_exists=lambda _object_id: False,
        )

    assert not destination.exists()
    assert "typed-token" not in str(raised.value)


@pytest.mark.parametrize(
    "refs",
    (
        b"0000",
        listed_ref()[:-4] + listed_ref(),
    ),
)
def test_list_refs_rejects_missing_or_duplicate_exact_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    refs: bytes,
):
    install_connections(monkeypatch, [discovery_response(), Response(refs)])

    with pytest.raises(GitHubUploadPackError):
        download_github_pack(
            repo="acme/widgets",
            token=SecretStr("typed-token"),
            sources=(SOURCE,),
            destination=tmp_path / "incoming.pack",
            object_exists=lambda _object_id: False,
        )


def test_repository_input_cannot_select_a_host_or_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls = install_connections(monkeypatch, [])

    with pytest.raises(ValueError, match="safe owner/name"):
        download_github_pack(
            repo="acme/widgets?redirect=attacker.invalid",
            token=SecretStr("typed-token"),
            sources=(SOURCE,),
            destination=tmp_path / "incoming.pack",
            object_exists=lambda _object_id: False,
        )

    assert not calls
