import gzip
import json
from typing import TYPE_CHECKING

import httpx
import pytest

from taxonomy_classifier.data.entrez import EntrezQuery, fetch_query
from taxonomy_classifier.data.files import partial_path
from taxonomy_classifier.exceptions import DownloadError

if TYPE_CHECKING:
    from pathlib import Path

QUERY = EntrezQuery(term="rbcL[Gene Name]", filename="rbcl.fasta.gz", batch_size=2)

SEARCH = {"esearchresult": {"count": "3", "webenv": "ENV", "querykey": "1"}}

PAGES = {0: ">A1 Alpha beta\nACGT\n>A2 Alpha beta\nACGA\n", 2: ">A3 Alpha gamma\nACGC\n", 4: ""}


class FakeEutils:
    def __init__(self, search: object = SEARCH, failures: int = 0) -> None:
        self.search = search
        self.failures = failures
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)

        if self.failures:
            self.failures -= 1

            return httpx.Response(503)

        if request.url.path.endswith("esearch.fcgi"):
            return httpx.Response(200, content=json.dumps(self.search))

        return httpx.Response(200, text=PAGES[int(request.url.params["retstart"])])

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def _fetch(server: FakeEutils, dest: Path, sleeps: list[float] | None = None) -> Path:
    recorded: list[float] = [] if sleeps is None else sleeps

    with server.client() as client:
        return fetch_query(QUERY, dest, client=client, pause_seconds=0.5, sleep=recorded.append)


def test_fetch_query_pages_through_the_history_server(tmp_path: Path) -> None:
    server = FakeEutils()
    sleeps: list[float] = []

    path = _fetch(server, tmp_path / "raw", sleeps)

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        content = handle.read()

    assert path == tmp_path / "raw" / "rbcl.fasta.gz"
    assert content.count(">") == 3
    assert [request.url.params.get("retstart") for request in server.requests] == [None, "0", "2"]
    assert {request.url.params["tool"] for request in server.requests} == {"geneflow"}
    assert server.requests[1].url.params["WebEnv"] == "ENV"
    assert sleeps == [0.5, 0.5, 0.5]
    assert not partial_path(path).exists()


def test_existing_downloads_are_reused(tmp_path: Path) -> None:
    (tmp_path / "rbcl.fasta.gz").write_bytes(b"")
    server = FakeEutils()

    _fetch(server, tmp_path)

    assert server.requests == []


def test_transient_failures_are_retried_with_growing_pauses(tmp_path: Path) -> None:
    server = FakeEutils(failures=2)
    sleeps: list[float] = []

    _fetch(server, tmp_path, sleeps)

    assert sleeps[:3] == [0.5, 1.0, 1.5]


def test_persistent_failures_raise_and_leave_no_partial_file(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="failed after 4 attempts"):
        _fetch(FakeEutils(failures=10), tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_incomplete_downloads_are_rejected(tmp_path: Path) -> None:
    server = FakeEutils(search={"esearchresult": {"count": "5", "webenv": "E", "querykey": "1"}})

    with pytest.raises(DownloadError, match="Expected 5 records"):
        _fetch(server, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_a_small_shortfall_is_tolerated_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    server = FakeEutils(search={"esearchresult": {"count": "4", "webenv": "E", "querykey": "1"}})
    query = EntrezQuery(term="x", filename="x.fasta.gz", batch_size=2, max_missing_fraction=0.5)

    with server.client() as client:
        fetch_query(query, tmp_path, client=client, sleep=lambda _: None)

    assert "returned 3 of 4 records" in caplog.text


def test_unexpected_search_responses_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="Unexpected esearch response"):
        _fetch(FakeEutils(search={"error": "bad term"}), tmp_path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"filename": "rbcl.fasta"}, "stored as .fasta.gz"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"max_missing_fraction": 1.0}, "max_missing_fraction must be in"),
    ],
)
def test_entrez_query_validates_its_fields(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EntrezQuery(**{"term": "x", "filename": "x.fasta.gz", **kwargs})  # type: ignore[arg-type]
