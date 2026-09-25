from dataclasses import dataclass
import gzip
import logging
import time
from typing import TYPE_CHECKING, Final

import httpx

from taxonomy_classifier.data.files import partial_path
from taxonomy_classifier.exceptions import DownloadError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_LOGGER: Final = logging.getLogger(__name__)

EUTILS: Final = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

TOOL: Final = "geneflow"

_ATTEMPTS: Final = 4


@dataclass(frozen=True, slots=True)
class EntrezQuery:
    term: str
    filename: str
    database: str = "nuccore"
    batch_size: int = 5000
    max_missing_fraction: float = 0.001

    def __post_init__(self) -> None:
        if not self.filename.endswith(".fasta.gz"):
            msg = f"Entrez downloads are stored as .fasta.gz, got {self.filename!r}"
            raise ValueError(msg)

        if self.batch_size < 1:
            msg = f"batch_size must be positive, got {self.batch_size}"
            raise ValueError(msg)

        if not 0.0 <= self.max_missing_fraction < 1.0:
            msg = f"max_missing_fraction must be in [0, 1), got {self.max_missing_fraction}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SearchResult:
    count: int
    web_env: str
    query_key: str


def fetch_query(
    query: EntrezQuery,
    dest_dir: Path,
    *,
    client: httpx.Client,
    pause_seconds: float = 0.4,
    sleep: Callable[[float], None] | None = None,
) -> Path:
    target = dest_dir / query.filename

    if target.exists():
        _LOGGER.info("Already downloaded: %s", target)

        return target

    dest_dir.mkdir(parents=True, exist_ok=True)
    partial = partial_path(target)

    try:
        written = _fetch_all(
            query, partial, client=client, pause_seconds=pause_seconds, sleep=sleep or time.sleep
        )

    except BaseException:
        partial.unlink(missing_ok=True)

        raise

    partial.replace(target)
    _LOGGER.info("Downloaded %d records for %r into %s", written, query.term, target)

    return target


def _fetch_all(
    query: EntrezQuery,
    partial: Path,
    *,
    client: httpx.Client,
    pause_seconds: float,
    sleep: Callable[[float], None],
) -> int:
    def get(endpoint: str, params: dict[str, str | int]) -> httpx.Response:
        return _get(client, endpoint, params, pause_seconds=pause_seconds, sleep=sleep)

    search = _search(get, query)
    written = 0

    _LOGGER.info("Fetching %d records for %r", search.count, query.term)

    with gzip.open(partial, "wt", encoding="utf-8") as handle:
        for start in range(0, search.count, query.batch_size):
            page = get(
                "efetch.fcgi",
                {
                    "db": query.database,
                    "query_key": search.query_key,
                    "WebEnv": search.web_env,
                    "retstart": start,
                    "retmax": query.batch_size,
                    "rettype": "fasta",
                    "retmode": "text",
                },
            ).text

            handle.write(page)
            written += page.count(">")

    missing = search.count - written

    if missing > query.max_missing_fraction * search.count:
        msg = f"Expected {search.count} records for {query.term!r}, received {written}"
        raise DownloadError(msg)

    if missing:
        _LOGGER.warning(
            "Entrez returned %d of %d records for %r", written, search.count, query.term
        )

    return written


def _search(
    get: Callable[[str, dict[str, str | int]], httpx.Response], query: EntrezQuery
) -> SearchResult:
    payload = get(
        "esearch.fcgi",
        {
            "db": query.database,
            "term": query.term,
            "usehistory": "y",
            "retmax": 0,
            "retmode": "json",
        },
    ).json()

    try:
        result = payload["esearchresult"]

        return SearchResult(
            count=int(result["count"]),
            web_env=str(result["webenv"]),
            query_key=str(result["querykey"]),
        )

    except (KeyError, TypeError, ValueError) as error:
        msg = f"Unexpected esearch response for {query.term!r}: {payload!r}"
        raise DownloadError(msg) from error


def _get(
    client: httpx.Client,
    endpoint: str,
    params: dict[str, str | int],
    *,
    pause_seconds: float,
    sleep: Callable[[float], None],
) -> httpx.Response:
    url = f"{EUTILS}/{endpoint}"
    last_error: httpx.HTTPError | None = None

    for attempt in range(1, _ATTEMPTS + 1):
        sleep(pause_seconds * attempt)

        try:
            response = client.get(url, params={**params, "tool": TOOL})
            response.raise_for_status()

        except httpx.HTTPError as error:
            last_error = error
            _LOGGER.warning("Entrez %s attempt %d failed: %s", endpoint, attempt, error)

            continue

        return response

    msg = f"Entrez {endpoint} failed after {_ATTEMPTS} attempts: {last_error}"
    raise DownloadError(msg)
