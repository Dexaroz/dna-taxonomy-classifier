from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Mapping

_SHA256: Final = re.compile(r"[0-9a-f]{64}")

_SILVA_EXPORTS: Final = (
    "https://www.arb-silva.de/fileadmin/silva_databases/release_{version}/Exports"
)


@dataclass(frozen=True, slots=True)
class RemoteFile:
    url: str
    sha256: str

    def __post_init__(self) -> None:
        if urlsplit(self.url).scheme != "https":
            msg = f"Remote files must be served over HTTPS: {self.url!r}"
            raise ValueError(msg)

        if _SHA256.fullmatch(self.sha256) is None:
            msg = f"sha256 must be 64 lowercase hex characters: {self.sha256!r}"
            raise ValueError(msg)

    @property
    def filename(self) -> str:
        return PurePosixPath(urlsplit(self.url).path).name


@dataclass(frozen=True, slots=True)
class SilvaRelease:
    version: str
    fasta: RemoteFile
    taxonomy: RemoteFile

    @property
    def slug(self) -> str:
        return f"silva_{self.version}"


def _silva_release(version: str, *, fasta_sha256: str, taxonomy_sha256: str) -> SilvaRelease:
    exports = _SILVA_EXPORTS.format(version=version)

    return SilvaRelease(
        version=version,
        fasta=RemoteFile(
            url=f"{exports}/SILVA_{version}_SSURef_NR99_tax_silva_trunc.fasta.gz",
            sha256=fasta_sha256,
        ),
        taxonomy=RemoteFile(
            url=f"{exports}/taxonomy/tax_slv_ssu_{version}.txt.gz",
            sha256=taxonomy_sha256,
        ),
    )


SILVA_144: Final = _silva_release(
    "144",
    fasta_sha256="2cbdfb7b31e117d8b30dec40f80eebeb947e538d478fd341b8865b09027c2862",
    taxonomy_sha256="88a4123775689299e5f14f8aa1da0a581519f3b3a5240c9233e7999651b0c74a",
)

RELEASES: Final[Mapping[str, SilvaRelease]] = MappingProxyType({SILVA_144.version: SILVA_144})

LATEST_RELEASE: Final = SILVA_144.version
