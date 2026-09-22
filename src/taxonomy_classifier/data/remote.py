from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from typing import Final
from urllib.parse import urlsplit

_SHA256: Final = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class RemoteFile:
    url: str
    sha256: str | None

    def __post_init__(self) -> None:
        if urlsplit(self.url).scheme != "https":
            msg = f"Remote files must be served over HTTPS: {self.url!r}"
            raise ValueError(msg)

        if self.sha256 is not None and _SHA256.fullmatch(self.sha256) is None:
            msg = f"sha256 must be 64 lowercase hex characters: {self.sha256!r}"
            raise ValueError(msg)

    @property
    def filename(self) -> str:
        return PurePosixPath(urlsplit(self.url).path).name

    @property
    def pinned(self) -> bool:
        return self.sha256 is not None
