from typing import TYPE_CHECKING, Self, override

if TYPE_CHECKING:
    from pathlib import Path


class GeneflowError(Exception):
    pass


class DataError(GeneflowError):
    pass


class DownloadError(DataError):
    pass


class ChecksumMismatchError(DownloadError):
    def __init__(self, path: Path, expected: str, actual: str) -> None:
        super().__init__(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
        self.path = path
        self.expected = expected
        self.actual = actual

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[Path, str, str]]:
        return type(self), (self.path, self.expected, self.actual)


class MalformedFastaError(DataError):
    pass


class MalformedHeaderError(DataError):
    pass


class InvalidSequenceError(DataError):
    pass


class TaxonomyError(DataError):
    pass


class UnknownTaxonPathError(TaxonomyError):
    def __init__(self, path: str) -> None:
        super().__init__(f"Lineage path not found in rank map: {path!r}")
        self.path = path

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[str]]:
        return type(self), (self.path,)


class DuplicateRankError(TaxonomyError):
    def __init__(self, rank: str, path: str) -> None:
        super().__init__(f"Rank {rank!r} assigned more than once in lineage {path!r}")
        self.rank = rank
        self.path = path

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[str, str]]:
        return type(self), (self.rank, self.path)
