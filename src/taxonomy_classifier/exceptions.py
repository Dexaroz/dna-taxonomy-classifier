from typing import TYPE_CHECKING, Self, override

if TYPE_CHECKING:
    from pathlib import Path


class GeneflowError(Exception):
    pass


class DataError(GeneflowError):
    pass


class TrainingDeadlineError(GeneflowError):
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
