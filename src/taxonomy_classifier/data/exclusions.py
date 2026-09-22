from enum import StrEnum


class ExclusionReason(StrEnum):
    MALFORMED_HEADER = "malformed_header"
    INVALID_SEQUENCE = "invalid_sequence"
    OFF_TARGET = "off_target"
    ORGANELLE = "organelle"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    TOO_AMBIGUOUS = "too_ambiguous"
