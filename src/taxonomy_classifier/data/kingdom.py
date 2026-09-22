from enum import StrEnum
from typing import Final


class Kingdom(StrEnum):
    BACTERIA = "Bacteria"
    ARCHAEA = "Archaea"
    ANIMALIA = "Animalia"
    FUNGI = "Fungi"
    PLANTAE = "Plantae"
    PROTISTA = "Protista"


_PROKARYOTES: Final = {"Bacteria": Kingdom.BACTERIA, "Archaea": Kingdom.ARCHAEA}


def prokaryote_kingdom(domain: str) -> Kingdom | None:
    return _PROKARYOTES.get(domain)
