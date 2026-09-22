import re
from typing import Final

_CANDIDATUS: Final = "Candidatus "

_UNNAMED: Final = re.compile(
    r"uncultured|unidentified|metagenome|environmental|unknown|unclassified",
    re.IGNORECASE,
)

_EPITHET: Final = re.compile(r"[a-z][a-z-]+")

_UNNAMED_EPITHETS: Final = frozenset({"bacterium", "archaeon", "eukaryote"})

_BINOMIAL_TOKENS: Final = 2


def split_organism(organism: str) -> tuple[str, str | None]:
    tokens = organism.strip().removeprefix(_CANDIDATUS).split()
    genus = tokens[0] if tokens else ""

    if len(tokens) < _BINOMIAL_TOKENS or _UNNAMED.search(organism):
        return genus, None

    epithet = tokens[1]

    if _EPITHET.fullmatch(epithet) is None or epithet in _UNNAMED_EPITHETS:
        return genus, None

    return genus, f"{genus} {epithet}"
