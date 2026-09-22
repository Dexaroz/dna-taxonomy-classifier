from typing import TYPE_CHECKING, Final

from taxonomy_classifier.data.remote import RemoteFile
from taxonomy_classifier.data.sources.gtdb import GtdbSource
from taxonomy_classifier.data.sources.pr2 import Pr2Source
from taxonomy_classifier.data.sources.refseq import RefSeqSource
from taxonomy_classifier.data.sources.silva import SilvaSource

if TYPE_CHECKING:
    from taxonomy_classifier.data.sources.base import DataSource

GTDB_R232: Final = GtdbSource(
    release="r232",
    fasta=RemoteFile(
        url="https://data.gtdb.ecogenomic.org/releases/release232/232.0/genomic_files_all/ssu_all_r232.fna.gz",
        sha256="dc5268786cc6e758283b03f851e9c7478608afda4e7bf31c6f533b429e18919c",
    ),
)

PR2_5_1_1: Final = Pr2Source(
    release="5.1.1",
    fasta=RemoteFile(
        url="https://github.com/pr2database/pr2database/releases/download/v5.1.1/pr2_version_5.1.1_SSU_taxo_long.fasta.gz",
        sha256="b9cef85537b49bf54d1033ad38a6cce1875123fb9f7c7c8f442aadccd2bbb8f5",
    ),
)

REFSEQ_16S: Final = RefSeqSource(
    bacteria=RemoteFile(
        url="https://ftp.ncbi.nlm.nih.gov/refseq/TargetedLoci/Bacteria/bacteria.16SrRNA.fna.gz",
        sha256=None,
    ),
    archaea=RemoteFile(
        url="https://ftp.ncbi.nlm.nih.gov/refseq/TargetedLoci/Archaea/archaea.16SrRNA.fna.gz",
        sha256=None,
    ),
)

SILVA_144: Final = SilvaSource(
    release="144",
    fasta=RemoteFile(
        url="https://www.arb-silva.de/fileadmin/silva_databases/release_144/Exports/SILVA_144_SSURef_NR99_tax_silva_trunc.fasta.gz",
        sha256="2cbdfb7b31e117d8b30dec40f80eebeb947e538d478fd341b8865b09027c2862",
    ),
)

DEFAULT_SOURCES: Final[tuple[DataSource, ...]] = (GTDB_R232, PR2_5_1_1, REFSEQ_16S, SILVA_144)
