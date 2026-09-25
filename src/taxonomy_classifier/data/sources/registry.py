from typing import TYPE_CHECKING, Final

from taxonomy_classifier.data.entrez import EntrezQuery
from taxonomy_classifier.data.markers import Marker
from taxonomy_classifier.data.remote import RemoteFile
from taxonomy_classifier.data.sources.eukribo import EukRiboSource
from taxonomy_classifier.data.sources.genbank import GenBankSource
from taxonomy_classifier.data.sources.gtdb import GtdbSource
from taxonomy_classifier.data.sources.midori import Midori2Source
from taxonomy_classifier.data.sources.ncbi import NcbiTaxonomy
from taxonomy_classifier.data.sources.pr2 import Pr2Source
from taxonomy_classifier.data.sources.refseq import RefSeqLocus, RefSeqSource
from taxonomy_classifier.data.sources.silva import SilvaSource
from taxonomy_classifier.data.sources.unite import UniteSource

if TYPE_CHECKING:
    from taxonomy_classifier.data.sources.base import DataSource, TaxonomySource

NCBI_TAXONOMY_2026_09: Final = NcbiTaxonomy(
    release="2026-09-01",
    archive=RemoteFile(
        url="https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump_archive/new_taxdump_2026-09-01.zip",
        sha256="6b0670e633daee01196e2c15305771e39b5050d30c1b7ff4567dd635b37a9c76",
    ),
)

GTDB_R232: Final = GtdbSource(
    release="r232",
    fasta=RemoteFile(
        url="https://data.gtdb.ecogenomic.org/releases/release232/232.0/genomic_files_all/ssu_all_r232.fna.gz",
        sha256="dc5268786cc6e758283b03f851e9c7478608afda4e7bf31c6f533b429e18919c",
    ),
)

MIDORI2_GB272_COI: Final = Midori2Source(
    release="gb272",
    marker=Marker.COI,
    fasta=RemoteFile(
        url="https://www.reference-midori.info/download/Databases/GenBank272_2026-06-07/SINTAX_sp/uniq/MIDORI2_UNIQ_NUC_SP_GB272_CO1_SINTAX.fasta.gz",
        sha256="9f65efc6160d6c02d5f09bc1919212580c9ff3ea1e3dce1e86fa0de802cdf194",
    ),
)

UNITE_2025_02_19: Final = UniteSource(
    release="2025-02-19",
    archive=RemoteFile(
        url="https://s3.hpc.ut.ee/plutof-public/original/b02db549-5f04-43fc-afb6-02888b594d10.tgz",
        sha256="b07021870a6a14cc5d4f0a3fe81b5cf41a3c16fb71525c7265f973f2c6750458",
    ),
    member="sh_general_release_dynamic_s_all_19.02.2025.fasta",
)

GENBANK_CUTOFF: Final = "1900/01/01:2026/09/01[PDAT]"

GENBANK_RBCL: Final = GenBankSource(
    name="rbcl_2026-09-01",
    marker=Marker.RBCL,
    query=EntrezQuery(
        term=f"rbcL[Gene Name] AND Viridiplantae[Organism] AND 400:2000[SLEN] AND {GENBANK_CUTOFF}",
        filename="rbcl.fasta.gz",
    ),
)

GENBANK_MATK: Final = GenBankSource(
    name="matk_2026-09-01",
    marker=Marker.MATK,
    query=EntrezQuery(
        term=f"matK[Gene Name] AND Viridiplantae[Organism] AND 400:2500[SLEN] AND {GENBANK_CUTOFF}",
        filename="matk.fasta.gz",
    ),
)

PR2_5_1_1: Final = Pr2Source(
    release="5.1.1",
    fasta=RemoteFile(
        url="https://github.com/pr2database/pr2database/releases/download/v5.1.1/pr2_version_5.1.1_SSU_taxo_long.fasta.gz",
        sha256="b9cef85537b49bf54d1033ad38a6cce1875123fb9f7c7c8f442aadccd2bbb8f5",
    ),
)

EUKRIBO_2: Final = EukRiboSource(
    release="2",
    fasta=RemoteFile(
        url="https://zenodo.org/records/6896896/files/46346_EukRibo-02_full_seqs_2022-07-22.fas.gz",
        sha256="62b19ff1dd1add45d6b4aed338a93bbbb6467da6b3549bec8bc93da2ee7dcf72",
    ),
)

REFSEQ_16S: Final = RefSeqSource(
    name="16s",
    loci=(
        RefSeqLocus(
            fasta=RemoteFile(
                url="https://ftp.ncbi.nlm.nih.gov/refseq/TargetedLoci/Bacteria/bacteria.16SrRNA.fna.gz",
                sha256=None,
            ),
            domain="Bacteria",
            marker="16S",
        ),
        RefSeqLocus(
            fasta=RemoteFile(
                url="https://ftp.ncbi.nlm.nih.gov/refseq/TargetedLoci/Archaea/archaea.16SrRNA.fna.gz",
                sha256=None,
            ),
            domain="Archaea",
            marker="16S",
        ),
    ),
)

REFSEQ_FUNGI_18S: Final = RefSeqSource(
    name="fungi_18s",
    loci=(
        RefSeqLocus(
            fasta=RemoteFile(
                url="https://ftp.ncbi.nlm.nih.gov/refseq/TargetedLoci/Fungi/fungi.18SrRNA.fna.gz",
                sha256=None,
            ),
            domain="Eukaryota",
            marker="18S",
        ),
    ),
)

SILVA_144: Final = SilvaSource(
    release="144",
    fasta=RemoteFile(
        url="https://www.arb-silva.de/fileadmin/silva_databases/release_144/Exports/SILVA_144_SSURef_tax_silva_trunc.fasta.gz",
        sha256="2e8ce1f937ae81771c2b27d001f1bc5c09e09998fcca83a4a3b12c28df14b078",
    ),
)

DEFAULT_SOURCES: Final[tuple[DataSource, ...]] = (
    GTDB_R232,
    PR2_5_1_1,
    REFSEQ_16S,
    REFSEQ_FUNGI_18S,
    EUKRIBO_2,
    SILVA_144,
    MIDORI2_GB272_COI,
    UNITE_2025_02_19,
    GENBANK_RBCL,
    GENBANK_MATK,
)

DEFAULT_TAXONOMIES: Final[tuple[TaxonomySource, ...]] = (NCBI_TAXONOMY_2026_09,)
