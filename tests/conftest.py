from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Protocol

import pytest

from taxonomy_classifier.data.build import BuildConfig, DataLayout, build_dataset
from taxonomy_classifier.data.filters import FilterConfig
from taxonomy_classifier.data.harmonize import BackboneIndex
from taxonomy_classifier.data.kingdom import Kingdom
from taxonomy_classifier.data.records import SequenceRecord
from taxonomy_classifier.data.remote import RemoteFile
from taxonomy_classifier.data.sources.eukribo import EukRiboSource
from taxonomy_classifier.data.sources.gtdb import GtdbSource
from taxonomy_classifier.data.sources.pr2 import Pr2Source
from taxonomy_classifier.data.sources.refseq import RefSeqLocus, RefSeqSource
from taxonomy_classifier.data.sources.silva import SilvaSource
from taxonomy_classifier.data.split import SplitConfig
from taxonomy_classifier.data.taxonomy import Lineage, Rank, Taxon
from taxonomy_classifier.training.oversampling import OversamplingConfig, oversample_train

if TYPE_CHECKING:
    from taxonomy_classifier.data.sources.base import DataSource

FIXTURES_DIR = Path(__file__).parent / "fixtures"

BACTERIA = Lineage.domain_only("Bacteria")

BACTERIA_TAXA = (Taxon(name="Bacteria", rank=Rank.DOMAIN),)


class RecordFactory(Protocol):
    def __call__(
        self,
        *,
        sequence: str = ...,
        lineage: Lineage = ...,
        source_lineage: tuple[Taxon, ...] = ...,
        source: str = ...,
        kingdom: Kingdom | None = ...,
    ) -> SequenceRecord: ...


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def make_record() -> RecordFactory:
    def factory(
        *,
        sequence: str = "ACGT" * 250,
        lineage: Lineage = BACTERIA,
        source_lineage: tuple[Taxon, ...] = BACTERIA_TAXA,
        source: str = "test",
        kingdom: Kingdom | None = Kingdom.BACTERIA,
    ) -> SequenceRecord:
        return SequenceRecord(
            source=source,
            accession="AB000001",
            sequence=sequence,
            source_lineage=source_lineage,
            lineage=lineage,
            kingdom=kingdom,
        )

    return factory


def _fixture_file(name: str) -> RemoteFile:
    return RemoteFile(url=f"https://example.org/fixtures/{name}", sha256=None)


GTDB_TEST = GtdbSource(release="test", fasta=_fixture_file("gtdb_ssu.fna"))

PR2_TEST = Pr2Source(release="test", fasta=_fixture_file("pr2_ssu.fasta"))

REFSEQ_TEST = RefSeqSource(
    name="16s",
    loci=(
        RefSeqLocus(fasta=_fixture_file("refseq_bacteria.fna"), domain="Bacteria", marker="16S"),
        RefSeqLocus(fasta=_fixture_file("refseq_archaea.fna"), domain="Archaea", marker="16S"),
    ),
)

REFSEQ_FUNGI_TEST = RefSeqSource(
    name="fungi_18s",
    loci=(RefSeqLocus(fasta=_fixture_file("refseq_fungi.fna"), domain="Eukaryota", marker="18S"),),
)

EUKRIBO_TEST = EukRiboSource(release="test", fasta=_fixture_file("eukribo_ssu.fas"))

SILVA_TEST = SilvaSource(release="test", fasta=_fixture_file("silva_ssu.fasta"))


@pytest.fixture
def gtdb_source() -> GtdbSource:
    return GTDB_TEST


@pytest.fixture
def pr2_source() -> Pr2Source:
    return PR2_TEST


@pytest.fixture
def refseq_source() -> RefSeqSource:
    return REFSEQ_TEST


@pytest.fixture
def refseq_fungi_source() -> RefSeqSource:
    return REFSEQ_FUNGI_TEST


@pytest.fixture
def eukribo_source() -> EukRiboSource:
    return EUKRIBO_TEST


@pytest.fixture
def silva_source() -> SilvaSource:
    return SILVA_TEST


@pytest.fixture
def sources() -> tuple[DataSource, ...]:
    return (SILVA_TEST, REFSEQ_TEST, GTDB_TEST, PR2_TEST)


@pytest.fixture
def backbone(fixtures_dir: Path) -> BackboneIndex:
    index = BackboneIndex()

    for source in (GTDB_TEST, PR2_TEST):
        index.add_all(
            outcome
            for outcome in source.read(fixtures_dir, index)
            if isinstance(outcome, SequenceRecord)
        )

    return index


@pytest.fixture
def built_layout(sources: tuple[DataSource, ...], fixtures_dir: Path, tmp_path: Path) -> DataLayout:
    layout = DataLayout(root=tmp_path / "built")

    for source in sources:
        raw_dir = layout.raw_dir(source)
        raw_dir.mkdir(parents=True)

        for remote in source.files:
            shutil.copy(fixtures_dir / remote.filename, raw_dir / remote.filename)

    build_dataset(
        sources,
        layout,
        config=BuildConfig(
            filters=FilterConfig(min_length=10, max_length=100),
            split=SplitConfig(val_fraction=0.3, test_fraction=0.1, seed=3),
        ),
    )
    oversample_train(
        layout.dataset_path,
        layout.synthetic_path,
        layout.augment_report_path,
        config=OversamplingConfig(floor=3),
    )

    return layout
