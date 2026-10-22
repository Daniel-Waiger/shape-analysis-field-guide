"""Registry of the Phenotype Lab datasets (chapter 10).

Each entry says what the dataset is, where it came from and under which
licence, and how to load it.  ``load()`` returns ``(points, labels)`` with
points ``(K, N, 2)`` float, ``(x, y)`` columns, and labels ``(K,)`` int in
``range(len(classes))``.  ``representation`` is one of:

- ``"landmarks"``: ordered points in correspondence (every shape has the
  same number of points and point k means the same anatomical place);
- ``"outline"``: an ordered closed outline (correspondence by arc length);
- ``"pointcloud"``: unordered points (order carries no meaning).

A dataset whose licence forbids redistribution is never stored in the
repository: its loader reads from ``data/_local/`` (git-ignored) and raises
``DatasetNotFetched`` with the fetch command when the files are absent.
See LAB_SPEC.md and ATTRIBUTION.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .datasets import load_celegans, load_mef_subset, resolve_data_dir


class DatasetNotFetched(RuntimeError):
    """Raised when a fetch-locally dataset has not been downloaded yet."""


@dataclass(frozen=True)
class Source:
    name: str
    citation: str
    url: str
    license: str
    redistributed: bool


@dataclass(frozen=True)
class LabDataset:
    id: str
    title: str
    question: str
    representation: str
    classes: tuple[str, ...]
    source: Source
    loader: Callable[[], tuple[np.ndarray, np.ndarray]] = field(repr=False)
    closed: bool = False          # for outlines: closed curve?
    shapes_shown: bool = True     # False when the licence forbids publishing derived coordinates
    fetch_command: str | None = None

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        return self.loader()


def _load_celegans() -> tuple[np.ndarray, np.ndarray]:
    points, labels = load_celegans()
    return points.astype(float), labels.astype(int)


def _load_mef() -> tuple[np.ndarray, np.ndarray]:
    points, labels, _keys = load_mef_subset()
    return points.astype(float), labels.astype(int)


def _npz_loader(filename: str, fetch_flag: str, local: bool = False) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
    def load() -> tuple[np.ndarray, np.ndarray]:
        base = Path(resolve_data_dir())
        path = (base / "_local" / filename) if local else (base / filename)
        if not path.exists():
            raise DatasetNotFetched(f"{path} is missing; run: python scripts/fetch_datasets.py {fetch_flag}")
        with np.load(path, allow_pickle=False) as f:
            return f["points"].astype(float), f["labels"].astype(int)
    return load


CELEGANS = LabDataset(
    id="celegans",
    title="C. elegans, live vs dead",
    question="Does body shape alone say whether a worm is alive?",
    representation="landmarks",
    classes=("dead", "live"),
    source=Source(
        name="BBBC010 (Broad Bioimage Benchmark Collection), as prepared in Shape Analysis 101",
        citation="Ljosa, Sokolnicki & Carpenter (2012) Nat. Methods 9:637; image set by Fred Ausubel; "
                 "landmarks by Hugger & Uhlmann (2021)",
        url="https://bbbc.broadinstitute.org/BBBC010",
        license="CC0 (images); BSD-3 (tutorial-prepared landmarks)",
        redistributed=True,
    ),
    loader=_load_celegans,
)

MEF = LabDataset(
    id="mef",
    title="Mouse fibroblasts, wild-type vs LMNA−/−",
    question="Can a cell's genotype be read from its outline alone?",
    representation="pointcloud",
    classes=("wild-type", "LMNA−/−"),
    source=Source(
        name="Micropattern_MEF_LMNA_Image (kukionfr), as prepared in Shape Analysis 101",
        citation="Kyu Sang Han (2020), github.com/kukionfr/Micropattern_MEF_LMNA_Image; outlines by Hugger & Uhlmann (2021)",
        url="https://github.com/kukionfr/Micropattern_MEF_LMNA_Image",
        license="MIT © 2020 Kyu Sang Han",
        redistributed=True,
    ),
    loader=_load_mef,
)

BRAIN = LabDataset(
    id="brain",
    title="Brain landmarks, schizophrenia vs control",
    question="Do 13 midsagittal brain landmarks separate 14 patients from 14 controls?",
    representation="landmarks",
    classes=("control", "schizophrenia"),
    source=Source(
        name="`schizophrenia` dataset of the R package shapes (I. L. Dryden)",
        citation="Bookstein, F. L. (1996) Bull. Math. Biol. 58:313–365; Dryden & Mardia (2016)",
        url="https://cran.r-project.org/package=shapes",
        license="GPL-2 (R package); see data/brain_landmarks.LICENSE",
        redistributed=True,
    ),
    loader=_npz_loader("brain_landmarks.npz", "--brain"),
    fetch_command="python scripts/fetch_datasets.py --brain",
)

GORILLA = LabDataset(
    id="gorilla",
    title="Gorilla skulls, female vs male",
    question="Do 8 skull landmarks tell female from male gorillas?",
    representation="landmarks",
    classes=("female", "male"),
    source=Source(
        name="`gorf.dat` and `gorm.dat` of the R package shapes (I. L. Dryden)",
        citation="O'Higgins & Dryden (1993); Dryden & Mardia (2016), data from Paul O'Higgins",
        url="https://cran.r-project.org/package=shapes",
        license="GPL-2 (R package); see data/gorilla_landmarks.LICENSE",
        redistributed=True,
    ),
    loader=_npz_loader("gorilla_landmarks.npz", "--gorilla"),
    fetch_command="python scripts/fetch_datasets.py --gorilla",
)

LEAVES = LabDataset(
    id="leaves",
    title="Leaf outlines, species",
    question="Can a leaf's species be read from its outline?",
    representation="outline",
    classes=(),  # filled by fetch_datasets.py from the data; see lab_datasets_meta.json
    source=Source(
        name="Outlines of Swedish leaves (figshare 14551005), from the Swedish Leaf dataset",
        citation="Leroi, Marsland & Salili-James (2022), figshare; Söderkvist, O. J. O. (2001) MSc thesis, Linköping University",
        url="https://figshare.com/articles/dataset/Outlines_of_Swedish_leaves/14551005",
        license="CC BY 4.0",
        redistributed=True,
    ),
    loader=_npz_loader("leaves_outlines.npz", "--leaves"),
    closed=True,
    fetch_command="python scripts/fetch_datasets.py --leaves",
)

RBC = LabDataset(
    id="rbc",
    title="Red blood cells, sickle-cell disease",
    question="Does outline shape separate circular, elongated and other erythrocytes?",
    representation="outline",
    classes=("circular", "elongated", "other"),
    source=Source(
        name="erythrocytesIDB v2 (Universitat de les Illes Balears), Zenodo 18299474",
        citation="Marrero Fernández et al.; related paper IEEE J. Biomed. Health Inform. 19(4):1514–1525 (2015)",
        url="https://zenodo.org/records/18299474",
        license="CC BY-NC-ND 4.0: no derived data may be redistributed; fetch locally",
        redistributed=False,
    ),
    loader=_npz_loader("rbc_outlines.npz", "--rbc", local=True),
    closed=True,
    shapes_shown=False,
    fetch_command="python scripts/fetch_datasets.py --rbc",
)

DATASETS: dict[str, LabDataset] = {d.id: d for d in (CELEGANS, MEF, BRAIN, GORILLA, LEAVES, RBC)}


def get(dataset_id: str) -> LabDataset:
    try:
        return DATASETS[dataset_id]
    except KeyError:
        raise KeyError(f"unknown lab dataset {dataset_id!r}; known: {sorted(DATASETS)}") from None
