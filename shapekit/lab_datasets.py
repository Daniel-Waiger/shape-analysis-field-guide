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
``caveats`` are known problems with what a dataset can show (e.g. a
confound); ``run_lab.py`` writes them into the dataset's JSON and flags the
dataset in ``index.json`` (``has_caveat``), and the page shows them.
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
    caveats: tuple[str, ...] = ()  # known confounds / limits, shown prominently on the page

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        return self.loader()


def _load_celegans() -> tuple[np.ndarray, np.ndarray]:
    points, labels = load_celegans()
    return points.astype(float), labels.astype(int)


def _load_mef() -> tuple[np.ndarray, np.ndarray]:
    points, labels, _keys = load_mef_subset()
    return points.astype(float), labels.astype(int)


def _npz_path(filename: str, fetch_flag: str, local: bool = False) -> Path:
    """Where ``fetch_datasets.py`` put the file; raises ``DatasetNotFetched`` if it is not there."""
    base = Path(resolve_data_dir())
    path = (base / "_local" / filename) if local else (base / filename)
    if not path.exists():
        raise DatasetNotFetched(f"{path} is missing; run: python scripts/fetch_datasets.py {fetch_flag}")
    return path


def _npz_loader(filename: str, fetch_flag: str, local: bool = False) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
    def load() -> tuple[np.ndarray, np.ndarray]:
        with np.load(_npz_path(filename, fetch_flag, local), allow_pickle=False) as f:
            return f["points"].astype(float), f["labels"].astype(int)
    return load


def leaves_classes() -> tuple[str, ...]:
    """The species names stored in ``data/leaves_outlines.npz`` (in label order)."""
    with np.load(_npz_path("leaves_outlines.npz", "--leaves"), allow_pickle=False) as f:
        return tuple(str(c) for c in f["classes"])


def _load_leaves() -> tuple[np.ndarray, np.ndarray]:
    """Leaves: the class names travel with the npz; refuse to load if they differ from the registry."""
    with np.load(_npz_path("leaves_outlines.npz", "--leaves"), allow_pickle=False) as f:
        classes = tuple(str(c) for c in f["classes"])
        points, labels = f["points"].astype(float), f["labels"].astype(int)
    if classes != LEAVES.classes:
        raise ValueError(
            f"data/leaves_outlines.npz carries classes {classes}, the registry says {LEAVES.classes}; "
            "re-run scripts/fetch_datasets.py --leaves or update LEAVES.classes"
        )
    return points, labels


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
    question="Can a classifier tell these two groups of fibroblasts apart from outlines alone, "
             "and what is it actually seeing?",
    representation="pointcloud",
    classes=("wild-type", "LMNA−/−"),
    source=Source(
        name="Micropattern_MEF_LMNA_Image (kukionfr), as prepared in Shape Analysis 101",
        citation="Kyu Sang Han (2020), github.com/kukionfr/Micropattern_MEF_LMNA_Image; outlines by Hugger & Uhlmann (2021)",
        url="https://github.com/kukionfr/Micropattern_MEF_LMNA_Image",
        license="MIT © 2020 Kyu Sang Han; see data/mef_outlines_subset.LICENSE",
        redistributed=True,
    ),
    loader=_load_mef,
    caveats=(
        "Genotype is confounded with substrate, and with whatever else differs between the two source folders "
        "(culture dish, day, imaging session); the original repository does not say what its folder suffixes 1 "
        "and 2 mean. All 14 sampled wild-type fields of the tutorial's data come from the original folder "
        "Control/LMNA++1 (cells spreading freely on an unpatterned surface) and all 14 sampled LMNA−/− fields "
        "from Circle/LMNA--1 (cells confined to a circle micropattern): 28 of 28 fields, each matched at its own "
        "xy number, no other folder matching. [Inference] The other 86 + 86 fields come from the same two "
        "folders, so the two classes differ in genotype, in substrate and in everything else that differs "
        "between the folders, and the lab's MEF accuracy cannot be attributed to genotype.",
        "Evidence: for fields xy001, xy005, xy009, xy018, xy027, xy036, xy045, xy054, xy063, xy072, xy077, "
        "xy081, xy090 and xy100, the tutorial's outlines of each class were laid on the actin image with the "
        "same xy number in each of the 12 folders of the original repository (kukionfr/Micropattern_MEF_LMNA_Image "
        "@ 7cbe667: Control, Circle and Triangle; LMNA++ and LMNA--; suffixes 1 and 2; 100 fields per folder in "
        "its file tree). Edges: an Otsu mask of the actin image smoothed with σ = 1 px, its intensities clipped "
        "at their 99.5th percentile before thresholding. In the matching folder the outline points lie a median "
        "1.0 to 6.8 px from those edges; in every other folder 29.9 px or more. Without the clip one wild-type "
        "field (xy054) registers nowhere (647 px; a saturated bright spot takes the threshold), the other 27 give "
        "1.2 to 7.4 px. The ratio of the median actin intensity inside the outlines to outside agrees: 1.3 to "
        "6.8 in the matching folder, 0.9 to 1.1 in every other. A comparison on the same substrate (both "
        "genotypes on circles, or both unpatterned) is possible from the original repository but needs "
        "segmentation; it is an open item.",
    ),
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
        license="GPL-2 (R package; https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt); see data/brain_landmarks.LICENSE",
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
        license="GPL-2 (R package; https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt); see data/gorilla_landmarks.LICENSE",
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
    # speciesNumber order of the figshare CSV (label = speciesNumber - 1); the npz carries the
    # same names and _load_leaves() checks they agree.
    classes=(
        "Ulmus Minor", "Acer Platanoides", "Ulmus Laevis", "Quercus Robur", "Betula Pubescens",
        "Tilia Platyphyllos", "Salix Fragilis", "Populus Tremula", "Ulmus Glabra", "Sorbus Aucuparia",
        "Prunus Padus", "Tilia Cordata", "Populus Sp.", "Scandosorbus Intermedia", "Fagus Sylvatica",
    ),
    source=Source(
        name="Outlines of Swedish leaves (figshare 14551005), from the Swedish Leaf dataset",
        citation="Leroi, Marsland & Salili-James, Outlines of Swedish leaves, figshare DOI 10.6084/m9.figshare.14551005 "
                 "(CC BY 4.0, https://creativecommons.org/licenses/by/4.0/); outlines extracted by those authors from the "
                 "photographs of Söderkvist, O. J. O. (2001) Swedish Leaf dataset, MSc thesis, Linköping University "
                 "(page states no terms; a Zenodo record listing Söderkvist as creator, 19663514, is CC BY 4.0, uploader "
                 "not identified); modified here: resampled by arc length, see data/leaves_outlines.ATTRIBUTION.md",
        url="https://figshare.com/articles/dataset/Outlines_of_Swedish_leaves/14551005",
        license="CC BY 4.0",
        redistributed=True,
    ),
    loader=_load_leaves,
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
        license="CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/): no adapted material shared, "
                "no commercial use; fetched locally, only counts and accuracies published",
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
