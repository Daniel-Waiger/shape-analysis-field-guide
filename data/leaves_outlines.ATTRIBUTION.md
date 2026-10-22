# data/leaves_outlines.npz

Written by `scripts/fetch_datasets.py --leaves` on 2026-10-22T21:01:23Z.

## Source

- **Outlines of Swedish leaves**, Armand Leroi, Stephen Marsland, Arianna Salili-James. figshare, DOI 10.6084/m9.figshare.14551005.v1,
  <https://figshare.com/articles/dataset/Outlines_of_Swedish_leaves/14551005>.
  Licence: **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>).
  figshare's description: "A CSV containing the x and y coordinates of 440 leaf outlines extracted
  from images found in the Swedish Leaf Dataset. All outlines have been scaled to +-1.5 in x and y
  direction, and each outline has 150 points."
- The outlines were traced by those authors from the **Swedish Leaf dataset** of Oskar J. O. Söderkvist
  (2001), *Computer vision classification of leaves from Swedish trees*, MSc thesis, Linköping University
  (<https://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/>; the author's Zenodo copy,
  record 19663514, is CC BY 4.0; the Linköping page states no terms).
- File downloaded: `Contours_SwedishLeaves.csv`, md5 c7984a894d3cbc017805c6ec27a60f5d, sha256 a246e845bbb045b966e3c323d8cc87c46ed8e4af42584b64574abe9947f1af15.

## What the CSV contains

Columns `indexNumber, indexName, speciesNumber, speciesName, X, Y, pointOrder`: 440 outlines, each with a species number and
species name (15 species), an outline name (e.g. `l10nr001` = species 10, leaf 1) and ordered
`(X, Y)` points in `pointOrder`.  Species labels therefore **come with the file**; nothing was inferred.
The file actually holds 200 points x 440 outlines (not the 150 of the description); the first point is repeated at the
end in 440 of 440 outlines (a closed polygon written with its closing vertex).

## What was done to the data

1. Rows grouped by `indexNumber`, points sorted by `pointOrder`, the repeated closing vertex dropped.
2. Each closed outline resampled to **150 points equally spaced along its arc length**
   (`shapekit.contours.resample_by_arclength(points, 150, closed=True)`, linear interpolation along
   the polygon; the start point is the file's first point).  Coordinates are otherwise unchanged
   (still within ±1.5).
3. Stored as `points` (440, 150, 2) float64, `labels` (0–14, = speciesNumber − 1),
   `classes` (species names in speciesNumber order), `names` (indexName) and `species_number`.

| label | species | outlines |
|---|---|---|
| 0 | Ulmus Minor | 29 |
| 1 | Acer Platanoides | 29 |
| 2 | Ulmus Laevis | 30 |
| 3 | Quercus Robur | 30 |
| 4 | Betula Pubescens | 29 |
| 5 | Tilia Platyphyllos | 28 |
| 6 | Salix Fragilis | 30 |
| 7 | Populus Tremula | 30 |
| 8 | Ulmus Glabra | 30 |
| 9 | Sorbus Aucuparia | 30 |
| 10 | Prunus Padus | 28 |
| 11 | Tilia Cordata | 30 |
| 12 | Populus Sp. | 30 |
| 13 | Scandosorbus Intermedia | 27 |
| 14 | Fagus Sylvatica | 30 |
