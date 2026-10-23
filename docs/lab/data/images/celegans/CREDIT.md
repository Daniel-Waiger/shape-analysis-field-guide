# Credit: BBBC010 sample images

The files in this folder are modified copies of images from **BBBC010**, the *C. elegans* infection
live/dead image set of the Broad Bioimage Benchmark Collection, <https://bbbc.broadinstitute.org/BBBC010>:
the brightfield images were contrast-stretched and converted to 8-bit PNG, and the foreground masks were
thresholded at 127 and written as 0 / 255 8-bit PNGs. The outlines and landmarks in `manifest.json` are
derived data, computed from BBBC010's per-worm ground-truth masks (see the end of this file). They were
made by `scripts/make_sample_images.py`; `manifest.json` records every processing step and the source file of each image.

Channel: the BBBC010 page says "Channel 1 = brightfield; channel 2 = GFP", but in the files `w2` is the
brightfield image (bright well, dark worms) and `w1` the low-signal fluorescence one, so these samples use
`w2`.

## License

**CC0 1.0** (<https://creativecommons.org/publicdomain/zero/1.0/>). BBBC010's page: "Copyright: CC0. To the extent
possible under law, the various contributors of the imagesets have waived all copyright and related or
neighboring rights to BBBC010. This work is published from: United States."

## Citation (as BBBC010 requests)

"We used the C.elegans infection live/dead image set version 1 provided by Fred Ausubel and available from the Broad Bioimage Benchmark Collection [Ljosa et al., Nature Methods, 2012]."

Ljosa, V., Sokolnicki, K. L. & Carpenter, A. E. (2012). Annotated high-throughput microscopy image sets
for validation. *Nature Methods* 9:637. The images were acquired for a screen in Fred Ausubel's lab at MGH
(Moy et al., *ACS Chemical Biology*, 2009).

## Files

| file | source | well | class |
|---|---|---|---|
| `B07_w2_brightfield.png` | `1649_1109_0003_Amp5-1_B_20070424_B07_w2_F3BB43AA-D215-44C1-B2FD-A89741699E5F.tif` (BBBC010_v2_images.zip) | B07 | live |
| `B07_foreground_mask.png` | `B07_binary.png` (BBBC010_v1_foreground.zip) | B07 | live |
| `A23_w2_brightfield.png` | `1649_1109_0003_Amp5-1_B_20070424_A23_w2_71C5704E-F745-4D29-B3B2-51216C02A45D.tif` (BBBC010_v2_images.zip) | A23 | dead |
| `A23_foreground_mask.png` | `A23_binary.png` (BBBC010_v1_foreground.zip) | A23 | dead |

The overlays in `manifest.json` (outlines and 12 landmarks per worm) were computed by `shapekit` from
BBBC010's per-worm ground-truth masks (`BBBC010_v1_foreground_eachworm.zip`).
