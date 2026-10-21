# FOTMIR

A Fast Optimal Transport-Based Tool for Multimodal Medical Image Registration.

Package version: **1.0.0**. Keep the package directory structure intact; `VERSION` is the shared version source used by the application and experiment outputs.

| Directory | Contents | Start here |
| --- | --- | --- |
| `Software/` | English desktop interface, command-line application, registration core and a real RIRE head CT-MRI example | [Software README](Software/README.md) |
| `Experiments/` | Dataset preparation, case manifests, experimental settings and FOTMIR/ICP/FGR/ANTs runs | [Experiments README](Experiments/README.md) |
| `Results/` | Original case-level measurements, summary tables and metric plots | [Results README](Results/README.md) |

## Install and launch

Use Python with Tkinter for the desktop interface. Run from this directory:

```sh
python -m pip install -r Software/requirements.txt
python Software/doctor_registration_gui.py
```

Select the moving image, fixed image, modalities and an output directory, then start registration. The application saves the registered volume and six separate grayscale images: fixed and registered views along each of the three image-grid directions. The [Software README](Software/README.md) also provides a command-line example.

## Experimental results

The result tables cover 45 RIRE, 8 AbdomenMRCT and 30 LungCT image pairs, with FOTMIR, ICP, FGR and rigid ANTs results for each pair. Open `Results/figures/` to view the metric images. Definitions, units, aggregation and timing scope are described in `Results/README.md`.

## License and citation

FOTMIR project code is distributed under the Apache License 2.0 in [LICENSE](LICENSE). The adapted Sinkhorn solver files `Software/fotmir_core/Transport.py` and `Experiments/fotmir_core/Transport.py` are distributed under the Mozilla Public License 1.1 in [Licenses/MPL-1.1.txt](Licenses/MPL-1.1.txt). Attribution and adaptation details are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The RIRE example images retain their dataset license; see [example attribution](Software/examples/README.md#attribution-and-license).

Software citation metadata is provided in [CITATION.cff](CITATION.cff).

## Version and support

Use `v1.0.0` as the tag when publishing this package. The project repository is [Medical-Image-Registration](https://github.com/USTC-DIAG-LAB/Medical-Image-Registration). To identify the installed package, run `python Software/one_click_registration.py --version`.

The authors commit to maintaining FOTMIR for at least two years after publication of the accompanying Frontiers of Computer Science article. Report installation problems, reproducibility questions and software issues through [GitHub Issues](https://github.com/USTC-DIAG-LAB/Medical-Image-Registration/issues). The repository will link to the published article when its final DOI is assigned.
