# Third-party attribution

## Sinkhorn Transport

`Software/fotmir_core/Transport.py` and `Experiments/fotmir_core/Transport.py` are Python adaptations of **Sinkhorn Transport**, by **Marco Cuturi**, copyright 2013, 2014.

The original author's [source file](https://marcocuturi.net/Code/sinkhornTransport.m), linked from the [project page](https://marcocuturi.net/SI.html), carries an express Mozilla Public License 1.1 notice. This package distributes the adapted files under MPL 1.1, with its full text in [Licenses/MPL-1.1.txt](Licenses/MPL-1.1.txt). The adapted source is supplied in the two paths above. The Apache 2.0 license for the surrounding FOTMIR project applies subject to this file-level license.

The solver computes entropically regularized optimal transport by alternating diagonal scalings. `Sinkhorn.py` calls it during registration. Scientific reference: Marco Cuturi, *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*, Advances in Neural Information Processing Systems 26, 2013.

Adaptation history: the Python implementations retained from the 2026-10-15 package translate the supplied solver into NumPy, preserve its two histogram modes and counter convention, and add shape, mass, finite-value and numerical-error handling. On 2026-10-16, the upstream license notice, full license text and this provenance record were added to this distribution. These licensing edits preserve the numerical implementation.

## Installed dependencies

Dependency versions are specified in the component `requirements.txt` files. Attribution and license terms supplied with NumPy, SciPy, SimpleITK, Matplotlib, pandas, Open3D, ANTsPyX and threadpoolctl apply to their respective installed distributions.

## RIRE example images

`Software/examples/data/moving.nii.gz` and `fixed.nii.gz` contain the CT and rectified PD-weighted MRI volumes of RIRE case `patient_001`, converted from local MetaImage files to compressed NIfTI. Voxel data and geometry are preserved. Source identifiers and conversion details are recorded in `Software/examples/data/source.json`.

Attribution: Retrospective Image Registration Evaluation project; J. Michael Fitzpatrick, Vanderbilt University; National Institutes of Health Project 8R01EB002124-03. The [official source page](https://rire.insight-journal.org/download_data) specifies [Creative Commons Attribution 3.0 United States](https://creativecommons.org/licenses/by/3.0/us/). The accompanying historical notice, which says “3.0 Unported”, is preserved verbatim at `Software/examples/RIRE_LICENSE.txt`. These image files retain their dataset license independently of the FOTMIR project license.
