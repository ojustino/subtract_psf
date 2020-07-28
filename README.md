### `subtract_psf`
- *generates mock reference and target stellar observations with NIRSpec's
Integral Field Unit (IFU)*
- *can use real spectral data to inject a companion brown dwarf into the target
observations*
- *aligns images and employs KLIP on reference observations to project the PSF
of each target's background star*
- *uses reference star differential imaging to attempt to subtract the
background star from each target image*
- *can visualize the results in a Jupyter notebook or export a directory of
FITS images for viewing in DS9*

#### To install locally:
```
git clone https://github.com/ojustino/subtract_psf
cd subtract_psf
pip install .
```
Add `-e` before the period in the final line if you intend to make changes to
the source code.

_(If you get import errors in Python even after trying the above, navigate to
  your cloned `subtract_psf` directory and try `python setup.py develop`
  instead.)_

#### Example usage:

See [`introduction.ipynb`](
  https://github.com/ojustino/subtract_psf/blob/master/notebooks/introduction.ipynb)
for a quick start and
[`inject_kappa_and.ipynb`](
  https://github.com/ojustino/subtract_psf/blob/master/notebooks/inject_kappa_and.ipynb)
for a worked example.

#### License:

This project uses the standard BSD-3 License, which is available in full [here](
  https://github.com/ojustino/subtract_psf/blob/master/LICENSE.txt).
