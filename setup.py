import numpy as np
import sys
import warnings

from setuptools import setup
from io import StringIO

# enable stdout to be saved as a variable to learn more about numpy's config.
# (see https://kite.com/python/answers/how-to-redirect-print-output-to-a-variable-in-python)
old_stdout = sys.stdout
new_stdout = StringIO()
sys.stdout = new_stdout

# save that output, then change stdout back to normal
np.show_config()
output = str.splitlines(new_stdout.getvalue())
sys.stdout = old_stdout

# check for mkl integration, throw error if not present
split_output = [line.lstrip() for line in output if len(line)]
has_mkl = [line for line in split_output
           if line.startswith('libraries =') and line.find('mkl')]

if len(has_mkl) == 0:
    raise ImportError('Please note that you need to be using an mkl-assisted '
                      'version of numpy for synthetic PSF generation in '
                      '`make_img_dirs.py` and `create_images.py` to work '
                      'properly. (Installing an Anaconda distribution gets you '
                      'that, but note that mkl only works with '
                      "Intel-compatible processors.)")

setup(name='subtract_psf',
      version='0.1',
      description=('Simulate PSF subtraction with '
                   'synthetic NIRSpec IFU observations.'),
      url='https://github.com/ojustino/subtract_psf',
      author='ojustino',
      author_email='ojustino@users.noreply.github.com',
      license='BSD-3',
      keywords=['Astronomy'],
      install_requires=['astropy>=3.0.0', 'numpy>=1.13.0', 'matplotlib>=2.0.0',
                        'poppy>=0.9.0', 'webbpsf>=0.9.0'],
      python_requires='>=3.5',
      zip_safe=False)
