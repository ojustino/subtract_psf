#!/usr/bin/env python3
import argparse
import numpy as np
import os
import pickle
import sys

from astropy.io import fits
from klip_create import KlipCreate

# toy examples
# ./make_img_dirs.py -n ''
# ./make_img_dirs.py -n 'testy' -i 3 -dx '.02 .01 -.01' -dy '.03, .05, .03'

parser = argparse.ArgumentParser()

# take in a comma/space delimited str, grab the floats, and make an array
read_as_arr = (lambda arg:
               np.array([float(pos) for pos in arg.split(' ') if len(pos) > 0]
                        if arg.find(',') < 0 # space delimited
                        else [float(pos) for pos # comma delimited
                              in arg.replace(' ','').split(',')]))

# define acceptable arguments for directory creation
parser.add_argument('-n', '--name',
                    help="base name for the directories you're creating "
                    "(e.g. base0, base1, etc.) [str]",
                    type=str, dest='base_name')
parser.add_argument('-i', '--iterations',
                    help='the number of directories to create [int, up to 16]',
                    type=int, dest='iterations')
parser.add_argument('-dx', '--dither_x',
                    help='comma/space delimited string of x-direction '
                    'dither cycle positions [floats in arcsec]',
                    type=read_as_arr, dest='dither_x')
parser.add_argument('-dy', '--dither_y',
                    help='comma/space delimited string of y-direction '
                    'dither cycle positions [floats in arcsec]',
                    type=read_as_arr, dest='dither_y')
parser.add_argument('-pterr', '--pointing_error',
                    help='whether to add overall pointing error to the '
                    'science and reference images [bool, default True]',
                    type=bool, dest='pnt_err')
parser.add_argument('-os', '--oversample',
                    help="The factor beyond the detector's pixel count by "
                    'which to sample the scene [int, default 4]',
                    type=int, dest='oversample')
parser.add_argument('-sl', '--temp_slices',
                    help='The number of slices to include per data cube. '
                    'Save computation time by choosing 6 [int, default ~1530]',
                    type=int, dest='temp_slices')
args = parser.parse_args()

# check that arguments match what we expect
try:
    # mandate that at least a base name and number of iterations are included
    if (args.base_name is None) or (args.iterations is None):
        raise ValueError('A name and number of iterations are required')

    # check lengths of dither arrays and which axes were included
    if (args.dither_x is not None) + (args.dither_y is not None) == 2:
        if len(args.dither_x) != len(args.dither_y):
            raise ValueError('Dither lists must be the same length')

        dithers = np.stack((args.dither_x, args.dither_y), axis=-1)

    elif (args.dither_x is not None) + (args.dither_y is not None) == 1:
        raise ValueError('Include either both dither axes or neither of them')

    else: # == 0
        dithers = None

    # set default oversampling if not specified by user
    if args.oversample is None:
        args.oversample = 4

    # set default pointing error if not specified by user
    if args.pnt_err is None:
        args.pnt_err = True

except Exception as err:
    parser.print_help()
    raise err

# create the directories
print(args)
KlipCreate(args.base_name, num_dir=args.iterations,
           dithers=dithers, pointing_error=args.pnt_err,
           oversample=args.oversample, temp_slices=args.temp_slices)

# For future reference, save a file containing the call made to terminal
def parent_dir(base_path):
    return os.path.realpath(os.path.join(base_path, os.pardir)) + '/'

# first, concatenate the arguments into one string
orig_call = './'
arg_list = [arg for arg in sys.argv]
for i, arg in enumerate(arg_list):
    if i == 0:
        orig_call += os.path.basename(arg) + ' '
    else:
        orig_call += arg + ' '

# then, write and save a file with the concatenated argument string
new_dirs = [dir for dir in os.listdir(parent_dir(args.base_name))
            if dir.startswith(os.path.basename(args.base_name))]

for dir in new_dirs:
    with open(parent_dir(args.base_name) + dir
              + '/original_call.pkl', 'wb') as file:
        pickle.dump(orig_call, file)

# Finally, remove extraneous information from the data cubes' FITS headers
def downsize_fits_headers(dir_name):
    '''
    WebbPSF's calc_datacube produces headers contain a 30 line readout for each
    slice in each extension of the output HDUList. For a 2 extension HDUList
    with 3,000 slice data cubes, that's ~200,000 lines. This information isn't
    needed for our purposes and demonstrably slows the loading process in DS9
    (and potentially KlipRetrieve), so we seek to eliminate it.

    We take advantage of its predictable length, key ('HISTORY') and position
    in the list of headers (last) to slice it out in each HDUList extension and
    save the downsized result.
    '''
    HISTORY_OUTPUT_LEN = 30
    new_dir_name = dir_name + ('/' if not dir_name.endswith('/') else '')

    # find all FITS files in the directory
    files = sorted([f for f in os.listdir(new_dir_name)
                if f.endswith('.fits')])

    # change each FITS file's headers, one by one
    print(f"*****\nin {new_dir_name}")
    for fl in files:
        print(f"editing {fl}")
        with fits.open(new_dir_name + fl, mode='update') as hdul:
            # count number of slices per cube, minus one
            # (keeps one slice's worth of info to summarize the rest)
            slices = hdul[0].data.shape[0] - 1

            # downsize each HDUList extension's data cubes
            for ext in hdul:
                # check whether a downsize is actually needed
                if len(ext.header['HISTORY']) <= HISTORY_OUTPUT_LEN:
                    print(f"{ext.name} is already short!")
                    continue
                else:
                    print(f"shortening {ext.name}...")

                # 'HISTORY' entries come last, so slice until first one ends
                ext.header = ext.header[:-(HISTORY_OUTPUT_LEN * slices)]

                # add context for the slice's info that remains
                ext.header['HISTORY'] = ('(above process is repeated '
                                         'for each cube slice)')

            # write the changes back to the original file
            hdul.flush()

generated_dirs = [o.name for o in os.scandir()
                  if o.name.startswith(args.base_name) and o.is_dir()]
for dir_name in generated_dirs:
    downsize_fits_headers(dir_name)

# not currently needed
#def guarantee_naming(base_name):
#    '''
#    os.path.basename gives different results depending on whether its argument
#    has a trailing slash or not. This function uses os.path.realpath to ensure
#    the same output no matter which scenario is the case in args.base_name.
#    '''
#    return os.path.basename(os.path.realpath(base_name))
