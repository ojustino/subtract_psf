#!/usr/bin/env python3
import numpy as np
import os
import pickle

from astropy.io import fits
from .align_images import AlignImages
from .inject_images import InjectCompanion
from .my_warnings import warnings
from .subtract_images import SubtractImages
from .visualize_images import VisualizeImages


KR_INJECT_DOCSTR = ('''
        Ingests finely sampled spectra for a star/companion pair, bins them
        down, then injects a synthetic companion scaled by their contrast ratio
        into an already aligned set of data cubes. The result simulates how a
        real, polychromatic observation of a companion at a given arcsecond
        separation might look. The first argument should be the HDUList of
        data cubes you'd like to inject. Typically this will be
        `self.stackable_cubes`; try a `PreInjectImages()` instance if you're
        trying to use `self.data_cubes` instead.
        ''')

PII_INJECT_DOCSTR = ('''
        Ingests finely sampled spectra for a star/companion pair, bins them
        down, then injects a synthetic companion scaled by their contrast ratio
        into a not-yet-aligned set of data cubes. The result simulates how a
        real, polychromatic observation of a companion at a given arcsecond
        separation might look. The first argument should be the HDUList of
        data cubes you'd like to inject. For `PreInjectImages()` class
        instances, this will typically be `self.data_cubes`.
        ''')


class ReadKlipDirectoryBase:
    '''
    This class is meant to be inherited by a KlipRetrieve() or PreInjectImages()
    class instance, not used individually.

    Reads in images and associated information (pointings, errors, etc.) from a
    KlipCreate-made directory of FITS images. Also provides methods for
    exporting an edited HDUList of data cubes to a new directory and for quickly
    copying an HDUList.
    '''
    def __init__(self, dir_name):
        # save directory name and original call to make_img_dirs.py
        self.dir_name = dir_name + ('/' if not dir_name.endswith('/') else '')
        self.terminal_call = self._get_og_call(self.dir_name)

        # retrieve data cubes from indicated directory
        data_cubes = self._retrieve_data_cubes(dir_name)

        # add stellar location information to headers of target data cubes
        self.data_cubes = self._locate_target_stars(data_cubes)

    def _pklcopy(self, obj):
        '''
        A faster alternative to `copy.deepcopy(obj)` that also remains accurate
        in some cases where `obj.copy()` doesn't for list-like objects.

        See https://stackoverflow.com/a/29385667 for more info -- deepcopy might
        still be better in some cases.
        '''
        return pickle.loads(pickle.dumps(obj, -1))

    def _retrieve_data_cubes(self, dir_name):
        '''
        Return a list of related data cubes (each list index is a different
        dither position, each cube is indexed by wavelength)
        '''
        # import attributes for this set of images
        with open(dir_name + '/attr.pkl', 'rb') as file:
            attrs = pickle.load(file)

        # add those attributes to this class instance
        for key, val in attrs.items():
            setattr(self, key, val)

        # check if this directory was created via another KlipRetrieve instance
        # (affects which HDUList extension is saved later on)
        try:
            if self._derivative == True:
                ex = 0
        except AttributeError:
            ex = 1

        # import the HDULists of data cubes
        img_files = sorted([f for f in os.listdir(dir_name)
                            if f.endswith('.fits')])

        data_cubes = fits.HDUList(fits.ImageHDU())
        for f in img_files:
            with fits.open(dir_name + '/' + f) as hdul:
                data_cubes.append(fits.ImageHDU(hdul[ex].data, hdul[ex].header))

        # name images by their set and observation number
        data_cubes = data_cubes[1:] # remove PrimaryHDU
        for i, img in enumerate(data_cubes[:len(data_cubes)//2]):
            img.name = f"REFERENCE{i}"

        for i, img in enumerate(data_cubes[len(data_cubes)//2:]):
            img.name = f"TARGET{i}"

        return data_cubes

    def _locate_target_stars(self, cube_list):
        '''
        Label the target star locations in `self.data_cubes` so it can be
        tracked through the alignment process, and not take up space in the
        `self._generate_contrasts()` loop.
        '''
        cube_list = self._pklcopy(cube_list)
        pix_len = .1

        # translate arcsecond coordinates to pixel coordinates based on the size
        # of images in the data cubes' slices
        pix_center = [(size - 1) / 2 for size
                      in cube_list[-1].shape[-2:]]
        # first index of cube_list doesn't matter

        star_locs = pix_center + self.draws_sci[:, ::-1] / pix_len
        # img_center is y, x and draws_* is x, y; one of them had to be flipped

        # update target data cubes with their corresponding star location info
        for im, cube in enumerate(cube_list[len(self.positions):]):
            # should ref cubes have PIXSTARY?
            cube.header['PIXSTARY'] = (star_locs[im, 0], 'brightest pixel in '
                                       f"TARGET{im}, Y direction")
            cube.header['PIXSTARX'] = (star_locs[im, 1], 'brightest pixel in '
                                       f"TARGET{im}, X direction")

        return cube_list

    def export_to_new_dir(self, cube_list, new_dir_name, overwrite=False):
        '''
        Create a new directory of data cubes that are still usable by
        KlipRetrieve.

        Argument `cube_list` is the HDUList that will be saved, extension-by-
        extension.

        Argument `new_dir_name` is the string path to your desired directory.

        Argument `overwrite` is a boolean that controls whether or not the
        method is allowed to overwrite an existing directory whose path matches
        `new_dir_name`.
        '''
        # check if all observations are present
        if len(cube_list) != len(self.positions) * 2:
            if len(cube_list) == len(self.positions):
                query = (' Did you forget to include the reference images? '
                         "If you're trying to save `self.injected_cubes`, try "
                         'stackable_cubes[:len(positions)] + injected_cubes or '
                         'data_cubes[:len(positions)] + injected_cubes -- but '
                         'make sure the sets have cubes with matching shapes.')
            else:
                query = ''
            raise ValueError('Some images are missing.' + query)

        # check alignment style (will not exist if called from PreInjectImages)
        try:
            align_style = self.align_style
        except AttributeError:
            align_style = None

        # for 'theoretical'-aligned data cubes, check that all refs have
        # the same shape (targets can vary, so their shapes don't tell much)
        if (len(np.unique([cb.shape for cb in cube_list[:len(self.positions)]],
                          axis=0)) != 1
            and align_style == 'theoretical'):
            warnings.warn('Not all reference cubes in this list have the same '
                          'shape. Is that intentional? You will have problems '
                          'using KlipRetrieve with this new directory.')
        # for others ('empirical2' and None), check if all have the same shape
        elif (len(np.unique([cb.shape for cb in cube_list], axis=0)) != 1
              and align_style == 'empirical2'):
            warnings.warn('Not all data cubes in this list have the same '
                          'shape. Is that intentional? You will have problems '
                          'using KlipRetrieve with this new directory.')

        # create the new directory
        os.makedirs(new_dir_name, exist_ok=overwrite)

        # save the FITS files
        new_dir_name += ('/' if not new_dir_name.endswith('/') else '')

        for i in range(len(cube_list)):
            fits.writeto(new_dir_name +
                         f"{'ref' if i < len(self.positions) else 'sci'}"
                         f"_image{i % len(self.positions)}.fits",
                         cube_list[i].data, cube_list[i].header,
                         overwrite=overwrite)

        # save attrs.pkl, adding a new attribute to signify that this new
        # directory is a derivative of a KlipRetrieve instance
        with open(self.dir_name + 'attr.pkl', 'rb') as file:
            attrs = pickle.load(file)
            attrs['_derivative'] = True
        with open(new_dir_name + 'attr.pkl', 'wb') as file:
            pickle.dump(attrs, file)

        # finally, copy over original_call.pkl
        with open(self.dir_name + 'original_call.pkl', 'rb') as file:
            call = pickle.load(file)
        with open(new_dir_name + 'original_call.pkl', 'wb') as file:
            pickle.dump(call, file)

    def export_subtracted_cubes(self, new_dir_name, overwrite=False):
        '''
        A convenience function for exporting `self.subtracted_cubes`, an
        HDUList of injected and then subtracted data cubes, to a new,
        KlipRetrieve() compatible directory. (`self.stackable_cubes` is made of
        the reference cubes from `self.stackable_cubes` and the difference
        between the target cubes in `self.injected_cubes` and
        `self.stackable_cubes`.)

        Argument `new_dir_name` is the string path to your desired directory.

        Argument `overwrite` is a boolean that controls whether or not the
        method is allowed to overwrite an existing directory whose path matches
        `new_dir_name`.
        '''
        self.export_to_new_dir(self.subtracted_cubes, new_dir_name, overwrite)

    def _get_og_call(self, dir_name):
        '''
        Return the original call to `make_img_dirs.py` made in either
        `gen_dirs.condor` or the terminal to generate the "observations" in
        this instance's `self.data_cubes` HDUList.
        '''
        with open(dir_name + 'original_call.pkl', 'rb') as file:
            call = pickle.load(file)

        return call


class KlipRetrieve(ReadKlipDirectoryBase, AlignImages,
                   SubtractImages, VisualizeImages):
    # inheritance order matters
    '''
    Use the KLIP algorithm to perform PSF subtraction on data cubes from a
    directory created by KlipCreate().

    This class inherits from AlignImages() to get methods that do the work of
    removing offsets from all data cube images and aligning the reference and
    target images. (This step requires class attributes created by reading from
    the `attr.pkl` file, so the data cubes alone are not enough. There also
    must be an equal number of reference and target images.)

    It also inherits from SubtractImages() to create KLIP bases for the target
    images from the reference images, carry out PSF subtraction in each slice
    of each target image, save the results, and save information about contrast
    and separation. See self.plot_subtraction() and self.plot_contrasts() for
    details on how to visualize these results.

    (To see help for these parent classes when working with an instance of
    KlipRetrieve, look at the output of `type(INSTANCE_NAME).__mro` and call
    `help` on the indices you're curious about.)

    Arguments:
    ----------
    dir_name : str, required
        The path to your directory of data cubes.

    align_style : str, optional
        The method to use in aligning the images. Currently 'theoretical',
        'empirical1', and 'empirical2' (default).

    remove_offsets : bool, optional
        Whether to remove offsets in data cube images caused by pointing error
        and dither shifting. This is necessary if you want your cubes to be
        stackable, so don't skip it unless you're just trying to view the
        images and aren't concerned with PSF subtraction. (default: True)
    '''

    def __init__(self, dir_name, align_style='empirical2',
                 remove_offsets=True, verbose=True):
        # inherit from parent classes (https://stackoverflow.com/a/50465583)
        super().__init__(dir_name)
        super(ReadKlipDirectoryBase, self).__init__()
        super(AlignImages, self).__init__()
        super(SubtractImages, self).__init__()

        if verbose:
            print(f"in {dir_name}...", end='')

        # remove dither offsets from data cube images
        # (also removes pointing error if align_style is 'theoretical')
        self.align_style = align_style
        padded_cubes = self._remove_offsets(self.align_style, verbose=verbose)

        # align all images in cubes, then make KLIP projections of the targets
        if self.align_style == 'empirical2':
            self.stackable_cubes = self._finalize_emp2_cubes(padded_cubes,
                                                             verbose=verbose)
            self.klip_proj = self._generate_klip_proj(self.stackable_cubes,
                                                      verbose=verbose)
        elif self.align_style == 'theoretical':
            (self.stackable_cubes,
             fine_ref_cubes) = self._finalize_theo_cubes(padded_cubes,
                                                         verbose=verbose)
            self.klip_proj = self._generate_theo_klip_proj(self.stackable_cubes,
                                                           fine_ref_cubes,
                                                           verbose=verbose)

        # calc. HDULists of assorted contrast/separation data at all wavelengths
        (self.pre_prof_hdu, self.post_prof_hdu, self.photon_prof_hdu,
         self.pre_avg_hdu) = self._generate_contrasts(self.stackable_cubes,
                                                      verbose=verbose)

        # if cubes weren't pre-injected, inject a companion in the target images
        if not hasattr(self, '_derivative') and align_style != 'theoretical':
            self.injected_cubes = self.inject_companion(self.stackable_cubes,
                                                        comp_scale=5,
                                                        verbose=verbose)
        else:
            inj_cbs = self._pklcopy(self.stackable_cubes[len(self.positions):])
            self.injected_cubes = inj_cbs

        # create attribute with ref. cubes and injected then subtracted targets
        ref_list = self._pklcopy(self.stackable_cubes[:len(self.positions)])
        tgt_list = self._pklcopy(self.injected_cubes)

        for i, cube in enumerate(tgt_list):
            cube.data -= self.klip_proj[i].data

        self.subtracted_cubes = fits.HDUList(ref_list + tgt_list)

        # modify docstring for self.inject_companion() based on inheriting class
        # (https://stackoverflow.com/a/4835557)
        docstr = self.inject_companion.__func__.__doc__
        self.inject_companion.__func__.__doc__ = KR_INJECT_DOCSTR + docstr


class PreInjectImages(ReadKlipDirectoryBase, InjectCompanion):
    # inheritance order matters
    '''
    Functionality to help inject a companion into a CreateImages()-created
    directory's data cubes.

    This class is distinct from KlipRetrieve because the injection happens
    *before* the images are aligned, KLIP projected, or subtracted. In fact,
    PreInjectImages() can't perform any of those steps.

    Instead, the typical workflow is to inject a companion using
    `self.inject_companion()` and then export the injected data cubes to a new
    directory with `self.export_to_new_dir()`. (See more about those methods in
    their inherited docstrings from InjectCompanion and ReadKlipDirectoryBase,
    respectively.) You would then use KlipRetrieve() on the new directory as
    normal.

    Arguments:
    ----------
    dir_name : str, required
        The path to your directory of data cubes.
    '''
    def __init__(self, dir_name):
        # inherit from parent classes (https://stackoverflow.com/a/50465583)
        super().__init__(dir_name)
        super(ReadKlipDirectoryBase, self).__init__()

        # modify docstring for self.inject_companion() based on inheriting class
        # (https://stackoverflow.com/a/4835557)
        docstr = self.inject_companion.__func__.__doc__
        self.inject_companion.__func__.__doc__ = PII_INJECT_DOCSTR + docstr
