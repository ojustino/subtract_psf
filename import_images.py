import numpy as np
import os
import pickle
import warnings

from astropy.io import fits
from align_images import AlignImages
from inject_images import InjectCompanion
from subtract_images import SubtractImages


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
        self.data_cubes = self._retrieve_data_cubes(dir_name)

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

    def export_to_new_dir(self, cube_list, new_dir_name, overwrite=True):
        '''
        Create a new directory of data cubes that are still usable by
        KlipRetrieve. `new_dir_name` is the path to your desired directory, and
        `cube_list` is the HDUList that will be saved, extension-by-extension.

        A typical use of this method is to save `self.injected_cubes` after
        injecting a companion in `self.data_cubes` or `self.stackable_cubes`.

        Parameters
        ----------
        overwrite : bool
            invoke special options to output and overwrite prior cubes
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

        # check if all data cubes have the same shape
        if len(np.unique([cube_list[i].shape for i in range(len(cube_list))],
                         axis=0)) != 1:
            warnings.warn('Not all data cubes in this list have the same '
                          'shape. Is that intentional? You will have problems '
                          'using KlipRetrieve with this new directory.')

        # create the new directory
        os.makedirs(new_dir_name, exist_ok=True)

        # save the FITS files
        new_dir_name += ('/' if not new_dir_name.endswith('/') else '')

        for i in range(len(cube_list)):
            fits.writeto(new_dir_name +
                         f"{'ref' if i < len(self.positions) else 'sci'}"
                         f"_image{i % 10}.fits",
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

    def _locate_target_stars(self, cube_list):
        '''
        Label the target star locations in `self.data_cubes` so it can be
        tracked through the alignment process, and not take up space in the
        `self._generate_contrasts()` loop. UNDER EXAMINATION.
        '''
        cube_list = self._pklcopy(cube_list)
        pix_len = .1

        # translate arcsecond coordinates to pixel coordinates based on the size
        # of images in the data cubes' slices
        pix_center = [(size - 1) / 2 for size
                      in self.data_cubes[-1].shape[-2:]] # y, x, while draws_sci is x, y
        #all_draws = np.vstack((self.draws_ref, self.draws_sci)) / pix_len #x, y
        star_locs = pix_center + self.draws_sci[:, ::-1]
        # img_center is y, x and draws_* is x, y; one of them had to be flipped

        # update target data cubes with their corresponding star location info
        for i, cube in enumerate(cube_list[len(self.positions):]):
            # should ref cubes have PIXSTARY?
            cube.header['PIXSTARY'] = (star_locs[i, 0], 'brightest pixel in '
                                       f"TARGET{im}, Y direction")
            cube.header['PIXSTARX'] = (star_locs[i, 1], 'brightest pixel in '
                                       f"TARGET{im}, X direction")

        return cube_list

    def _get_og_call(self, dir_name):
        '''
        Return the original call made in the terminal to generate the data
        cubes in this class instance.
        '''
        with open(dir_name
                  + 'original_call.pkl', 'rb') as file:
            call = pickle.load(file)

        return call


class KlipRetrieve(ReadKlipDirectoryBase, AlignImages, SubtractImages):
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

    ********
    QUESTION: Should there be an argument for KlipCreate that indicates how many modes/percentage of variance to explain through KLIP?
    ********

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

    def __init__(self, dir_name, align_style='empirical2', remove_offsets=True):
        # inherit from parent classes (https://stackoverflow.com/a/50465583)
        super().__init__(dir_name)
        super(ReadKlipDirectoryBase, self).__init__()
        super(AlignImages, self).__init__()

        # remove dither offsets from data cube images
        # (also removes pointing error if align_style is not 'empirical2')
        aligned_cubes = self._remove_offsets(align_style)
        #self.plot_shifts(self.dir_name)

        # remove aligned_cubes' padding so only non-NaN pixels remain...
        self.stackable_cubes = self._get_best_pixels(aligned_cubes, False)

        # make KLIP projections of target images in stackable_cubes
        self.klip_proj = self._generate_klip_proj()

        # calc. HDULists of assorted contrast/separation data at all wavelengths
        (self.pre_prof_hdu, self.post_prof_hdu, self.photon_prof_hdu,
         self.pre_avg_hdu) = self._generate_contrasts(self.stackable_cubes)

        # if cubes weren't pre-injected, inject a companion in the target images
        if not hasattr(self, '_derivative'):
            self.injected_cubes = self.inject_companion(self.stackable_cubes,
                                                        comp_scale=5)

        # modify docstring for inject_companion() based on inheriting class
        # (https://stackoverflow.com/a/4835557)
        docstr = self.inject_companion.__func__.__doc__
        self.inject_companion.__func__.__doc__ = KR_INJECT_DOCSTR + docstr


class PreInjectImages(ReadKlipDirectoryBase, InjectCompanion):
    # inheritance order matters
    '''
    Inject a companion into a directory's worth of data cubes.

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

        # modify docstring for inject_companion() based on inheriting class
        # (https://stackoverflow.com/a/4835557)
        docstr = self.inject_companion.__func__.__doc__
        self.inject_companion.__func__.__doc__ = PII_INJECT_DOCSTR + docstr
