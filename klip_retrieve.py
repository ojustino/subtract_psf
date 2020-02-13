import copy
import os
import pickle

from astropy.io import fits
from align_images import AlignImages
from subtract_images import SubtractImages

class KlipRetrieve(AlignImages, SubtractImages):
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
    and separation. See self.plot_subtraction and self.plot_contrasts() for
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

    remove_offsets : bool, optional
        Whether to remove offsets in data cube images caused by pointing error
        and dither shifting. This is necessary if you want your cubes to be
        stackable, so don't skip it unless you're just trying to view the
        images and aren't concerned with PSF subtraction. (default: True)

    align_style : str, optional
        The method to use in aligning the images. Currently 'theoretical'
        (default), 'empirical1', and 'empirical2'.
    '''

    def __init__(self, dir_name, align_style='empirical2',
                 remove_offsets=True):
        super().__init__()

        self.terminal_call = self._get_og_call(dir_name)
        #self.plot_shifts(dir_name)

        # retrieve data cubes, and remove their offsets
        self.data_cubes = self._retrieve_data_cubes(dir_name)
        self.stackable_cubes = self._remove_offsets(align_style)

        # remove self.stackable_cubes' padding so only non-NaN pixels remain...
        self._get_best_pixels(show_footprints=False)

        # then change it from a list of HDULists to one all-encompassing HDUList
        self._flatten_hdulist()

        # make KLIP projections of target images
        self.klip_proj = self._generate_klip_proj()

        # calc. HDULists of assorted contrast/separation data at all wavelengths
        (self.pre_prof_hdu, self.post_prof_hdu,
         self.photon_prof_hdu, self.pre_avg_hdu) = self._generate_contrasts()

        # create a new set of target images with an injected companion
        self.injected_cubes = self.inject_companion()

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

        # import the HDULists of data cubes
        img_files = sorted([f for f in os.listdir(dir_name)
                            if f.endswith('.fits')])

        data_cubes = []
        for f in img_files:
            with fits.open(dir_name + '/' + f) as hdul:
                data_cubes.append(copy.deepcopy(hdul))

        return data_cubes

    def _get_og_call(self, dir_name):
        '''
        Return the original call made in the terminal to generate the data
        cubes in this class instance.
        '''
        with open(dir_name
                  + ('/' if not dir_name.endswith('/') else '')
                  + 'original_call.pkl', 'rb') as file:
            call = pickle.load(file)

        return call
