import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pickle
import poppy
import time
import warnings
import webbpsf

from astropy.constants import codata2014 as const
from astropy.modeling.blackbody import blackbody_lambda
from astropy.io import fits
from astropy import units as u
from functools import reduce


class AlignImages():
    '''
    Helper class for KlipCreate().

    Takes in data cubes from a directory of images and brings the bright points
    of the reference and target images as close to true alignment as possible.
    (It's not an exact removal, instead approximating these shifts at the size
    of a NirSpec IFU pixel -- .1 arcseconds.) The more aligned the two sets of
    images are, the better the results of PSF subtraction will be later on.

    There are three steps -- removing the dither offsets
    (self._shift_dither_pos()), removing offsets due to overall pointing error
    (self._shift_overall_ptg()), and finally, shifting the reference images if
    the previous two steps centered them on a different pixel than the science
    images (self._align_brights()). (Dither pointing error is not considered
    since it's so small compared to the NirSpec IFU's pixel scale.)

    The end product is self.stackable_cubes, which is used by KlipRetrieve()
    for PSF subtraction.

    self.plot_shifts() helps visualize the progression of these alignment
    procedures by plotting the first and last steps.

    MAKE SURE self._align_brights_old() and self._align_brights() ALWAYS PRODUCE IDENTICAL RESULTS
    '''

    def __init__(self):
        self._shifted = False

    def _remove_offsets(self):
        '''
        Removes shifts in images due to dither cycle, overall pointing error in
        both the reference and science sets of images. Then, makes sure both
        sets are centered over the same pixel.
        '''
        if self._shifted == True:
            raise ValueError('PSF images have already been shifted')

        # remove offsets
        self.padded_cubes = self._shift_dither_pos()
        self.multipad_cubes = self._shift_overall_ptg(self.padded_cubes)
        stackable_cubes = self._align_brights(self.multipad_cubes)
        self._shifted = True

        return stackable_cubes

    def _pad_cube_list(self, cube_list, pad_x, pad_y):
        '''
        Pads both axes of images in a list of data cubes so the real contents
        can be shifted without rolling over an edge. The amount of padding to
        add on both axes is user-specified, and padded indices contain NaNs.
        '''
        print(f"{cube_list[0][1].data.shape} data cube shape at beginning")

        # create NaN array with proper dimensions for necessary padding
        pad_x = int(abs(pad_x)); pad_y = int(abs(pad_y))
        for cube in cube_list:
            padded = np.full((len(self.wvlnths),
                              cube[1].data.shape[1] + pad_y * 2,
                              cube[1].data.shape[2] + pad_x * 2), np.nan)
            # (6,34,34) post-padding in _shift_dither_pos() for example_images/

            # place this HDUList's ext 1 table inside each wvlth's padded array
            # (if a pad amount is 0, slicing 0:None includes all indices)
            padded[:,
                   pad_y:-pad_y if pad_y else None,
                   pad_x:-pad_x if pad_x else None] = cube[1].data

            # replace the original ext 1 table with the new one
            cube[1].data = padded

        print(f"{cube_list[0][1].data.shape} data cube shape at end",
              end='\n\n')
        return cube_list

    def _shift_dither_pos(self):
        '''
        On a pixel scale, removes offsets in images that occur as a result of
        the dither process used to generate the current directory's images.
        '''
        print('commence removal of dither shifts')

        # What are the shifts due to the dither cycle?
        pix_len = .1 # length of IFU pixel in arcseconds
        dith_shifts = np.round(self.positions / pix_len).astype(int)

        max_dith_x = dith_shifts[:,0].max()
        max_dith_y = dith_shifts[:,1].max()

        # shifts due to uncertainty in dither cycle pointing (stddev .004 arcsec)
        # are too small to shift for with the IFU's .1 arcsecond/pixel resolution

        # Add padding before undoing dithers
        padded_cubes = self._pad_cube_list(copy.deepcopy(self.data_cubes),
                                           max_dith_x, max_dith_y)

        # Undo the dither steps (as much as possible, pixel-wise)\
        # by shifting the data with np.roll
        for i, cube in enumerate(padded_cubes):
            cube[1].data = np.roll(cube[1].data,
                                   -dith_shifts[i if i < len(self.positions)
                                                else i - len(self.positions)][0],
                                   axis=2)
            cube[1].data = np.roll(cube[1].data,
                                   -dith_shifts[i if i < len(self.positions)
                                                else i - len(self.positions)][1],
                                   axis=1)
            # negative movement because you're *undoing* the original shift
            # in that direction. note that axis 0 is the wavelength dimension
            # we're focused on x (2) and y (1) in the PSF

        return padded_cubes

    def _shift_overall_ptg(self, padded_cubes):
        '''
        On a pixel scale, removes offsets in images that occur as a result of
        JWST's overall pointing error on both axes.
        '''
        print('commence removal of pointing error')

        # What are the shifts due to uncertainty in overall pointing?
        # (1-sig is .25 arcseconds)
        pix_len = .1 # length of IFU pixel in arcseconds
        ptg_shifts_ref = np.round(self.point_err_ax[:2] / pix_len).astype(int)
        ptg_shifts_sci = np.round(self.point_err_ax[2:] / pix_len).astype(int)

        max_ptg_x = np.abs([ptg_shifts_ref[0], ptg_shifts_sci[0]]).max()
        max_ptg_y = np.abs([ptg_shifts_ref[1], ptg_shifts_sci[1]]).max()

        # Add padding before undoing pointing error
        multipad_cubes = self._pad_cube_list(copy.deepcopy(padded_cubes),
                                             max_ptg_x, max_ptg_y)

        # Undo the pointing error (as much as possible, pixel-wise)
        # by shifting the data with np.roll
        for i, cube in enumerate(multipad_cubes):
            cube[1].data = np.roll(cube[1].data,
                                   -(ptg_shifts_ref if i < len(self.positions)
                                     else ptg_shifts_sci)[0],
                                   axis=2)
            cube[1].data = np.roll(cube[1].data,
                                   -(ptg_shifts_ref if i < len(self.positions)
                                     else ptg_shifts_sci)[1],
                                   axis=1)
            # negative movement because you're *undoing* the original shift
            # in that direction. note that axis 0 is the wavelength dimension
            # we're focused on x (2) and y (1) in the PSF

        return multipad_cubes

    def _align_brights(self, multipad_cubes, offsets_only=False):
        '''
        After removing offsets in self._shift_dither_pos() and
        self._shift_overall_ptg(), follow a process similar to
        _generate_klip_proj() and _generate_contrasts() in KlipRetrieve to
        calculate the mean positions of the brightest pixels from the reference
        and science images.

        If the two sets ended up centered on different pixels, shift the
        reference images to overlap with the target images so they can be
        stacked for PSF subtraction later.

        (Assumes that the brightest pixel in an image is the one containing
        the central star. Argument offsets_only is used to return pixel
        offset for plotting in self.plot_shifts().)
        '''
        # collect reference and target images as separate arrays with all slices
        ref_pre_bright = np.array([cube[1].data for cube
                                   in multipad_cubes[:len(self.positions)]])
        tgt_pre_bright = np.array([cube[1].data for cube
                                   in multipad_cubes[len(self.positions):]])

        # for each set, find indices of brightest pixel in each cube's 0th slice
        # (all slices of a given image cube should have the same bright spot)
        ref_bright_pixels = [np.unravel_index(np.nanargmax(ref_pre_bright[i][0]),
                                              ref_pre_bright[i][0].shape)
                             for i in range(ref_pre_bright.shape[0])]
        tgt_bright_pixels = [np.unravel_index(np.nanargmax(tgt_pre_bright[i][0]),
                                              tgt_pre_bright[i][0].shape)
                             for i in range(tgt_pre_bright.shape[0])]

        # get mean pixel position of the brightest pixel among cubes in each set
        # (also, remembering that y comes before x, flip their orders)
        ref_mean_bright = np.mean(ref_bright_pixels, axis=0)[::-1]
        tgt_mean_bright = np.mean(tgt_bright_pixels, axis=0)[::-1]

        # calculate pixel offset between the two sets' mean bright pixels
        pix_offset = np.round(tgt_mean_bright - ref_mean_bright).astype(int)

        if offsets_only:
            #print('r', ref_mean_bright, 't', tgt_mean_bright, 'off', pix_offset)
            return pix_offset

        print('Pixel offset of mean brightest pixel in sci & ref sets is '
              f"{pix_offset}")

        # if there is a pixel offset, shift the reference images to eliminate it
        if any(pix_offset != 0):
            # pix_offset values should only be 0 or +/-1; warn if not the case
            #if any(pix_offset > 1) or any(pix_offset < -1):
            if all(x not in [-1, 0, 1] for x in pix_offset):
                warnings.warn('Offset greater than 1 pixel found between mean '
                              'bright pixels in reference and science sets. '
                              'Defaulting to _align_brights_old()')
                return self._align_brights_old(multipad_cubes)

            print('commence alignment of ref images with sci images')
            # add padding before shifting (for ALL cubes)
            stackable_cubes = self._pad_cube_list(copy.deepcopy(multipad_cubes),
                                                  pix_offset[0], pix_offset[1])

            # make the adjustment and align the ref & sci images as best we can
            for cube in stackable_cubes[:len(self.positions)]:
                cube[1].data = np.roll(cube[1].data, pix_offset[0], axis=2)
                cube[1].data = np.roll(cube[1].data, pix_offset[1], axis=1)
                # note that axis 0 is the wavelength dimension
                # we're focused on x (2) and y (1) in the PSF

            return stackable_cubes

        # if no pixel offset, return multipad_cubes -- it's already stackable
        else:
            return multipad_cubes

    def _align_brights_old(self, multipad_cubes):
        '''
        THIS IS THE OLD METHOD OF ALIGNING THE REFERENCE AND SCIENCE SETS.
        MARKED FOR ARCHIVAL IF THE NEW METHOD CONTINUES TO WORK.

        After removing offsets in self._shift_dither_pos() and
        self._shift_overall_ptg(), calculate the mean positions of the brightest
        pixels from the reference and science images. Use these values to
        determine whether the two sets of images ended up centered on different
        pixels (and print the result).

        If this is so, follow a similar process as before to shift the reference
        images to the pixel upon which the science images are centered.
        '''
        # What are the shifts due to the dither cycle?
        dith_shifts = np.array(np.round(self.positions / .1), dtype=int)

        # What are the shifts due to uncertainty in overall pointing?
        # (stddev is .25 arcseconds)
        ptg_shifts_ref = np.round(self.point_err_ax[:2] / .1).astype(int)
        ptg_shifts_sci = np.round(self.point_err_ax[2:] / .1).astype(int)

        # record mean positions of bright spots in ref and sci images
        pix_len = .1 # length of IFU pixel in arcseconds
        ref_means = (self.draws_ref
                     - (dith_shifts + ptg_shifts_ref) * pix_len).mean(axis=0)
        sci_means = (self.draws_sci
                     - (dith_shifts + ptg_shifts_sci) * pix_len).mean(axis=0)

        ref_xs, ref_ys = ref_means
        sci_xs, sci_ys = sci_means

        print('mean star position by set',
              '---',
              f"ref: ({ref_xs:.4f}, {ref_ys:.4f})",
              f"sci: ({sci_xs:.4f}, {sci_ys:.4f})", sep='\n', end='\n\n')

        pythag_sep = np.sqrt((ref_xs - sci_xs)**2 + (ref_ys - sci_ys)**2)
        ideal_sep = .01 # arcseconds

        reshift = True if pythag_sep > ideal_sep else False
        print('pythagorean separation between sets is '
              f"{'LARGER THAN OPTIMAL at' if reshift else ''} "
              f"{pythag_sep:.4f} arcseconds")

        if reshift:
            # Shift the ref images to each of the four pixels around (0, 0),
            # choosing the one with the smallest dist from the sci images

            # pre-create arrays to store distances for each case and the
            # adjustment made to ref_means used to get there
            dists = np.zeros(4)
            shifts = np.zeros((4,2), dtype=int)

            # simulate shifts by flipping signs of ref images' mean position,
            # e.g. +x,-y; -x,-y; -x,+y; +x,+y
            last_coord = False
            j = 0

            while j <= 3:
                ind = int(last_coord)
                if ref_means[ind] > 0:
                    ref_means[ind] -= pix_len

                    shifts[j] = shifts[j - 1]
                    shifts[j][ind] += 1
                else:
                    ref_means[ind] += pix_len

                    shifts[j] = shifts[j - 1]
                    shifts[j][ind] -= 1

                dists[j] = np.sqrt((ref_means[0] - sci_xs)**2
                                   + (ref_means[1] - sci_ys)**2)

                #print(ref_means) # ensure all values are less than pix_len
                last_coord = not last_coord
                j += 1

            # save the ref_means adjustment that minimizes trans-set separation
            closest_dist = shifts[np.argmin(dists)]
            # (note that dists[-1] is the original ref_xs/ref_ys)

            print('post-alignment separation is '
                  f"{'still notable at' if dists.min() < ideal_sep else ''}"
                  f" {dists.min():.4f} arcseconds")

            # make the adjustment and align the ref & sci images as best we can
            for i, cube in enumerate(stackable_cubes[:len(stackable_cubes)//2]):
                cube[1].data = np.roll(cube[1].data, -closest_dist[0], axis=2)
                cube[1].data = np.roll(cube[1].data, -closest_dist[1], axis=1)
                # note that axis 0 is the wavelength dimension
                # we're focused on x (2) and y (1) in the PSF

        return stackable_cubes

    def plot_shifts(self, dir_name='', return_plot=False):
        '''
        Compare the original mean stellar positions of each image in the
        reference and science sets to where they ended up after this class'
        three-step realignment process.

        Use the `dir_name` argument to specify a location for the plot if you'd
        like to save it. If `return_plot` is True, this method will return the
        plot as output without saving anything to disk (even if you specified a
        path for `dir_name`).
        '''
        fig, ax = plt.subplots(figsize=(10, 10))

        # draw representations of NIRSpec pixels (.1 x .1 arcsecond)
        for i in range(-5, 6):
            ax.axhline(i*0.1, ls=':')
            ax.axvline(i*0.1, ls=':')

        # plot ideal, errorless dither cycle positions
        ax.scatter(self.positions[:,0], self.positions[:,1],
                    marker='+', color='#d4bd8a', lw=10, s=4**2,
                    label='original, errorless dither cycle')

        # plot original center positions for each image
        ax.plot(self.draws_ref[:,0], self.draws_ref[:,1],
                 marker='s', color='#ce1141', alpha=.4,
                 linestyle=':', label='reference case pointings')
        ax.plot(self.draws_sci[:,0], self.draws_sci[:,1],
                 marker='s', color='#1d1160', alpha=.4,
                 linestyle=':', label='science case pointings')

        # What are the shifts due to the dither cycle?
        dith_shifts = np.array(np.round(self.positions / .1), dtype=int)

        # What are the shifts due to uncertainty in overall pointing?
        # (1-sig is .25 arcseconds)
        ptg_shifts_ref = np.array(np.round(self.point_err_ax[:2] / .1),
                                 dtype=int)
        ptg_shifts_sci = np.array(np.round(self.point_err_ax[2:] / .1),
                                 dtype=int)

        # are the reference and science cases centered on the same pixel?
        pix_offsets = self._align_brights(self.multipad_cubes,offsets_only=True)
        pix_len = .1 # length of IFU pixel in arcseconds

        if any(pix_offsets != 0):
            ax.plot((self.draws_ref
                      - (dith_shifts + ptg_shifts_ref[0]) * pix_len)[:,0],
                     (self.draws_ref
                      - (dith_shifts + ptg_shifts_ref[1]) * pix_len)[:,1],
                     marker='^', color='hotpink', linestyle=':',
                     label='unaligned ref sans pnt. error & dither offsets')

        ax.plot((self.draws_ref - (dith_shifts + ptg_shifts_ref[0]
                                    - pix_offsets[0]) * pix_len)[:,0],
                 (self.draws_ref - (dith_shifts + ptg_shifts_ref[1]
                                    - pix_offsets[1]) * pix_len)[:,1],
                 marker='^', color='#ce1141',
                 linestyle=':', label='ref sans pnt. error & dither offsets')

        ax.plot((self.draws_sci
                  - (dith_shifts + ptg_shifts_sci[0]) * pix_len)[:,0],
                 (self.draws_sci
                  - (dith_shifts + ptg_shifts_sci[1]) * pix_len)[:,1],
                 marker='^', color='#1d1160',
                 linestyle=':', label='sci sans pnt. error & dither offsets')

        if return_plot:
            return ax

        ax.set_xlabel('arcseconds', fontsize=14)
        ax.set_ylabel('arcseconds', fontsize=14)
        leg = ax.legend(bbox_to_anchor=(1.04, 1), fontsize=14)
        plt.gca().set_aspect('equal')

        if dir_name:
            plt.savefig(dir_name
                        + ('/' if not dir_name.endswith('/') else '')
                        + 'pointing_shifts.png',
                        dpi=300, bbox_extra_artists=(leg,), bbox_inches='tight')

        plt.show()

class SubtractImages():
    '''
    Helper class for KlipCreate().

    Creates KLIP bases, carries out PSF subtraction, saves the results, and
    also calculates and saves information about contrast and separation in the
    resulting subtracted images. Allows the user to visualize last two features
    on a slice-by-slice basis.

    The key methods that handle calculation and new object creation are...
        - self._generate_klip_proj(): uses the reference images as a library
        from which to make KLIP projections of each slice of each target image.
        Its output is an HDUList of these projections.
        - self._generate_contrasts(): uses the result of the previous method to
        calculate radial profiles for different versions of the target images.
        Its output is an HDUList with contrast versus separation information
        for each version of every slice of every target image.

    The key methods for visualizing results are...
        - self.plot_subtraction(): View a slice of a target image both before
        and after subtraction to judge the subtraction's effectiveness.
        - self.plot_contrasts(): View pre and post-subtraction
        contrast/separation curves for at least one slice of a target image.

    self._get_best_pixels() also returns indices that are non-NaN in each image
    so the user can removes the padded indices from self.stackable_cubes before
    performing further operations on its images.

    self.best_pixels, self.klip_proj, self.*_hdu all come from the methods
    mentioned above.
    '''

    def _get_best_pixels(self, show_footprints=True):
        '''
        Find indices common to all PSFs that don't contain NaNs used for
        padding.

        (Measured per-pixel. The value at each index represents how many images
        in which that pixel was non-NaN. A pixel present in all slices should
        have a value of len(cube[1].data), or the total number of slices in
        this class' instance's cubes.)
        '''
        coverage_map = np.zeros(self.stackable_cubes[0][1].shape[1:])

        for cube in self.stackable_cubes:
            coverage_map += np.isfinite(cube[1].data[-1])
            # wavelength slice (last index above) doesn't matter

        if show_footprints:
            # plot footprints for each slice to compare of non-nan pixels by eye
            plt.imshow(coverage_map, norm=mpl.colors.LogNorm(),
                       cmap=plt.cm.magma)
            plt.show()

        # save the coordinates of these never-padded pixels
        # (note the flipped axes from array (y by x) to plot (x by y))
        present = [arr for arr
                   in np.where(coverage_map == coverage_map.max())][::-1]
        maxed_pixels = np.array(present)
        non_pad_ind = [(i.min(), i.max()) for i in maxed_pixels]

        return non_pad_ind

    def _count_photons(self,
                       temp_star=6000*u.K, rad_star=1*u.solRad, dist=1.5*u.pc,
                       exp_time=2000*u.second, throughput=.3, wv=4*u.micron):
        '''
        Returns the number of photons received by a detector based on the
        stellar and instrument parameters specified as arguments in astropy
        units.

        Remember that photon counting has Poisson-type error, so photon noise is
        the square root of this function's result. For a fuller explanation of
        the process, see my notes in `subtract_psfs.ipynb` (CHECK FOR NAME CHANGE AFTER MAKING REPO)
        '''
        # interpet unitless quantities (e.g. source_proj below) in radians
        u.set_enabled_equivalencies(u.dimensionless_angles())

        # calculate stellar attributes (luminosity, bolometric flux, sky proj.)
        lum_star = const.sigma_sb * temp_star * np.pi * (rad_star)**2
        flux_bol = lum_star / (4 * np.pi * dist**2)
        #source_proj = np.arctan(rad_star / dist)**2 # exact
        source_proj = (rad_star / dist)**2 # approximated

        # define JWST info
        diam_jwst = 6.5 * u.m
        area_scope = np.pi * (diam_jwst / 2)**2
        wv = np.mean([self.lo_wv, self.hi_wv]) * u.m
        slice_width = wv / self.resolve_pwr

        # calculate blackbody radiation & photon info based on target wavelength
        bb_rad = blackbody_lambda(wv, temp_star)
        photon_nrg = const.h * const.c / wv

        # get number of photons received by detector and resulting noise
        num_photons = (throughput * area_scope * source_proj * slice_width
                       *photon_nrg**(-1) * exp_time * bb_rad).decompose().to('')
        #photon_noise = np.sqrt(num_phot)

        return num_photons#, photon_noise

    def _get_klip_basis(self, ref, explain=None, modes=None, verbose=False):
        '''
        Use a a Karhunen-Loève transform to create a set of basis vectors from a
        reference library to be used for KLIP projection later on.

        `ref` is a numpy array (not HDUList) of some number of reference images.

        `explain` is the fraction of variance you want explained by `ref`'s
        eigenvalues, throwing out those aren't needed after the KL transform.

        `modes` is the explicit (maximum) number of eigenvalues to keep.

        (You can use either `explain` or `modes`, but not both.)

        Pass this function's output to self._project_onto_basis() to complete
        the KLIP projection process.
        '''
        if (explain is not None) + (modes is not None) > 1:
            raise ValueError('only one of `explain`/`modes` can have a value')
        elif (explain is not None) + (modes is not None) < 1:
            raise ValueError('either `explain` or `modes` must have a value')

        # flatten psf arrays and find eigenv*s for the result
        ref_flat = ref.reshape(ref.shape[0], -1)
        e_vals, e_vecs = np.linalg.eig(np.dot(ref_flat, ref_flat.T))
        if verbose:
            print('********')
            print(f"eigenvalues are {e_vals}")

        # sort eigenvalues ("singular values") in descending order
        desc = np.argsort(e_vals)[::-1] # sort indices of e_vals in desc. order
        sv = np.sqrt(e_vals[desc]).reshape(-1, 1)

        # do the KL transform
        Z = np.dot(1 / sv * e_vecs[:,desc].T, ref_flat)
        if verbose:
            print(f"Z shape is {Z.shape}")

        if explain:
            test_vars = [np.sum(e_vals[0:i+1]) / np.sum(e_vals) > explain
                         for i, _ in enumerate(e_vals)]
            modes = np.argwhere(np.array(test_vars) == True).flatten()[0] + 1

        # limit Z to a certain number of bases
        Z_trim = Z[:modes,:]
        if verbose:
            print(f"trimmed Z shape is {Z_trim.shape}")

        return Z_trim

    def _project_onto_basis(self, target, Z_trim, verbose=False):
        '''
        Use the output of self._get_klip_basis() for a particular set of
        reference images to project a target image onto a KL basis to estimate
        PSF intensity.

        (Separating the two self.*basis methods helps with the speed of
        self._generate_klip_proj() since target images with the same wavelength
        will also share the same library of reference images.)
        '''
        # flatten target arrays
        targ_flat = target.flatten()
        if verbose:
            print(f"target shape is {targ_flat.shape}")
            print('********')

        # project onto KL basis to estimate PSF intensity
        proj = np.dot(targ_flat, Z_trim.T)
        klipped = np.dot(Z_trim.T, proj).reshape(target.shape)

        return klipped

    def _generate_klip_proj(self, verbose=True):
        '''
        Generate a HDUList of KLIP projections for every slice of each
        post-padded target image data cube. Use the result in
        self.plot_subtraction() and self.plot_contrasts().

        (At the moment, accesses data cube class attributes instead of taking a
        list of reference data cubes as an argument. I could also imagine a
        scenario where both are arguments so the user can provide different
        data cubes from the one in the class, but first things first.)
        '''
        ast = lambda: print('*********')
        print_ast = lambda text: print('\n********',
                                       text,
                                       '********', sep='\n')

        # get indices of never-padded (non-NaN) pixels
        non_pad_ind = self.best_pixels

        # collect all reference and target images in one 4D array.
        # dimensions are... number of (ref or sci) images,
        # number of wavelength slices, and the 2D shape of a post-padded image
        refs_all = np.array([cube[1].data for cube
                             in self.stackable_cubes[:len(self.positions)]])
        tgts_all = np.array([cube[1].data for cube
                             in self.stackable_cubes[len(self.positions):]])

        # slice the final 2D image slices to never-padded pixels only
        refs_all = refs_all[:, :, non_pad_ind[1][0] : non_pad_ind[1][1] + 1,
                            non_pad_ind[0][0] : non_pad_ind[0][1] + 1]
        tgts_all = tgts_all[:, :, non_pad_ind[1][0] : non_pad_ind[1][1] + 1,
                            non_pad_ind[0][0] : non_pad_ind[0][1] + 1]

        if verbose:
            print_ast(f"non-padded image shape: {refs_all.shape[2:]}")

        # set up hdulist of klip projections for all slices of all target images
        # (length is the number of targets, each entry's dimensions should be
        #  number of wavelength slices by the 2D shape of a post-padded image)
        klip_proj = fits.HDUList([fits.PrimaryHDU(np.full(tgts_all.shape[1:],
                                                          np.nan))
                                  for n_img in range(tgts_all.shape[0])])

        # set up some header labels that list wavelength values at each slice
        wvlnth_labels = [key for key
                         in list(self.stackable_cubes[0][1].header.keys())
                         # first index of stackable_cubes above shouldn't matter
                         if key.startswith('WVLN') or key.startswith('WAVELN')]

        for i, img in enumerate(klip_proj):
            img.name = 'TARGET' + str(i)
            for j, key in enumerate(wvlnth_labels):
                img.header[key] = (self.stackable_cubes[0][1].header[key],
                                   f"wavelength of image slice {j:,}")
                # first index of stackable_cubes above shouldn't matter

        # carry out klip projections for all slices of every target image
        # and insert them into the HDUList generated above
        for sl in range(tgts_all.shape[1]): # number of wavelength slices
            refs_sliced = np.array([cube[sl] for cube in refs_all])
            tgts_sliced = np.array([cube[sl] for cube in tgts_all])

            ref_klip_basis = self._get_klip_basis(refs_sliced,
                                                  #explain=.99)
                                                  modes=10)

            for j, tg in enumerate(tgts_sliced):
                ref_klip = self._project_onto_basis(tg, ref_klip_basis)
                klip_proj[j].data[sl] = ref_klip

        return klip_proj

    def _generate_contrasts(self):
        '''
        Generate and return HDULists containing contrast/separation curves for
        every available wavelength over a few variations of the target images:

        1. The radial profile of standard deviation in the original target
        images ("pre-subtraction")
        2. The radial profile of standard deviation in the original target
        images minus their corresponding KLIP projections ("post-subtraction")
        3. The photon noise...
        (STILL WORKING ON THIS. NOT THE RADIAL PROFILE, BUT...)
        4. The radial profile of average pixel value in the original target
        image

        All of these are normalized by the brightest pixel in the original
        target image. Each of these measurements is its own HDUList with length
        equal to the number of target images in the directory. Each entry is a
        stack of 2D separation/contrast arrays (in that order), the number in
        the stack matches the number of wavelength slices available in
        self.stackable_cubes.

        SHOULD USER BE ABLE TO CHOOSE ARGUMENTS FOR _count_photons HERE?
        '''
        # get indices of never-padded (non-NaN) pixels in target images
        non_pad_ind = self.best_pixels

        # collect all target images and projections at all wavelengths
        tgt_images = np.array([cube[1].data for cube
                               in self.stackable_cubes[len(self.positions):]])
        tgt_images = tgt_images[:, :, non_pad_ind[1][0] : non_pad_ind[1][1] + 1,
                                non_pad_ind[0][0] : non_pad_ind[0][1] + 1]

        prj_hdu = self.klip_proj
        prj_images = np.array([cube.data for cube in prj_hdu])

        # create the first contrast/separation hdulist to be filled,
        # also setting up entry titles and header labels for wavelength
        pre_prof_hdu = fits.HDUList([fits.PrimaryHDU() for im
                                     in range(len(self.positions))])
        wvlnth_labels = [key for key
                         in list(self.stackable_cubes[0][1].header.keys())
                         # first index of stackable_cubes shouldn't matter
                         if key.startswith('WVLN') or key.startswith('WAVELN')]

        for num, entry in enumerate(pre_prof_hdu):
            entry.name = 'TARGET' + str(num)
            for j, key in enumerate(wvlnth_labels):
                entry.header[key] = (self.stackable_cubes[0][1].header[key],
                                     f"wavelength of image slice {j:,}")
                # first index of stackable_cubes above shouldn't matter

        # copy the first contrast/separation hdulist to create the rest,
        # which will have the same structure
        post_prof_hdu = copy.deepcopy(pre_prof_hdu)
        photon_prof_hdu = copy.deepcopy(pre_prof_hdu)
        pre_avg_hdu = copy.deepcopy(pre_prof_hdu)

        # create a dummy hdulist to match poppy's expected format
        # (len-3 for num. of unique images at a time for which we take profiles)
        temp_hdu = fits.HDUList([copy.deepcopy(self.stackable_cubes[0][1])
                                 for _ in range(3)])

        # calculate radial profiles at each wavelength, pre & post-subtraction,
        # as well as the radial profile for photon noise and
        # the (radial) average intensity of the original target image
        #start = time.time()
        n_imgs = tgt_images.shape[0]
        n_slcs = tgt_images.shape[1]

        for im in range(n_imgs):
            to_pre_prof_hdu = []
            to_post_prof_hdu = []
            to_photon_prof_hdu = []
            to_pre_avg_hdu = []

            for sl in range(n_slcs):
                # get slice's max pixel value for contrast unit conversion;
                # save indices to help center later radial profile calculations,
                # remembering from non_pad_ind that y and x are flipped
                norm_ind = np.unravel_index(tgt_images[im][sl].argmax(),
                                            tgt_images[im][sl].shape)[::-1]

                # calculate photon noise contribution at current wavelength
                # (it has Poisson-type error, so noise is sqrt(N))
                try: # header name varies with number of slices
                    curr_wv = prj_hdu[0].header[f"WAVELN{sl:02d}"] * u.m
                    # index of prj_hdu shouldn't matter
                except KeyError:
                    curr_wv = prj_hdu[0].header[f"WVLN{sl:04d}"] * u.m

                num_phot = self._count_photons(wv=curr_wv, dist=40*u.pc)
                phot_noise_frac = np.sqrt(num_phot) / num_phot

                # fill temp hdu with current slice's target/proj data
                temp_hdu[0].data = tgt_images[im][sl]
                temp_hdu[1].data = tgt_images[im][sl] - prj_images[im][sl]
                temp_hdu[2].data = np.sqrt(tgt_images[im][sl]) * phot_noise_frac

                # calculate radial profiles for each...
                # (incl. average intensity per radius of pre-subtracted target)
                rad, pre_prof = poppy.radial_profile(temp_hdu, stddev=True,
                                                     ext=0, center=norm_ind)
                _, post_prof = poppy.radial_profile(temp_hdu, stddev=True,
                                                    ext=1, center=norm_ind)
                _, photon_prof = poppy.radial_profile(temp_hdu, stddev=True,
                                                      ext=2, center=norm_ind)
                rad2, pre_avg = poppy.radial_profile(temp_hdu, stddev=False,
                                                     ext=0, center=norm_ind)

                # normalize all profiles, flipping max index back to x by y
                norm = tgt_images[im][sl][norm_ind[::-1]]

                pre_prof /= norm; post_prof /= norm
                photon_prof /= norm; pre_avg /= norm

                # limit stddev measurements to points with non-NaN contrasts
                # (eliminates vertical line that appears in some plots)
                nonz = np.flatnonzero(pre_prof > 0)

                rad = rad[nonz]
                pre_prof = pre_prof[nonz]; post_prof = post_prof[nonz]
                photon_prof = photon_prof[nonz]

                # join separations with contrasts
                to_pre_prof_hdu.append(np.stack((rad, pre_prof)))
                to_post_prof_hdu.append(np.stack((rad, post_prof)))
                to_photon_prof_hdu.append(np.stack((rad, photon_prof)))
                to_pre_avg_hdu.append(np.stack((rad2, pre_avg)))

            # save location of this image's brightest pixel for future use
            # (all slices of the same image should have same bright pixel)
            pre_prof_hdu[im].header[f"PIXSTARY"] = (norm_ind[1],
                                                    'brightest pixel in '
                                                    f"TARGET{im}, Y direction")
            post_prof_hdu[im].header[f"PIXSTARY"] = (norm_ind[1],
                                                    'brightest pixel in '
                                                    f"TARGET{im}, Y direction")
            photon_prof_hdu[im].header[f"PIXSTARY"] = (norm_ind[1],
                                                    'brightest pixel in '
                                                    f"TARGET{im}, Y direction")
            pre_avg_hdu[im].header[f"PIXSTARY"] = (norm_ind[1],
                                                    'brightest pixel in '
                                                    f"TARGET{im}, Y direction")

            pre_prof_hdu[im].header[f"PIXSTARX"] = (norm_ind[0],
                                                    'brightest pixel in '
                                                    f"TARGET{im}, X direction")
            post_prof_hdu[im].header[f"PIXSTARX"] = (norm_ind[0],
                                                    'brightest pixel in '
                                                    f"TARGET{im}, X direction")
            photon_prof_hdu[im].header[f"PIXSTARX"] = (norm_ind[0],
                                                    'brightest pixel in '
                                                    f"TARGET{im}, X direction")
            pre_avg_hdu[im].header[f"PIXSTARX"] = (norm_ind[0],
                                                    'brightest pixel in '
                                                    f"TARGET{im}, X direction")

            # append the current image's data in each hdulist's matching entry
            for att in range(2):
                try:
                    pre_prof_hdu[im].data = np.array(to_pre_prof_hdu)
                    post_prof_hdu[im].data = np.array(to_post_prof_hdu)
                    photon_prof_hdu[im].data = np.array(to_photon_prof_hdu)
                    pre_avg_hdu[im].data = np.array(to_pre_avg_hdu)

                    if att != 0:
                        print('trim successful.')
                    break
                # if array conversion throws an error (typically for different-
                # length entries), only keep separations common to each slice
                # (retry after the except clause to the error was remedied)
                except ValueError as e:
                    if att != 0:
                        raise(e)

                    print('\ndifferent length radial_profile results in '
                          f"image {im}; attempting trim...")

                    # save radii from std and avg calculations separately
                    rad_std = [r[0] for r in to_pre_prof_hdu]
                    rad_avg = [r[0] for r in to_pre_avg_hdu]

                    # find radii that are present in all slices of each array
                    shared_rad_std = reduce(np.intersect1d, rad_std)
                    shared_rad_avg = reduce(np.intersect1d, rad_avg)

                    # mark the corresponding indices in each slice
                    keep_std = [np.in1d(r, shared_rad_std) for r in rad_std]
                    keep_avg = [np.in1d(r, shared_rad_avg) for r in rad_avg]

                    # re-stack the contrast/separation array, now only keeping
                    # the separations marked above and their matching contrasts
                    to_pre_prof_hdu = [np.stack((rad_std[i][k],
                                                 to_pre_prof_hdu[i][1][k]))
                                       for i, k in enumerate(keep_std)]
                    to_post_prof_hdu = [np.stack((rad_std[i][k],
                                                  to_post_prof_hdu[i][1][k]))
                                       for i, k in enumerate(keep_std)]
                    to_photon_prof_hdu= [np.stack((rad_std[i][k],
                                                   to_photon_prof_hdu[i][1][k]))
                                       for i, k in enumerate(keep_std)]
                    to_pre_avg_hdu = [np.stack((rad_avg[i][k],
                                                to_pre_avg_hdu[i][1][k]))
                                       for i, k in enumerate(keep_avg)]

        #print(time.time() - start, 's')
        return pre_prof_hdu, post_prof_hdu, photon_prof_hdu, pre_avg_hdu

    def inject_companion(self, times_sigma=5):
        '''
        Create a new HDUList, `self.injected_cubes`, where every slice of each
        target images from `self.stackable_cubes` has been injected with a
        synthetic, randomly placed companion.

        Argument `times_sigma` controls the brightness of the companion
        relative to the standard deviation of pixel intensity at its radial
        distance from the star. (Radial profile information comes from
        `self.pre_prof_hdu`.)

        Its location changes each time this method is run and is purposely kept
        from being too close to the star or the edge of the frame.

        The user can then subtract corresponding slices of `self.klip_proj`
        from the HDUList produced by this method and determine whether the
        generated companion is recoverable.
        '''
        print_ast = lambda text: print('\n********',
                                           text,
                                           '********', sep='\n')
        print_ast('injecting companion with (location-specific) '
                  f"{times_sigma}-sigma intensity.")

        # retrieve appropriate target image/slice based on arguments
        non_pad_ind = self.best_pixels

        tgt_images = np.array([cube[1].data for cube
                               in self.stackable_cubes[len(self.positions):]])
        tgt_images = tgt_images[:, :, non_pad_ind[1][0] : non_pad_ind[1][1] + 1,
                                non_pad_ind[0][0] : non_pad_ind[0][1] + 1]

        # create HDUList that will the hold new, injected target images
        # (using klip_proj means slices will already have dimensions of non_pad_ind) # REVISIT AFTER CHANGING STACKABLE CUBES
        injected_cubes = copy.deepcopy(self.klip_proj)

        # get location of star in each image
        # (all slices of an image should have the same position)
        star_ys = np.array([i.header['PIXSTARY'] for i in self.pre_prof_hdu])
        star_xs = np.array([i.header['PIXSTARX'] for i in self.pre_prof_hdu])

        # record extreme positions on each axis
        # (we aligned images earlier, so range for each axis should be <= 1)
        star_min_y = star_ys.min(); star_max_y = star_ys.max()
        star_min_x = star_xs.min(); star_max_x = star_xs.max()

        # begin process of randomly selecting companion location
        pixels_y = np.arange(tgt_images.shape[-2])
        pixels_x = np.arange(tgt_images.shape[-1])

        # limit to pixels that aren't right next to the star or at the edge
        # (ALSO assumes the star will never be near the edge -- seems safe?)
        edge_gap = 3; st_gap = 2 #pixels
        poss_y = np.delete(pixels_y,
                           np.s_[star_min_y-st_gap : star_max_y+st_gap])
        poss_x = np.delete(pixels_x,
                           np.s_[star_min_x-st_gap : star_max_x+st_gap])

        poss_y = poss_y[edge_gap:-edge_gap]
        poss_x = poss_x[edge_gap:-edge_gap]

        # calculate mean star pixel position (essentially the mode of each axis)
        star_pix_y = star_ys.mean().round().astype(int)
        star_pix_x = star_xs.mean().round().astype(int)

        # randomly select companion location from remaining pixels
        comp_pix_y = np.random.choice(poss_y)
        comp_pix_x = np.random.choice(poss_x)

        # get contrasts at radial separation between star and companion
        # (due to all earlier realignment, we should be able to inject into
        #  the same pixel of each slice without further adjustment)
        pix_len = .1 #arcseconds
        dist_y = abs(np.round(comp_pix_y - star_pix_y)) * pix_len #arcseconds
        dist_x = abs(np.round(comp_pix_x - star_pix_x)) * pix_len #arcseconds

        for im in range(tgt_images.shape[0]):
            for sl in range(tgt_images.shape[1]):
                # get contrast/separation info (pre-subtraction) for this slice
                separations = self.pre_prof_hdu[im].data[sl][0]
                contrasts = self.pre_prof_hdu[im].data[sl][1]

                # get contrasts at radial separation between star and companion
                rad_dist_ind = np.argmin(abs(separations - max(dist_y, dist_x)))
                comp_contrast = contrasts[rad_dist_ind] * times_sigma

                # simulate the addition of a companion to the frame by copying
                # the image and multiplying it by the companion's contrast
                tgt_slice = tgt_images[im][sl]
                comp_slice = tgt_slice.copy() * comp_contrast

                # roll bright pixel to previously chosen companion location
                comp_slice = np.roll(comp_slice, comp_pix_x-star_pix_x, axis=1)
                comp_slice = np.roll(comp_slice, comp_pix_y-star_pix_y, axis=0)

                # add both images together and complete the simulated injection
                injected_cubes[im].data[sl] = tgt_slice + comp_slice

            # add companions' location info to header
            injected_cubes[im].header['PIXCOMPY'] = comp_pix_y
            injected_cubes[im].header['PIXCOMPX'] = comp_pix_x

        return injected_cubes

    def plot_subtraction(self, target_image=0, wv_slice=0, companion=False,
                         dir_name='', return_plot=False):
        '''
        Creates a four panel plot to demonstrate effect of subtraction.
        Also prints pre- and post-subtraction intensity measured in the scene.

        Requried arguments allow for the user to select which target image
        (from 0 to len(self.positions) - 1) and which wavelength slice (from 0
        to len(self.stackable_cubes[WHICHEVER].data[1]) - 1) to display

        When companion=True, the plots show the effect of our subtraction on
        companion detection. Can you see it? (NEED WAY TO ADJUST SCALING BASED ON times_sigma)

        Optional arguments allow users to return the figure
        (`return_plot=True`) or save it to disk (`dir_name=PATH/TO/DIR`), but
        not both.

        The subtraction is done by using the required arguments to index the
        HDUList from returned by self._generate_klip_proj() to get the
        appropriate KLIP projection of the chosen target.

        The plot is as follows:
        1) target image; 2) KLIP-projected ref. image (same scale for 1 & 2);
        3) target image; 4) target minus ref. image (same scale for 3 & 4).
        '''
        print_ast = lambda text: print('\n********',
                                       text,
                                       '********', sep='\n')

        # retrieve the proper wavelength of the target image with its projection
        if companion:
            img = self.injected_cubes[target_image]
            tgt_image = img.data[wv_slice]

            # get companion's location
            comp_pix_y = img.header['PIXCOMPY']
            comp_pix_x = img.header['PIXCOMPX']
        else:
            non_pad_ind = self.best_pixels
            tgt_image = self.stackable_cubes[len(self.positions)
                                             + target_image][1].data[wv_slice]
            tgt_image = tgt_image[non_pad_ind[1][0] : non_pad_ind[1][1] + 1,
                                  non_pad_ind[0][0] : non_pad_ind[0][1] + 1]

        proj = self.klip_proj[target_image].data[wv_slice]

        # get star's location
        star_pix_y = self.pre_prof_hdu[target_image].header['PIXSTARY']
        star_pix_x = self.pre_prof_hdu[target_image].header['PIXSTARX']

        # build the plot
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        loc = mpl.ticker.MultipleLocator(base=5)

        # panel 1
        curr_ax = axs[0, 0]
        img = curr_ax.imshow(tgt_image,
                             norm=mpl.colors.LogNorm(vmin=1e-5, vmax=1e0),
                             cmap=plt.cm.magma)
        curr_ax.plot(star_pix_x, star_pix_y,
                     marker='+', color='#1d1160', markersize=4**2, mew=2)
        cbar = fig.colorbar(img, ax=curr_ax)
        cbar.ax.tick_params(labelsize=16)
        curr_ax.tick_params(axis='both', labelsize=15)
        curr_ax.set_xlabel("pixels (.1'' x .1'')", fontsize=16)
        curr_ax.yaxis.set_major_locator(loc)
        curr_ax.set_title('observed target', size=22)

        # panel 2
        curr_ax = axs[0, 1]
        img = curr_ax.imshow(proj,
                             norm=mpl.colors.LogNorm(vmin=1e-5, vmax=1e0),
                             cmap=plt.cm.magma)
        curr_ax.plot(star_pix_x, star_pix_y,
                     marker='+', color='#1d1160', markersize=4**2, mew=2)
        cbar = fig.colorbar(img, ax=curr_ax)
        cbar.ax.tick_params(labelsize=16)
        curr_ax.tick_params(axis='both', labelsize=15)
        curr_ax.set_xlabel("pixels (.1'' x .1'')", fontsize=16)
        curr_ax.yaxis.set_major_locator(loc)
        curr_ax.set_title('klipped target', size=22)

        # panel 3
        curr_ax = axs[1, 0]
        img = curr_ax.imshow(tgt_image, vmin=-0.0000, vmax=5e-4,
                             cmap=plt.cm.magma)
        if companion:
            curr_ax.plot(comp_pix_x, comp_pix_y,
                         marker='+', color='#008ca8', mew=2)
        cbar = fig.colorbar(img, ax=curr_ax)
        cbar.ax.tick_params(labelsize=16)
        curr_ax.tick_params(axis='both', labelsize=15)
        curr_ax.set_xlabel("pixels (.1'' x .1'')", fontsize=16)
        curr_ax.yaxis.set_major_locator(loc)
        curr_ax.set_title('observed target (again)', size=22)

        # panel 4
        curr_ax = axs[1, 1]
        img = curr_ax.imshow(tgt_image - proj, vmin=-0.0000, vmax=5e-4,
                             cmap=plt.cm.magma)
        if companion:
            curr_ax.plot(comp_pix_x, comp_pix_y,
                         marker='+', color='#008ca8', mew=2)
        cbar = fig.colorbar(img, ax=curr_ax)
        cbar.ax.tick_params(labelsize=16)
        curr_ax.tick_params(axis='both', labelsize=15)
        curr_ax.set_xlabel("pixels (.1'' x .1'')", fontsize=16)
        curr_ax.yaxis.set_major_locator(loc)
        curr_ax.set_title('observed minus klipped', size=22)

        print_ast(f"total intensity pre-subtract:  {tgt_image.sum():.4e}\n"
                  f"total intensity post-subtract: {np.abs(tgt_image - proj).sum():.4e}")

        if return_plot:
            return axs

        if dir_name:
            plt.savefig(dir_name
                        + ('/' if not dir_name.endswith('/') else '')
                        + 'image' + str(target_image) + '_slice' + str(wv_slice)
                        + '_subtraction.png', dpi=300)

        plt.show()

    def plot_contrasts(self, target_image=0, wv_slices=None, times_sigma=5,
                       show_radial=True, return_plot=False, dir_name=''):
        '''
        Reads from the result of self._generate_contrasts() to create
        contrast/separation plots for...

        1. The radial profile(s) of standard deviation for the user-selected
        wavelength(s) of a chosen target image ("pre-subtraction")
        2. The radial profile(s) of standard deviation for the user-selected
        wavelength(s) of the chosen target image minus its corresponding KLIP
        projections ("post-subtraction")
        3. The photon noise...
        (STILL WORKING ON THIS. NOT THE RADIAL PROFILE, BUT... WHAT?)
        4. The radial profile(s) of average pixel values for the user-selected
        wavelength(s) of the chosen target image. (Display is optional.)

        Also prints readout of pre- and post-subtraction contrast at 1 arcsecond
        separation for quick reference.

        As for arguments, `target_image` allows the user to select which target
        image (from 0 to len(self.positions) - 1) to display.

        If `wv_slices` is left blank, the lowest, middle, and highest wavelength
        curves are shown. Otherwise, it should be a list or numpy array of
        wavelength indices
        (from 0 to len(self.stackable_cubes[WHICHEVER].data[1]) - 1).

        `times_sigma` is the amount by which to multiply each curve's standard
        deviation measurment before plotting it.

        Finally, users can return the plot (`return_plot=True`) or save it to
        disk (`dir_name=PATH/TO/DIR`), but not both.
        '''
        print_ast = lambda text: print('\n********',
                                       text,
                                       '********', sep='\n')

        # set up default plotting case (show low, mid, and hi wavelengths)
        num_wv = self.stackable_cubes[0][1].data.shape[0]
        # (first index of stackable_cubes above shouldn't matter)
        if wv_slices is None:
            wv_slices = np.array([0, num_wv // 2, -1])

        # handle negative wavelength indices
        wv_slices = [num_wv + wv if wv < 0 else wv for wv in wv_slices]

        # get curves corresponding to user-given image/slice combination
        # (separation is index 0, contrast is index 1)
        pre_prof = self.pre_prof_hdu[target_image].data[wv_slices]
        post_prof = self.post_prof_hdu[target_image].data[wv_slices]
        photon_prof = self.photon_prof_hdu[target_image].data[wv_slices]
        if show_radial:
            pre_avg = self.pre_avg_hdu[target_image].data[wv_slices]

        # create axes and custom colormap
        fig, ax = plt.subplots(figsize=(13,13))
        cmap_from_list = mpl.colors.LinearSegmentedColormap.from_list
        magma_slice = cmap_from_list('', mpl.cm.magma.colors[70:200], 200)

        for i in range(len(wv_slices)):
            # what multiple of pre/post subtraction stddev are we tracking?
            # (we're saying for now that 5-sig is an unamibguous detection --
            #  **** CONFIRM THIS ONCE COMPANION INJECTION IS WRITTEN) ****
            pre_prof[i][1] *= times_sigma; post_prof[i][1] *= times_sigma
            photon_prof[i][1] *= times_sigma

            # map colors to curves by wavelength
            curr_col = magma_slice(wv_slices[i] / (num_wv - 1))

            # record the wavelength (in microns) of the current curves
            try: # header name varies with number of slices
                curr_wv = self.pre_prof_hdu[0].header['WAVELN'
                                                      + f"{wv_slices[i]:02d}"]
                # (specific hdu and index used above shouldn't matter)
            except KeyError:
                curr_wv = self.pre_prof_hdu[0].header['WVLN'
                                                      + f"{wv_slices[i]:04d}"]
            curr_wv *= 1e6

            # plot contrast needed for obvious companion detections
            # as a function of radius
            ax.plot(pre_prof[i][0], pre_prof[i][1],
                    label=f"pre-sub STDEV @{curr_wv:.2f} $\mu$m",
                    #alpha=.1,
                    linestyle='-.', c=curr_col)
            ax.plot(post_prof[i][0], post_prof[i][1],
                    label=f"post-sub STDEV @{curr_wv:.2f} $\mu$m",
                    #alpha=.1,
                    c=curr_col)

            if show_radial:
                ax.plot(pre_avg[i][0], pre_avg[i][1],
                        label=f"pre-sub AVG @{curr_wv:.2f} $\mu$m",
                        #alpha=.1,
                       linestyle=(0, (3, 10)), c=curr_col) # looser dotting

            ax.plot(photon_prof[i][0], photon_prof[i][1],
                    label=f"pre-sub phot. noise @{curr_wv:.2f} $\mu$m",
                    linestyle=(0, (2, 2)), c=curr_col)
            #### REMEMBER TO MENTION WHAT KIND OF STAR WAS USED FOR PHOTON NOISE CALCULATION IN PLOT LEGEND

            print_ast(f"1 arcsecond contrast @{curr_wv:.2f} microns\n"
                      f"pre-sub:  {pre_prof[i][1][np.argmin(np.abs(pre_prof[i][0] - 1))]:.4e}"
                      ' | '
                      f"post-sub: {post_prof[i][1][np.argmin(np.abs(post_prof[i][0] - 1))]:.4e}")

        if return_plot:
            return ax

        ax.set_xlim(0,)
        ax.set_xlabel('radius (arcseconds)', fontsize=16)
        ax.set_ylabel('contrast', fontsize=16)
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title(f"{times_sigma}$\sigma$ detection level, "
                     f"target image {target_image}",
                     fontsize=16)
        ax.legend(fontsize=14, ncol=2 if show_radial else 1)

        if dir_name:
            plt.savefig(dir_name
                        + ('/' if not dir_name.endswith('/') else '')
                        + 'image' + str(target_image)
                        + '_contrast_curves.png', dpi=300)

        plt.show()

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
    ADD return_plot OPTION TO plot_shifts() IN AlignImages
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
    '''

    def __init__(self, dir_name, remove_offsets=True):
        super().__init__()

        self.data_cubes = self._retrieve_data_cubes(dir_name)
        self.stackable_cubes = self._remove_offsets()

        self.terminal_call = self._get_og_call(dir_name)
        self.plot_shifts(dir_name)

        self.best_pixels = self._get_best_pixels(show_footprints=False)
        self.klip_proj = self._generate_klip_proj()
        # access already generated hdulists with contrast/separation curves
        # (each contains all available wavelengths)
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
