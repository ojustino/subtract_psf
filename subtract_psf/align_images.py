#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class AlignImages:
    '''
    This class is meant to be inherited by a KlipRetrieve() or PreInjectImages()
    class instance, not used individually.

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
    '''

    def __init__(self):
        self._shifted = False

    def _remove_offsets(self, align_style, verbose=True):
        '''
        Removes shifts in images due to dither cycle in both the reference and
        science sets of images. Then, makes sure both sets are centered over
        the same pixel.

        Argument `align_style` controls which method is used for alignment --
        'theoretical' (default) removes overall pointing error and aligns
        bright pixels afterward, 'empirical2' skips directly to bright pixel
        alignment, choosing a target position for the bright pixel and shifting
        all slices of all data cubes to match it.

        Argument `verbose` is a boolean that, when True, allows the chosen
        alignment method to print progress messages.
        '''
        if self._shifted == True:
            raise ValueError('PSF images have already been shifted')

        if align_style == 'empirical2':
            # shift all slices of all images so their bright pixels coincide
            padded_cubes = self._align_brights_empirical2(self.data_cubes,
                                                          verbose=verbose)
        elif align_style == 'theoretical':
            # calculate stellar position in each cube and align them
            padded_cubes = self._align_brights_theoretical(self.data_cubes,
                                                           verbose=verbose)
        else:
            raise ValueError("align_style must be 'theoretical' or "
                             "'empirical2'.")
        self._shifted = True
        return padded_cubes

    def _pad_and_shift_cubes(self, cube_imgs, shifts, buffer=True):
        '''
        Pads and uses creative slicing to shift images in data cubes by
        prescribed amounts. Padded indices contain NaNs, and the amount of
        padding depends on the maxiumum shift value in the x/y axes. An
        upgraded version of the former `self.pad_cube_list()` method.

        Argument `cube_imgs` should be a 4D array made from an HDUList of data
        cubes. (`self.data_cubes`, `self.stackable_cubes`, etc.).

        Argument `shifts` is a 2D array of the translations that will be made
        to the images in each cube. It should be the same length as `imgs`; all
        slices of a given data cube are shifted by the same amount. The amount
        of padding in the returned depends on the maximum value in both axes of
        `shifts`.

        Argument `buffer` is boolean that controls whether to add an extra row
        and column of padding on both ends of the x/y axes. It is mainly used
        when `align_style` is 'theoretical' and fine shifts with
        `self._fine_shift_theo_cubes()` are needed.
        '''
        cube_imgs = cube_imgs.copy()

        # decide x/y padding by finding maximum shift required in each axis
        # (plus one to account for possible fine adjustments later on)
        pad_x, pad_y = np.abs(shifts).max(axis=0).astype(int) + 1

        # add an extra row/column to the start and end of x/y axes in final
        # array so there's data to shift if a ref fine adjust is needed later
        if buffer == True:
            shifted_imgs = np.zeros(cube_imgs.shape[0:2]
                                    + (cube_imgs.shape[2] + 2,
                                       cube_imgs.shape[3] + 2))
        else:
            shifted_imgs = np.zeros(cube_imgs.shape)

        # pad images in each cube by prescribed amounts, then
        # slice them to shifted_imgs' shape
        for i, cube in enumerate(cube_imgs):
            y_ind0 = pad_y + shifts[i][1]
            y_ind1 = -pad_y + shifts[i][1]

            x_ind0 = pad_x + shifts[i][0]
            x_ind1 = -pad_x + shifts[i][0]

            if buffer == True:
                y_ind0 -= 1; y_ind1 += 1
                x_ind0 -= 1; x_ind1 += 1

            y_ind1 = None if y_ind1 == 0 else y_ind1
            x_ind1 = None if x_ind1 == 0 else x_ind1

            shifted_imgs[i] = np.pad(cube, #cube.data
                                     ((0, 0), (pad_y, pad_y), (pad_x, pad_x)),
                                     mode='constant',
                                     constant_values=np.nan)[:,
                                                             y_ind0 : y_ind1,
                                                             x_ind0 : x_ind1]
            # note flipped axes

        return shifted_imgs

    def _align_brights_empirical2(self, cube_list, verbose=True):
        '''
        A cube alignment strategy that shifts pixels based only on what it sees
        empirically, assuming that the brightest pixel is the one that holds the
        star and ensuring that pixel falls in the same location in all cubes.
        Adapted from former `self._align_brights_newest()` method.

        Argument `cube_list` is the HDUList of data cubes to be aligned;
        `self.data_cubes` is the usual candidate.

        Argument `verbose` is a boolean that, when True, allows the method to
        print progress messages.

        The logic is that although we know our desired dither cycle, we may not
        know our errors a priori, so the other approach may be too idealized to
        trust completely.

        The process is similar to those of `self._generate_klip_proj()` and
        `self._generate_contrasts()` in KlipRetrieve(), which calculate the mean
        positions of the brightest pixels from the reference and science images
        and assume that to be the star's location.
        '''
        print_ast = lambda text: print('\n********',
                                       text,
                                       '********', sep='\n', end='')
        my_pr = lambda txt: print_ast(txt) if verbose else None
        my_pr("aligning all data cubes' images by their bright pixels...")

        padded_list = self._pklcopy(cube_list)
        cube_imgs = np.array([cb.data for cb in padded_list])

        # for each set, find indices of brightest pixel in each cube's 0th slice
        # (most times, all slices of a cube will share the same bright spot)
        bright_pixels = np.array([np.unravel_index(np.argmax(cube_imgs[i][0]),
                                                   cube_imgs[i][0].shape)[::-1]
                                  for i in range(cube_imgs.shape[0])])

        # choose the center pixel (in x/y) as the intended star location
        chosen_pixel = np.array(cube_imgs.shape[2:]) // 2

        # get distances of other images' bright pixels from that location
        tot_shifts = bright_pixels - chosen_pixel

        # pad images and make appropriate shifts
        aligned_cubes = self._pad_and_shift_cubes(cube_imgs, tot_shifts,
                                                  buffer=False)

        # update the new HDUList
        for i, cube in enumerate(padded_list):
            cube.data = aligned_cubes[i]

            # update target star location info to account for padding/shifts
            if i >= len(self.positions):
                cube.header['PIXSTARY'] += -tot_shifts[i][1]
                cube.header['PIXSTARX'] += -tot_shifts[i][0]

        return padded_list

    def _align_brights_theoretical(self, cube_list, verbose=True):
        '''
        An idealized alignment strategy that, at pixel-scale, begins by
        shifting cubes through removing their respective observations' dither
        steps and overall pointing errors. (Takes from former
        `self._shift_overall_ptg()`, `self._shift_dither_pos()`, and
        `self._align_brights_old()` methods.)

        Argument `cube_list` is the HDUList of data cubes to be aligned;
        `self.data_cubes` is the usual candidate.

        Argument `verbose` is a boolean that, when True, allows the method to
        print progress messages.

        (Jitter in the dither cycle is small enough that it's ignored here.)
        '''
        print_ast = lambda text: print('\n********',
                                       text,
                                       '********', sep='\n', end='')
        my_pr = lambda txt: print_ast(txt) if verbose else None
        my_pr('aligning data cube imgs by removing respective '
              'dither offsets/ptg. errors...')

        pix_len = .1
        padded_list = self._pklcopy(cube_list)
        cube_imgs = np.array([cb.data for cb in padded_list])

        # shifts that would align every image if subpixel shifts were possible
        # (removes dithers and overall pointing error, but not dither error)
        ideal_shifts_ref = self.positions + self.point_err_ax[0]
        ideal_shifts_sci = self.positions + self.point_err_ax[1]

        # translate these to rougher, pixel-scale shifts
        pixel_shifts_ref = np.round(ideal_shifts_ref, 1)
        pixel_shifts_sci = np.round(ideal_shifts_sci, 1)

        # get amounts by which to shift the actual image arrays
        integer_shifts_ref = np.round(ideal_shifts_ref / pix_len).astype(int)
        integer_shifts_sci = np.round(ideal_shifts_sci / pix_len).astype(int)
        tot_shifts = np.concatenate((integer_shifts_ref, integer_shifts_sci))

        # pad images, shift so star locations are as close to (0, 0) as possible
        padded_cubes = self._pad_and_shift_cubes(cube_imgs, tot_shifts)

        for i, cube in enumerate(padded_list):
            cube.data = padded_cubes[i]

            # update target star location info to account for padding/shifts
            if i >= len(self.positions):
                cube.header['PIXSTARY'] += 1 - tot_shifts[i][1]
                cube.header['PIXSTARX'] += 1 - tot_shifts[i][0]

        return padded_list

    def _finalize_emp2_cubes(self, cube_list, verbose=True):
        '''
        Slices out any outer rows/columns containing NaNs from an HDUList of
        padded data cubes that went through the 'empirical2' alignment process.
        Adapted from former `self._get_best_pixels()` method.

        Argument `verbose` is a boolean that, when True, allows the method to
        print progress messages.
        '''
        print_ast = lambda text: print('\n********',
                                       text,
                                       '********', sep='\n', end='')
        my_pr = lambda txt: print_ast(txt) if verbose else None
        my_pr('removing padding from alignment process...')

        cube_list = self._pklcopy(cube_list)
        all_imgs = np.array([cb.data for cb in cube_list])

        # slice each image down to pixels that are non-NaN in every image
        is_pix_finite = np.isfinite(all_imgs).sum(axis=(0,1))
        finite_inds = np.argwhere(is_pix_finite == is_pix_finite.max()).T
        good_pix = np.stack((finite_inds.min(axis=1),
                             finite_inds.max(axis=1) + 1), axis=1)

        final_imgs = all_imgs[:, :,
                              good_pix[0, 0] : good_pix[0, 1],
                              good_pix[1, 0] : good_pix[1, 1]]

        # update cube_list with the finalized data cubes
        for i, cube in enumerate(cube_list):
            cube.data = final_imgs[i]

            # update target star location info after adjusting image shape
            if i >= len(self.positions):
                cube.header['PIXSTARY'] += -good_pix[0, 0]
                cube.header['PIXSTARX'] += -good_pix[1, 0]

        return cube_list

    def _finalize_theo_cubes(self, cube_list, verbose=True):
        '''
        Completes the theoretical alignment process on argument `cube_list` (a
        padded HDUList) and slices images to remove NaNs. It returns the edited
        HDUList along with two lists that are necessary due to intricacies of
        the theoretical alignment process.

        (Argument `verbose` is a boolean that, when True, allows the method to
        print progress messages.)

        For each target cube, the references are "fine-adjusted" so their star
        locations are as close to the pertinent target's as possible. (You can
        visualize the process by calling `self.plot_shifted_pointings()`.)

        Although the adjustments on an axis (x/y) are never more than a pixel
        in either direction, this process can lead to as many distinct
        combinations of references as there are target observations. These
        differences also mean that different sets will have different shapes.

        Instead of making an HDUList for each target + reference combination,
        the method saves the shifted references and sliced targets in separate
        lists and returns them as the edited version of the HDUList passed
        in as `cube_list`.
        '''
        print_ast = lambda text: print('\n********',
                                       text,
                                       '********', sep='\n', end='')
        my_pr = lambda txt: print_ast(txt) if verbose else None
        my_pr('fine-adjusting a set of reference cubes to each target cube...')

        cube_list = self._pklcopy(cube_list)
        ruff_ref_imgs = np.array([cb.data for cb
                                 in cube_list[:len(self.positions)]])
        ruff_tgt_imgs = np.array([cb.data for cb
                                 in cube_list[len(self.positions):]])

        # fine-adjust references to each target image
        fine_tgt_cubes = []
        fine_ref_cubes = []
        for im in range(ruff_tgt_imgs.shape[0]): # number of target images
            # merge references and pertinent target into one array
            refs_and_tgt = np.concatenate((ruff_ref_imgs,
                                           ruff_tgt_imgs[im][np.newaxis]))

            # what pixel shifts of ref images bring them closest to target's
            # star location? (should be 1 pixel or less in either direction)
            ref_img_shifts = self._find_fine_shifts(im, arcsec_scale=False)

            # remove padding from all cubes; make shifts in ref cubes
            aligned_cubes = self._fine_shift_theo_cubes(refs_and_tgt,
                                                        ref_img_shifts)
            # returns a 4D array of ref AND sci images

            # slice each image down to pixels that are non-NaN in every image
            is_pix_finite = np.isfinite(aligned_cubes).sum(axis=(0,1))
            finite_inds = np.argwhere(is_pix_finite == is_pix_finite.max()).T
            good_pix = np.stack((finite_inds.min(axis=1),
                                 finite_inds.max(axis=1) + 1), axis=1)

            final_cubes = aligned_cubes[:, :,
                                        good_pix[0, 0] : good_pix[0, 1],
                                        good_pix[1, 0] : good_pix[1, 1]]

            # update target data & header (star loc.) after adjusting img shape
            # (accounts for extra row/col added in _align_brights_theoretical)
            pad_ind = im + len(self.positions)
            cube_list[pad_ind].data = final_cubes[-1]
            cube_list[pad_ind].header['PIXSTARY'] += -1 - good_pix[0, 0]
            cube_list[pad_ind].header['PIXSTARX'] += -1 - good_pix[1, 0]

            # save sliced target and its fine adjusted reference images
            fine_ref_cubes.append(final_cubes[:-1])

        # remove the buffer NaN rows/columns from cube_list's reference cubes
        for cube in cube_list[:len(self.positions)]:
            cube.data = cube.data[:, 1:-1, 1:-1]

        return cube_list, fine_ref_cubes

    def _fine_shift_theo_cubes(self, cube_imgs, ref_shifts):
        '''
        Called from `self._finalize_theo_cubes()`; meant for data cubes that
        are going through the fine alignment process
        (`align_style='theoretical'`) and were first padded in
        `self._pad_and_shift_cubes()` only.

        Removes padding from all cubes and performs the final, "fine" shift on
        reference cubes.

        Argument `cube_imgs` is a 4D array made from an HDUList of all
        reference cubes and one target cube.

        Argument `ref_shifts` is a 2D array containing the fine shifts that
        will be made to images in the reference cubes. (The target will not be
        shifted.)
        '''
        # compare padded/unpadded images to find how much padding was added
        sample_pad_img = cube_imgs[0][0]
        sample_orig_img = self.data_cubes[0].data[0] # an original input image

        # new images will go back to the original, unpadded shape
        shift_imgs = np.zeros((cube_imgs.shape[0:2] + sample_orig_img.shape))

        # (pad amounts are the same in all cubes after _pad_and_shift_cubes(),
        #  and both ends of y axis are padded by the same amount; same for x)
        pad_y = (sample_pad_img.shape[0] - sample_orig_img.shape[0]) // 2
        pad_x = (sample_pad_img.shape[1] - sample_orig_img.shape[1]) // 2

        # slice all cubes to orig. shape & shift ref cubes by prescribed amounts
        # (resulting cubes may still include NaNs; they'll be removed later)
        for i, cube in enumerate(cube_imgs):
            if i < len(self.positions):
                y_ind0 = pad_y + ref_shifts[i][1]
                y_ind1 = -pad_y + ref_shifts[i][1]
                y_ind1 = None if y_ind1 == 0 else y_ind1

                x_ind0 = pad_x + ref_shifts[i][0]
                x_ind1 = -pad_x + ref_shifts[i][0]
                x_ind1 = None if x_ind1 == 0 else x_ind1
            else: # no shifting necessary for target; only remove padding
                y_ind0 = pad_y; y_ind1 = -pad_y
                x_ind0 = pad_x; x_ind1 = -pad_x

            shift_imgs[i] = cube[:, y_ind0 : y_ind1, x_ind0 : x_ind1]
            # note flipped axes

        return shift_imgs

    def _find_fine_shifts(self, target_image, arcsec_scale=True):
        '''
        (Should only be used when align_style='theoretical'.)

        After bringing the pointings' star locations as close to (0, 0) for all
        images and a specific target image (specified by an integer in argument
        `target_image`), find out whether there are pixel-scale shifts that
        could bring the former closer to the latter.

        Argument `arcsec_scale` is a boolean that controls the scale of the
        returned array of x/y shift information to be applied to the reference
        cubes. If True, the shifts are given in arcseconds (as in
        `self.plot_shifted_pointings()`). If False, the shifts are given in
        pixels (as in `self._finalize_theo_cubes()`; the conversion uses the
        NIRSpec IFU's pixel size of .1 x .1 arcseconds).
        '''
        pix_len = .1

        # shifts that would align every image if subpixel shifts were possible
        # (removes dithers and overall pointing error, but not dither error)
        ideal_shifts_ref = self.positions + self.point_err_ax[0]
        ideal_shifts_sci = self.positions + self.point_err_ax[1]

        # translate these to rougher, pixel-scale shifts
        pixel_shifts_ref = np.round(ideal_shifts_ref, 1)
        pixel_shifts_sci = np.round(ideal_shifts_sci, 1)

        # perform the pixel-scale shifts on ref images and chosen sci image
        rough_aligned_refs = self.draws_ref - pixel_shifts_ref
        rough_aligned_sci_img = (self.draws_sci
                                 - pixel_shifts_sci)[target_image]

        # measure how far reference star locations are from target's star
        residual_ref_offsets = rough_aligned_refs - rough_aligned_sci_img

        # in either axis, if a reference star is more than half a pixel away,
        # move it by a pixel in the direction that will get it closer
        fine_aligned_refs = np.zeros_like(residual_ref_offsets)
        fine_aligned_refs[residual_ref_offsets > pix_len / 2] -= pix_len
        fine_aligned_refs[residual_ref_offsets < -pix_len / 2] += pix_len

        # get an array of the resulting shifts as integers (pixel scale)
        int_aligned_refs = np.round(fine_aligned_refs / pix_len).astype(int)

        if arcsec_scale:
            return fine_aligned_refs
        else:
            return int_aligned_refs
