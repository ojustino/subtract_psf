import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings


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
    '''

    def __init__(self):
        self._shifted = False

    def _remove_offsets(self, align_style):
        '''
        Removes shifts in images due to dither cycle in both the reference and
        science sets of images. Then, makes sure both sets are centered over
        the same pixel.

        Argument `align_style` controls which method is used for alignment --
        'theoretical' (default) removes overall pointing error and aligns
        bright pixels afterward, while 'empirical1' and 'empirical2' skip
        directly to bright pixel alignment.

        'empirical1' aligns based on the mean brightest pixel in each data cube
        -- even if particular slices have a bright pixel that's slightly
        off-line. 'empirical2' chooses a target position the bright pixel and
        shifts all slices of all data cubes to match it.

        I believe that 'empirical2' should lead to better results than
        'empirical1' since there won't be any off-line stragglers. The
        comparison with 'theoretical' will be interesting to see.
        '''
        if self._shifted == True:
            raise ValueError('PSF images have already been shifted')

        # remove dither offsets
        self.padded_cubes = self._shift_dither_pos()

        # shift all slices of all images so their bright pixels coincide
        if align_style == 'empirical2': # (newest)
            stackable_cubes = self._align_brights_newest(self.padded_cubes)

        # or remove pointing error, then....
        else:
            self.multipad_cubes = self._shift_overall_ptg(self.padded_cubes)
            if align_style == 'theoretical': # (old)
                # ...calculate star's position and align
                stackable_cubes = self._align_brights_old(self.multipad_cubes)
            elif align_style == 'empirical1': # (newer)
                # ...align based on mean brightest pixel in each set of images
                # (even if particular slices are slightly offset)
                stackable_cubes = self._align_brights_new(self.multipad_cubes)
            else:
                raise ValueError("align_style must equal 'theoretical', "
                                 "'empirical1', or 'empirical2'.")

        self._shifted = True
        return stackable_cubes

    def _pad_cube_list(self, cube_list, pad_x, pad_y):
        '''
        Pads both axes of images in a list of data cubes so the real contents
        can be shifted without rolling over an edge. The amount of padding to
        add on both axes is user-specified, and padded indices contain NaNs.
        '''
        print(f"{cube_list[0].data.shape} data cube shape at beginning")

        # create NaN array with proper dimensions for necessary padding
        pad_x = int(abs(pad_x)); pad_y = int(abs(pad_y))
        for cube in cube_list:
            padded = np.full((len(self.wvlnths),
                              cube.data.shape[1] + pad_y * 2,
                              cube.data.shape[2] + pad_x * 2), np.nan)
            # (6,34,34) post-padding in _shift_dither_pos() for example_images/

            # place this ImageHDU's data inside each wvlth's padded array
            # (if a pad amount is 0, slicing 0:None includes all indices)
            padded[:,
                   pad_y:-pad_y if pad_y else None,
                   pad_x:-pad_x if pad_x else None] = cube.data

            # replace the original data with this new, padded version
            cube.data = padded

        print(f"{cube_list[0].data.shape} data cube shape at end",
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

        # shifts due to dither cycle's pointing uncertainty (stddev .004 arcsec)
        # are too small to shift for with the IFU's .1 arcsecond/pix resolution

        # Add padding before undoing dithers
        padded_cubes = self._pad_cube_list(self._pklcopy(self.data_cubes),
                                           max_dith_x, max_dith_y)

        # Undo the dither steps (as much as possible, pixel-wise)\
        # by shifting the data with np.roll
        for i, cube in enumerate(padded_cubes):
            cube.data = np.roll(cube.data,
                                -dith_shifts[i if i < len(self.positions)
                                             else i - len(self.positions)][0],
                                axis=2)
            cube.data = np.roll(cube.data,
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
        # (1-sig is .1 arcseconds)
        pix_len = .1 # length of IFU pixel in arcseconds
        ptg_shifts_ref = np.round(self.point_err_ax[0] / pix_len).astype(int)
        ptg_shifts_sci = np.round(self.point_err_ax[1] / pix_len).astype(int)

        max_ptg_x = np.abs([ptg_shifts_ref[0], ptg_shifts_sci[0]]).max()
        max_ptg_y = np.abs([ptg_shifts_ref[1], ptg_shifts_sci[1]]).max()

        # Add padding before undoing pointing error
        multipad_cubes = self._pad_cube_list(self._pklcopy(padded_cubes),
                                             max_ptg_x, max_ptg_y)

        # Undo the pointing error (as much as possible, pixel-wise)
        # by shifting the data with np.roll
        for i, cube in enumerate(multipad_cubes):
            cube.data = np.roll(cube.data,
                                -(ptg_shifts_ref if i < len(self.positions)
                                  else ptg_shifts_sci)[0],
                                axis=2)
            cube.data = np.roll(cube.data,
                                -(ptg_shifts_ref if i < len(self.positions)
                                  else ptg_shifts_sci)[1],
                                axis=1)
            # negative movement because you're *undoing* the original shift
            # in that direction. note that axis 0 is the wavelength dimension
            # we're focused on x (2) and y (1) in the PSF

        return multipad_cubes

    def _align_brights_new(self, multipad_cubes, pixels_only=False):
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
        the central star. Argument `pixels_only` is used to return each image's
        average bright pixel location and is in self.plot_shifts().)
        '''
        # collect reference and target images as separate arrays with all slices
        ref_images = np.array([cube.data for cube
                               in multipad_cubes[:len(self.positions)]])
        tgt_images = np.array([cube.data for cube
                               in multipad_cubes[len(self.positions):]])

        # for each set, find indices of brightest pixel in each cube's 0th slice
        # (all slices of a given image cube should have the same bright spot)
        ref_bright_pixels = [np.unravel_index(np.nanargmax(ref_images[i][0]),
                                              ref_images[i][0].shape)
                             for i in range(ref_images.shape[0])]
        tgt_bright_pixels = [np.unravel_index(np.nanargmax(tgt_images[i][0]),
                                              tgt_images[i][0].shape)
                             for i in range(tgt_images.shape[0])]

        # get mean pixel position of the brightest pixel among cubes in each set
        # (also, remembering that y comes before x, flip their orders)
        ref_mean_bright = np.mean(ref_bright_pixels, axis=0)[::-1]
        tgt_mean_bright = np.mean(tgt_bright_pixels, axis=0)[::-1]
        print(ref_bright_pixels, tgt_bright_pixels, sep='\n')
        print('r', ref_mean_bright, 't', tgt_mean_bright)
        print('r', ref_mean_bright.round(), 't', tgt_mean_bright.round())
        print(np.round(tgt_mean_bright - ref_mean_bright).astype(int))

        # calculate pixel offset between the two sets' mean bright pixels
        pix_offset = np.round(tgt_mean_bright - ref_mean_bright).astype(int)

        if pixels_only:
            #print('r', ref_mean_bright, 't', tgt_mean_bright, 'off', pix_offset)
            return ref_mean_bright, tgt_mean_bright

        print('Pixel offset of mean brightest pixel in sci & ref sets is '
              f"{pix_offset}")

        # if there is a pixel offset, shift the reference images to eliminate it
        if any(pix_offset != 0):
            # pix_offset values should only be 0 or +/-1; warn if not the case
            if all(x not in [-1, 0, 1] for x in pix_offset):
                warnings.warn('Offset greater than 1 pixel found between mean '
                              'bright pixels in reference and science sets. '
                              'Defaulting to _align_brights_old()')
                return self._align_brights_old(multipad_cubes)

            print('commence alignment of ref images with sci images')
            # add padding before shifting (for ALL cubes)
            stackable_cubes = self._pad_cube_list(self._pklcopy(multipad_cubes),
                                                  pix_offset[0], pix_offset[1])

            # make the adjustment and align the ref & sci images as best we can
            for cube in stackable_cubes[:len(self.positions)]:
                cube.data = np.roll(cube.data, pix_offset[0], axis=2)
                cube.data = np.roll(cube.data, pix_offset[1], axis=1)
                # note that axis 0 is the wavelength dimension
                # we're focused on x (2) and y (1) in the PSF

            return stackable_cubes

        # if no pixel offset, return multipad_cubes -- it's already stackable
        else:
            return multipad_cubes

    def _align_brights_old(self, multipad_cubes, offsets_only=False):
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
        # (stddev is .1 arcseconds)
        ptg_shifts_ref = np.round(self.point_err_ax[0] / .1).astype(int)
        ptg_shifts_sci = np.round(self.point_err_ax[1] / .1).astype(int)

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
                    shifts[j][ind] -= 1
                else:
                    ref_means[ind] += pix_len

                    shifts[j] = shifts[j - 1]
                    shifts[j][ind] += 1

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

            if offsets_only:
                return closest_dist

            # add padding before shifting (for ALL cubes)
            stackable_cubes = self._pad_cube_list(self._pklcopy(multipad_cubes),
                                                  1, 1)

            # make the adjustment and align the ref & sci images as best we can
            for cube in stackable_cubes[:len(self.positions)]:
                cube.data = np.roll(cube.data, closest_dist[0], axis=2)
                cube.data = np.roll(cube.data, closest_dist[1], axis=1)
                # note that axis 0 is the wavelength dimension
                # we're focused on x (2) and y (1) in the PSF

        return stackable_cubes

    def _align_brights_newest(self, padded_cubes, offsets_only=False):
        '''
        ALTERNATE METHOD of alignment that tries to shift pixels only based on
        what it sees empirically after self._shift_dither_pos() has removed the
        dither offsets. (We may not know the other errors a priori, so the
        other approach could be idealized to some extent.)

        Argument `offsets_only` is used to return pixel offset for plotting in
        self.plot_shifts().

        The process is similar to those of  _generate_klip_proj() and
        _generate_contrasts() in KlipRetrieve, which calculate the mean
        positions of the brightest pixels from the reference and science images
        and assume that to be the star's location.

        If the two sets ended up centered on different pixels, shift the
        reference images to overlap with the target images so they can be
        stacked for PSF subtraction later.

        NEEDS ITS OWN PLOTTING FUNCTION; STILL THINKING OF BEST WAY TO VISUALIZE. HEAT MAP?
        '''
        # collect reference and target images as separate arrays with all slices
        ref_images = np.array([cube.data for cube
                               in padded_cubes[:len(self.positions)]])
        tgt_images = np.array([cube.data for cube
                               in padded_cubes[len(self.positions):]])

        # for each set, find indices of brightest pixel in each cube's 0th slice
        # (all slices of a given image cube should have the same bright spot)
        ref_bright_pixels = [np.unravel_index(np.nanargmax(ref_images[i][0]),
                                              ref_images[i][0].shape)[::-1]
                             for i in range(ref_images.shape[0])]
        tgt_bright_pixels = [np.unravel_index(np.nanargmax(tgt_images[i][0]),
                                              tgt_images[i][0].shape)[::-1]
                             for i in range(tgt_images.shape[0])]

        print(ref_bright_pixels, '\n', tgt_bright_pixels)

        if offsets_only:
            #print('r', ref_mean_bright, 't', tgt_mean_bright, 'off', pix_offset)
            return ref_bright_pixels, tgt_bright_pixels

        # save 0th tgt image's bright pixel as intended location in other images
        chosen_pixel = tgt_bright_pixels[0]

        # get distance of other images' bright pixels from that location
        # (tgt_offsets[0] will be (0, 0) by construction)
        all_offsets = (chosen_pixel
                       - np.concatenate((ref_bright_pixels, tgt_bright_pixels)))

        # pad by max separation of other images' bright pixels from chosen_pixel
        max_off_y, max_off_x = np.abs(all_offsets).max(axis=0)
        print(max_off_y, max_off_x)
        #max_off_y, max_off_x = np.maximum(ref_offsets, tgt_offsets).max(axis=0)
        stackable_cubes = self._pad_cube_list(self._pklcopy(padded_cubes),
                                              max_off_y, max_off_x)

        # shift other images so their bright pixels align with 0th target img's
        for i, cube in enumerate(stackable_cubes):
            if any(all_offsets[i] != 0): # ...if necessary
                cube.data = np.roll(cube.data, all_offsets[i][0], axis=2)
                cube.data = np.roll(cube.data, all_offsets[i][1], axis=1)
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
                 marker='s', color='#13274f', alpha=.4,
                 linestyle=':', label='science case pointings')

        # What are the shifts due to the dither cycle?
        dith_shifts = np.array(np.round(self.positions / .1), dtype=int)

        # What are the shifts due to uncertainty in overall pointing?
        # (1-sig is .1 arcseconds)
        ptg_shifts_ref = np.array(np.round(self.point_err_ax[0] / .1),
                                 dtype=int)
        ptg_shifts_sci = np.array(np.round(self.point_err_ax[1] / .1),
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
                 marker='^', color='#13274f',
                 linestyle=':', label='sci sans pnt. error & dither offsets')

        if return_plot:
            return ax

        ax.set_xlabel('arcseconds', fontsize=14)
        ax.set_ylabel('arcseconds', fontsize=14)
        leg = ax.legend(bbox_to_anchor=(1.04, 1), fontsize=14)
        plt.gca().set_aspect('equal')

        # draw representations of NIRSpec pixels (.1 x .1 arcsecond)
        #loc = mpl.ticker.MultipleLocator(base=.1)
        #ax.xaxis.set_major_locator(loc)
        #ax.yaxis.set_major_locator(loc)
        #ax.grid(True, which='both', linestyle='--')

        if dir_name:
            plt.savefig(dir_name
                        + ('/' if not dir_name.endswith('/') else '')
                        + 'pointing_shifts.png',
                        dpi=300, bbox_extra_artists=(leg,), bbox_inches='tight')

        plt.show()
