import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings


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
        padded_cubes = self._shift_dither_pos(self.data_cubes)

        # shift all slices of all images so their bright pixels coincide
        if align_style == 'empirical2': # (newest)
            stackable_cubes = self._align_brights_newest(padded_cubes)

        # or remove pointing error, then....
        else:
            multipad_cubes = self._shift_overall_ptg(padded_cubes)
            if align_style == 'theoretical': # (old)
                # ...calculate star's position and align
                stackable_cubes = self._align_brights_old(multipad_cubes)
            elif align_style == 'empirical1': # (newer)
                # ...align based on mean brightest pixel in each set of images
                # (even if particular slices are slightly offset)
                stackable_cubes = self._align_brights_new(multipad_cubes)
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

        # create array with all slices of all images in cube_list
        padded_list = self._pklcopy(cube_list)
        cube_imgs = np.array([cb.data for cb in padded_list])

        # pad images with NaNs by requested amounts on the image axes
        pad_x = int(abs(pad_x)); pad_y = int(abs(pad_y))
        padded_imgs = np.pad(cube_imgs, ((0, 0), (0, 0),
                                         (pad_y, pad_y), (pad_x, pad_x)),
                             mode='constant', constant_values=np.nan)
        # (y axis is dimension 2, x axis is dimension 3. others are left as-is)

        # update the new HDUList with the padded data
        for i, cube in enumerate(padded_list):
            cube.data = padded_imgs[i]

        print(f"{padded_list[0].data.shape} data cube shape at end",
              end='\n\n')
        return padded_list

    def _shift_dither_pos(self, cube_list):
        '''
        On a pixel scale, removes offsets in a data cube's images that occur as
        a result of the dither process used in its "observations."
        '''
        print('commence removal of dither shifts')

        # What are the shifts due to the dither cycle?
        pix_len = .1 # length of IFU pixel in arcseconds
        dith_shifts = np.round(self.positions / pix_len).astype(int)

        # duplicate shifts -- one set for ref images, repeat for sci images
        dith_shifts = np.tile(dith_shifts, (2,1))

        max_dith_x, max_dith_y = np.abs(dith_shifts).max(axis=0)

        # shifts due to dither cycle's pointing uncertainty (stddev .004 arcsec)
        # are too small to shift for with the IFU's .1 arcsecond/pix resolution

        # Add padding before undoing dithers
        padded_cubes = self._pad_cube_list(self._pklcopy(cube_list),
                                           max_dith_x, max_dith_y)

        # Undo dither steps (as much as possible, pixel-wise) by shifting data
        for i, cube in enumerate(padded_cubes):
            cube.data = np.roll(cube.data,
                                (-dith_shifts[i, 1], -dith_shifts[i, 0]),
                                axis=(1,2))
            # negative movement because you're *undoing* the original shift
            # in that direction. note that axis 0 of cube.data is the
            # wavelength dimension; we're focused on x (2) and y (1), the PSF

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

        # Undo pointing error (as much as possible, pixel-wise) by shifting data
        for i, cube in enumerate(multipad_cubes):
            which_ptg = (ptg_shifts_ref if i < len(self.positions)
                         else ptg_shifts_sci)

            cube.data = np.roll(cube.data, (-which_ptg[1], -which_ptg[0]),
                                axis=(1,2))
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
        ref_bright_pixels = [np.unravel_index(np.nanargmax(ref_images[i, 0]),
                                              ref_images[i, 0].shape)
                             for i in range(ref_images.shape[0])]
        tgt_bright_pixels = [np.unravel_index(np.nanargmax(tgt_images[i, 0]),
                                              tgt_images[i, 0].shape)
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
                cube.data = np.roll(cube.data, (pix_offset[1], pix_offset[0]),
                                    axis=(1,2))
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

        pythag_sep = np.hypot(ref_xs - sci_xs, ref_ys - sci_ys)
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
                    shifts[j, ind] -= 1
                else:
                    ref_means[ind] += pix_len

                    shifts[j] = shifts[j - 1]
                    shifts[j, ind] += 1

                dists[j] = np.hypot(ref_means[0]-sci_xs, ref_means[1]-sci_ys)

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
                cube.data = np.roll(cube.data,
                                    (closest_dist[1], closest_dist[0]),
                                    axis=(1,2))
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
        ref_bright_pixels = [np.unravel_index(np.nanargmax(ref_images[i, 0]),
                                              ref_images[i, 0].shape)[::-1]
                             for i in range(ref_images.shape[0])]
        tgt_bright_pixels = [np.unravel_index(np.nanargmax(tgt_images[i, 0]),
                                              tgt_images[i, 0].shape)[::-1]
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
                cube.data = np.roll(cube.data,
                                    (all_offsets[i, 1], all_offsets[i, 0]),
                                    axis=(1,2))
                # note that axis 0 is the wavelength dimension
                # we're focused on x (2) and y (1) in the PSF

        return stackable_cubes

    def plot_original_pointings(self, dir_name='', return_plot=False):
        '''
        View a plot of the original, un-shifted stellar positions of each image
        in the reference and science sets.

        Use the `dir_name` argument to specify a location for the plot if you'd
        like to save it. If `return_plot` is True, this method will return the
        plot as output without saving anything to disk (even if you specified a
        path for `dir_name`).

        The positions in this plot are driven by the dither positions specified
        in KlipCreate (`self.positions`), overall pointing error
        (`self.point_err_ax`), and (minimally) jitter in the dither pointings
        (`self.dith_err_ax`). The method also plots what the pointings would
        look like if they were error-free.
        '''
        pix_len = .1

        fig, ax = plt.subplots(figsize=(8,8))
        plt.gca().set_aspect('equal')

        # get shape of cubes' x/y axes and draw pixels on plot
        half_y_len = self.data_cubes[0].shape[-2] // 2
        half_x_len = self.data_cubes[0].shape[-1] // 2
        # index of data_cubes doesn't matter

        for i in range(-half_y_len, half_y_len + 1):
            ax.axhline(i * pix_len + pix_len/2,
                       c='k', alpha=.5, ls=':', lw=1)

        for i in range(-half_x_len, half_x_len + 1):
            ax.axvline(i * pix_len + pix_len/2,
                       c='k', alpha=.5, ls=':', lw=1)

        #plt.plot(0, 0, '+', c='lightsalmon', mew=5, markersize=5**2,
        #         zorder=-10, label='center')

        # plot randomly drawn pointings
        plt.scatter(self.draws_ref[:,0], self.draws_ref[:,1],
                    c='#ce1141', label='reference set pointings')
        plt.scatter(self.draws_sci[:,0], self.draws_sci[:,1],
                    c='#13274f', label='target set pointings')
        plt.scatter(self.positions[:,0], self.positions[:,1],
                    marker='+', c='k', s=14**2, zorder=-5,
                    label='intended (error-free) pointings')

        ax.set_xlim(-half_x_len * pix_len, half_x_len * pix_len)
        ax.set_ylim(-half_y_len * pix_len, half_y_len * pix_len)

        ax.set_xlabel('arcesconds', fontsize=14)
        ax.set_ylabel('arcesconds', fontsize=14)
        ax.set_title('star locations in original pointings', fontsize=14)
        ax.legend(fontsize=14)

        if return_plot:
            return fig

        if dir_name:
            plt.savefig(dir_name
                        + ('/' if not dir_name.endswith('/') else '')
                        + 'original_pointings.png', dpi=300,
                        bbox_extra_artists=(leg,), bbox_inches='tight')

        plt.show()

    def plot_shifted_pointings(self, dir_name='', return_plot=False):
        '''
        **(Note that this method is only a valid representation of the alignment
        process if you initiated KlipRetrieve() with
        `align_style='theoretical'` -- the empirical alignment strategies work
        differently.)**

        View a suite of plots of the shifted stellar positions of each image in
        the reference and target sets. Initially, both sets are shifted to come
        as close to (0, 0) as possible, then the reference set is shifted once
        more to come as close to the target set as possible.

        Use the `dir_name` argument to specify a location for the plot if you'd
        like to save it. If `return_plot` is True, this method will return the
        plot as output without saving anything to disk (even if you specified a
        path for `dir_name`).

        The positions in this plot are driven by the dither positions specified
        in KlipCreate (`self.positions`), overall pointing error
        (`self.point_err_ax`), and (minimally) jitter in the dither pointings
        (`self.dith_err_ax`). The method also plots what the pointings would
        look like if they were error-free.
        '''
        pix_len = .1

        # shifts that would align everything, if you could make subpixel moves
        # (accounts for initial pointing offset, but not fine pointing jitter)
        ideal_shifts_ref = ex.positions + ex.point_err_ax[0]
        ideal_shifts_sci = ex.positions + ex.point_err_ax[1]

        # how do those translate to more rough, pixel-scale shifts?
        pixel_shifts_ref = np.round(ideal_shifts_ref, 1)
        pixel_shifts_sci = np.round(ideal_shifts_sci, 1)

        # perform the pixel-scale shifts on both sets of images
        rough_aligned_sci = ex.draws_sci - pixel_shifts_sci
        rough_aligned_ref = ex.draws_ref - pixel_shifts_ref

        # make subplots to feature each target image's star along with the
        # adjustments made to bring the references as close as possible to them
        fig, axs = plt.subplots(2,5, figsize=(15, 10))
        plt.gca().set_aspect('equal')

        for i, row in enumerate(axs):
            for j, ax in enumerate(row):
                img = len(row) * i + j
                fine_aligned_ref = self._align_to_sci_img(img)[0]

                # plot this iteration's target star location
                ax.scatter(rough_aligned_sci[:,0][img],
                           rough_aligned_sci[:,1][img],
                           marker='X', s=15**2, color='#ce1141',
                           label='target image star location')

                # plot before and after of reference fine align process
                ax.scatter(rough_aligned_ref[:,0], rough_aligned_ref[:,1],
                           marker='D', edgecolors='#008ca8', s=10**2,
                           lw=3, facecolor='w', alpha=.4,
                           label='reference image star location (rough)')
                ax.scatter((rough_aligned_ref + fine_aligned_ref)[:,0],
                           (rough_aligned_ref + fine_aligned_ref)[:,1],
                           c='#13274f',
                           label='reference image star location (fine)')

                # draw the center pixel in the field of view
                ax.axhline(-pix_len / 2, color='k', alpha=.5)
                ax.axhline(pix_len / 2, color='k', alpha=.5)
                ax.axvline(-pix_len / 2, color='k', alpha=.5)
                ax.axvline(pix_len / 2, color='k', alpha=.5)

                ax.set_title(f"target image {img}", fontsize=14)
                ax.set_aspect('equal')
                ax.plot(0, 0, marker='+', color='k')
                ax.set_xlim(-pix_len * 2.5/2, pix_len * 2.5/2)
                ax.set_ylim(-pix_len * 2.5/2, pix_len * 2.5/2)

                handles, labels = ax.get_legend_handles_labels()

        axs[0, 0].set_xlabel('arcesconds', fontsize=14)
        axs[0, 0].set_ylabel('arcesconds', fontsize=14)

        fig.suptitle('star locations in shifted pointings, theoretical case',
                     y= .88, fontsize=20)
        fig.legend(handles, labels, loc='center', fontsize=14)

        if return_plot:
            return fig

        if dir_name:
            plt.savefig(dir_name
                        + ('/' if not dir_name.endswith('/') else '')
                        + 'shifted_pointings.png', dpi=300,
                        bbox_extra_artists=(leg,), bbox_inches='tight')

        plt.show()
