import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.visualization import ImageNormalize, SinhStretch
from .my_warnings import warnings


class VisualizeImages:
    '''
    This class is meant to be inherited by a KlipRetrieve() class instance, not
    used individually.

    All of these methods visualize some element of the imported data cubes,
    including their pre- and post-alignment star locations
    [self.plot_original_pointings() and self.plot_shifted_pointings(),
    respectively], their pre- and post-subtraction contrast curves
    [self.plot_contrasts() and self.plot_contrasts_avg()], and the results
    of PSF subtraction [self.plot_subtraction()].
    '''
    def __init__(self):
        super().__init__()

    def plot_original_pointings(self, dir_name='', return_plot=False):
        '''
        View a plot of the original, un-shifted stellar positions of each image
        in the reference and science sets.

        The positions in this plot are driven by the dither positions specified
        in KlipCreate (`self.positions`), overall pointing error
        (`self.point_err_ax`), and (minimally) jitter in the dither pointings
        (`self.dith_err_ax`). The method also plots what the pointings would
        look like if they were error-free.

        Argument `dir_name` is a string file path to the location in which to
        save the plot. The filename is chosen automatically.

        Argument `return_plot` is a boolean that allows this method to return
        the matplotlib axes object upon which the plots are drawn if True. This
        option overrides `dir_name`, so no file will be saved if `return_plot`
        is True, even if you specified a path.
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
            return ax

        if dir_name:
            plt.savefig(dir_name
                        + ('/' if not dir_name.endswith('/') else '')
                        + 'original_pointings.png', dpi=300,
                        bbox_extra_artists=(leg,), bbox_inches='tight')

        plt.show()

    def plot_shifted_pointings(self, dir_name='', return_plot=False):
        '''
        Note that this method is only a valid representation of the alignment
        process if you initiated KlipRetrieve() with
        `align_style='theoretical'` -- the empirical strategy works differently.

        View a suite of plots of the shifted stellar positions of each image in
        the reference and target sets. Initially, both sets are shifted to come
        as close to (0, 0) as possible, then the reference sets are shifted once
        more to come as close to each target as possible.

        Argument `dir_name` is a string file path to the location in which to
        save the plot. The filename is chosen automatically.

        Argument `return_plot` is a boolean that allows this method to return
        the matplotlib Figure object upon which the subplots are drawn if True.
        This option overrides `dir_name`, so no file will be saved if
        `return_plot` is True, even if you specified a path.
        '''
        if self.align_style != 'theoretical':
            warnings.warn('This plot does not represent how the alignment '
                          'process worked in this KlipRetrieve() instance '
                          'since the `align_style` you chose was *not* '
                          "'theoretical'.")

        pix_len = .1

        # shifts that would align everything, if you could make subpixel moves
        # (accounts for initial pointing offset, but not fine pointing jitter)
        ideal_shifts_ref = self.positions + self.point_err_ax[0]
        ideal_shifts_sci = self.positions + self.point_err_ax[1]

        # how do those translate to more rough, pixel-scale shifts?
        pixel_shifts_ref = np.round(ideal_shifts_ref, 1)
        pixel_shifts_sci = np.round(ideal_shifts_sci, 1)

        # perform the pixel-scale shifts on both sets of images
        rough_aligned_sci = self.draws_sci - pixel_shifts_sci
        rough_aligned_ref = self.draws_ref - pixel_shifts_ref

        # make subplots to feature each target image's star along with the
        # adjustments made to bring the references as close as possible to them
        fig, axs = plt.subplots(2,5, figsize=(15, 10))
        plt.gca().set_aspect('equal')

        for i, row in enumerate(axs):
            for j, ax in enumerate(row):
                img = len(row) * i + j
                fine_aligned_ref = self._find_fine_shifts(img,arcsec_scale=True)

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

    def plot_subtraction(self, target_image=0, wv_slice=0, companion=True,
                         return_plot=False, dir_name='', no_plot=False,
                         cmap=plt.cm.magma, norm=None):
        '''
        Creates a four panel plot to demonstrate effect of subtraction on a
        single slice of a single target pointing. Also prints (and can return)
        the pre- and post-subtraction intensity measured in the scene.

        The plot is as follows:
        1) target image; 2) KLIP-projected ref. image (same scale for 1 & 2);
        3) target image; 4) target minus ref. image (same scale for 3 & 4).

        Argument `target_image` is an integer representing the pointing from
        which to get an image. Valid values are 0 to len(self.positions) - 1.

        Argument `wv_slice` is an integer representing the wavelength slice
        whose image will be shown. Valid values are 0 to len(self.wvlnths) - 1.

        Argument `companion` is a boolean. When True (default), plots 3 and 4
        show the effect of the subtraction on companion detection. When False,
        plots 3 and 4 show the extent of background star subtraction and are
        hopefully as close to blank as possible. If the data cubes you provided
        when initializing KlipRetrieve already had a pre-injected companion, it
        will appear even if `companion` is False.

        Argument `return_plot` is a boolean that allows this method to return
        the matplotlib Figure object upon which the subplots are drawn if True.

        Argument `dir_name` is a string that, if you'd like to save the figure
        to disk, represents the path to your desired save *directory*. The
        filename is chosen automatically based on `target_image` and `wv_slice`.

        Argument `no_plot` is a boolean that, when True, makes the method skip
        building a figure and just return the pre- and post-subtraction total
        scene intensities. This usage is displayed in `compare_aligns.ipynb`.

        Argument `cmap` is an object that controls the colormap used in all
        plots. Examples can be found in matplotlib.pyplot's cm class.

        Argument `norm` is an object that controls the scaling for plots 3 and
        4. Examples of these objects can be found in astropy's ImageNormalize
        and matplotlib's colors classes.
        '''
        print_ast = lambda text: print('\n********', text, '********', sep='\n')

        # get star's location
        tgt_cubes = self.stackable_cubes[len(self.positions):]
        star_pix_y = tgt_cubes[target_image].header['PIXSTARY']
        star_pix_x = tgt_cubes[target_image].header['PIXSTARX']

        # get the selected target image
        img = (self.injected_cubes[target_image] if companion
               else tgt_cubes[target_image])
        img_data = img.data[wv_slice]

        # if the images have a companion, mark its location
        try:
            # get companion's location (will cause a KeyError if none exists)
            comp_pix_y = star_pix_y + img.header['PIXCOMPY']
            comp_pix_x = star_pix_x + img.header['PIXCOMPX']

            if companion == False:
                warnings.warn('Your original data cubes were pre-injected, so '
                              'proceeding as if `companion`==True.')
                companion = True
        except KeyError:
            title_str = ''
            if companion == True:
                warnings.warn('No companion info was saved for these cubes, so '
                              'proceeding as if `companion`==False. When you '
                              'initialized KlipRetrieve, if you chose '
                              "`align_style`=='theoretical' on a directory of "
                              'data cubes that was *not* pre-injected, this '
                              'happened because the target images take on '
                              'different shapes after the theoretical '
                              'alignment process and thus cannot be put into '
                              'an array for `inject_companion()` to work with.')
                companion = False

        # get the corresponding KLIP projection of the selected target image
        proj_obj = self.klip_proj[target_image]
        proj = proj_obj.data[wv_slice]

        if no_plot:
            return img_data.sum(), np.abs(img_data - proj).sum()

        # search the data cube header for the slice's associated wavelength
        wvln_key = [k for k in img.header
                    if k.startswith('WVLN') or k.startswith('WAVELN')][wv_slice]
        wvln = proj_obj.header[wvln_key]

        # build the plot
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        loc = mpl.ticker.MultipleLocator(base=5)
        norm_top = mpl.colors.LogNorm(vmin=1e-5, vmax=1e0)
        cmap.set_bad(cmap(0))

        # panel 1 (target image)
        curr_ax = axs[0, 0]

        curr_ax.plot(star_pix_x, star_pix_y,
                     marker='+', color='#1d1160', markersize=4**2, mew=2)
        panel = curr_ax.imshow(img_data, norm=norm_top,
                               cmap=cmap, origin='lower')

        cbar = fig.colorbar(panel, ax=curr_ax)
        cbar.ax.tick_params(labelsize=16)
        curr_ax.tick_params(axis='both', labelsize=15)
        curr_ax.set_xlabel("pixels (.1'' x .1'')", fontsize=16)
        curr_ax.yaxis.set_major_locator(loc)
        curr_ax.set_title('observed target', size=22)

        # panel 2 (klip projection)
        curr_ax = axs[0, 1]

        curr_ax.plot(star_pix_x, star_pix_y,
                     marker='+', color='#1d1160', markersize=4**2, mew=2)
        panel = curr_ax.imshow(proj, norm=norm_top, cmap=cmap, origin='lower')

        cbar = fig.colorbar(panel, ax=curr_ax)
        cbar.ax.tick_params(labelsize=16)
        curr_ax.tick_params(axis='both', labelsize=15)
        curr_ax.set_xlabel("pixels (.1'' x .1'')", fontsize=16)
        curr_ax.yaxis.set_major_locator(loc)
        curr_ax.set_title('reference PSF from KLIP', size=22)

        # panel 3 (target image again, different scaling)
        curr_ax = axs[1, 0]

        # if asked, mark companion's location and get its flux info
        if companion:
            curr_ax.plot(comp_pix_x, comp_pix_y,
                         marker='+', color='#008ca8', mew=2)

            try: # random injection case
                # ratio of comp. flux to std dev of scene flux at its radial sep
                times_sigma = img.header['XSIGMA']
                title_str = f", with {times_sigma:.0f}$\sigma$ companion"

                # use a rough version of comp/star contrast to scale the cmap
                cp_flux = img_data[int(comp_pix_y), int(comp_pix_x)]
                st_y = int(star_pix_y); st_x = int(star_pix_x);
                st_flux = tgt_cubes[target_image].data[wv_slice, st_y, st_x]

                contrast = cp_flux / st_flux
                norm_bottom = ImageNormalize(vmin=-contrast/5, vmax=contrast/5)

            except KeyError: # spectrum-based injection case
                # get companion/star contrast ratio from data cube header
                contrast = img.header[[k for k in img.header
                                       if k.startswith('CONT')][wv_slice]]
                title_str = f", with {contrast:.0e} contrast companion"

                # use that contrast to scale the cmap
                norm_bottom = ImageNormalize(vmin=-contrast, vmax=contrast,
                                             stretch=SinhStretch())
        # else, use a similar scaling as above that also accounts for negatives
        else:
            norm_bottom = ImageNormalize(vmin=-1e-4, vmax=5e-4)

        norm_bottom = norm_bottom if norm is None else norm
        panel = curr_ax.imshow(img_data, norm=norm_bottom,
                               cmap=cmap, origin='lower')

        cbar = fig.colorbar(panel, ax=curr_ax, format='%.2e')
        cbar.ax.tick_params(labelsize=16)
        curr_ax.tick_params(axis='both', labelsize=15)
        curr_ax.set_xlabel("pixels (.1'' x .1'')", fontsize=16)
        curr_ax.yaxis.set_major_locator(loc)
        curr_ax.set_title('observed target (again)', size=22)

        # panel 4 (target minus klip projection)
        curr_ax = axs[1, 1]

        if companion:
            curr_ax.plot(comp_pix_x, comp_pix_y,
                         marker='+', color='#008ca8', mew=2)

        panel = curr_ax.imshow(img_data - proj, norm=norm_bottom,
                               cmap=cmap, origin='lower')

        cbar = fig.colorbar(panel, ax=curr_ax, format='%.2e')
        cbar.ax.tick_params(labelsize=16)
        curr_ax.tick_params(axis='both', labelsize=15)
        curr_ax.set_xlabel("pixels (.1'' x .1'')", fontsize=16)
        curr_ax.yaxis.set_major_locator(loc)
        curr_ax.set_title('observed $\minus$ KLIP reference', size=22)

        print_ast(f"total intensity pre-subtract:  {img_data.sum():.4e}\n"
                  f"total intensity post-subtract: {np.abs(img_data - proj).sum():.4e}")

        # generate title
        fig.suptitle(f"Target image {target_image} "
                     f"(at {wvln * 1e6:.3f} $\mu$m){title_str}",
                     x=.5, y=.93, fontsize=26)

        if return_plot:
            return fig

        if dir_name:
            plt.savefig(dir_name
                        + ('/' if not dir_name.endswith('/') else '')
                        + 'image' + str(target_image) + '_slice' + str(wv_slice)
                        + '_subtraction.png', dpi=300)

        plt.show()
        return img_data.sum(), np.abs(img_data - proj).sum()

    def plot_contrasts(self, target_image=0, wv_slices=None, times_sigma=5,
                       companion=False, show_radial=True,
                       return_plot=False, dir_name=''):
        '''
        Plots contrast/separation curves for...

        1. The radial profile(s) of standard deviation for the user-selected
        wavelength(s) of a chosen target image ("pre-subtraction", from
        `self.pre_prof_hdu`)
        2. The radial profile(s) of standard deviation for the user-selected
        wavelength(s) of the chosen target image minus its corresponding KLIP
        projections ("post-subtraction", from `self.post_prof_hdu`)
        3. The photon noise (apparently incorrect, so commented out for now)
        4. The radial profile(s) of average pixel values for the user-selected
        wavelength(s) of the chosen target image. (from `self.pre_avg_hdu`)

        Also prints readout of pre- and post-subtraction contrast at 1 arcsecond
        separation of curves 1 and 2 for quick reference.

        Argument `target_image` is an integer that allows the user to select
        which target image (from 0 to len(self.positions) - 1) to display.

        If argument `wv_slices` is left blank, the lowest, middle, and highest
        wavelength curves in the selected data cube are shown. Otherwise, it
        should be a list or numpy array of integer wavelength indices with
        values between 0 and len(self.wlvnths) - 1).

        Argument `times_sigma` a float representing the amount by which to
        multiply each curve's standard deviation measurment before plotting it.

        Argument `companion` is a boolean that controls whether to plot a point
        representing an injected companion's separation and average contrast
        over the wavelength slices selected. The default value is 0. Will cause
        an error if neither `self.stackable_cubes` nor `self.injected_cubes`
        has a companion.

        Argument `show_radial` is a boolean that controls whether or not to show
        curve #4 (for each requested wavelength slice) on the plot.

        Argument `return_plot` is a boolean that allows this method to return
        the matplotlib axes object upon which the plots are drawn if True.

        Argument `dir_name` is the string file path to the location in which to
        save the plot. The filename is chosen automatically based on
        `target_image`.
        '''
        print_ast = lambda text: print('\n********', text, '********', sep='\n')

        # set up default plotting case (show low, mid, and hi wavelengths)
        num_wv = len(self.wvlnths)
        # (first index of stackable_cubes above shouldn't matter)
        if wv_slices is None:
            wv_slices = np.array([0, num_wv // 2, -1])

        # handle negative wavelength indices
        wv_slices = [num_wv + wv if wv < 0 else wv for wv in wv_slices]

        # get curves corresponding to user-given image/slice combination
        # (separation is index 0, contrast is index 1)
        pre_prof = self.pre_prof_hdu[target_image].data[wv_slices]
        post_prof = self.post_prof_hdu[target_image].data[wv_slices]
        #photon_prof = self.photon_prof_hdu[target_image].data[wv_slices]
        if show_radial:
            pre_avg = self.pre_avg_hdu[target_image].data[wv_slices]

        # create axes and custom colormap
        fig, ax = plt.subplots(figsize=(13,13))
        cmap_from_list = mpl.colors.LinearSegmentedColormap.from_list
        magma_slice = cmap_from_list('', mpl.cm.magma.colors[70:200], 200)

        for i in range(len(wv_slices)):
            # what multiple of pre/post subtraction stddev are we tracking?
            pre_prof[i, 1] *= times_sigma; post_prof[i, 1] *= times_sigma
            #photon_prof[i, 1] *= times_sigma

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
            ax.plot(pre_prof[i, 0], pre_prof[i, 1],
                    label=f"pre-sub STDEV @{curr_wv:.2f} $\mu$m",
                    #alpha=.1,
                    linestyle='-.', c=curr_col)
            ax.plot(post_prof[i, 0], post_prof[i, 1],
                    label=f"post-sub STDEV @{curr_wv:.2f} $\mu$m",
                    #alpha=.1,
                    c=curr_col)

            if show_radial:
                ax.plot(pre_avg[i, 0], pre_avg[i, 1],
                        label=f"pre-sub AVG @{curr_wv:.2f} $\mu$m",
                        #alpha=.1,
                       linestyle=(0, (3, 10)), c=curr_col) # looser dotting

            # ax.plot(photon_prof[i, 0], photon_prof[i, 1],
            #         label=f"pre-sub phot. noise @{curr_wv:.2f} $\mu$m",
            #         linestyle=(0, (2, 2)), c=curr_col)
            # # REMEMBER TO MENTION WHAT KIND OF STAR WAS USED FOR PHOTON NOISE CALCULATION IN PLOT LEGEND

            print_ast(f"1 arcsecond contrast @{curr_wv:.2f} microns\n"
                      f"pre-sub:  {pre_prof[i, 1, np.argmin(np.abs(pre_prof[i, 0] - 1))]:.4e} | "
                      f"post-sub: {post_prof[i,1, np.argmin(np.abs(post_prof[i, 0] - 1))]:.4e}")

        # if asked, plot companion's separation and average contrast info
        if companion:
            pix_len = .1
            try: # if there's a pre-injected companion...
                inj_img = self.stackable_cubes[-1]
                # (index doesn't matter as long as it's in the latter half)
                sep_pix = np.hypot(inj_img.header['PIXCOMPY'],
                                   inj_img.header['PIXCOMPX'])
                sep_arc = sep_pix * pix_len
            except KeyError: # else, use the post-alignment injection
                try:
                    inj_img = self.injected_cubes[0]
                    # (index doesn't matter)
                    sep_pix = np.hypot(inj_img.header['PIXCOMPY'],
                                       inj_img.header['PIXCOMPX'])
                    sep_arc = sep_pix * pix_len
                except KeyError:
                    raise ValueError('No companion found in any of your data '
                                     'cubes. Retry with `companion=False`.')

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try: # if it's a spectrum-based injection...
                    wv_conts = np.array([inj_img.header[h]
                                         for h in inj_img.header
                                         if h.startswith('CONT')])
                    cont = wv_conts[wv_slices].mean()
                except (IndexError, RuntimeWarning):
                    # else, it's a random injection
                    pre_prof = self.pre_prof_hdu[target_image].data[wv_slices]

                    pre_prof_mean = pre_prof[:,1].mean(axis=0)
                    rads = pre_prof[0, 0]
                    # all profiles should have the same separation array;
                    # penultimate index shouldn't matter
                    closest_sep = np.argmin(np.abs(rads - sep_arc))

                    comp_times_sigma = inj_img.header['XSIGMA']
                    cont = (comp_times_sigma
                            * pre_prof_mean[closest_sep] / times_sigma)

            ax.plot(sep_arc, cont, 'P', markersize=20, mew=1.5,
                    markerfacecolor='#d4bd8a', markeredgecolor='k',
                    label='average companion contrast')

        ax.set_xlim(0,)
        ax.set_xlabel('radius (arcseconds)', fontsize=16)
        ax.set_ylabel('contrast', fontsize=16)
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title(f"{times_sigma}$\sigma$ detection level, "
                     f"target image {target_image}",
                     fontsize=16)
        ax.legend(fontsize=14, ncol=2 if show_radial else 1)

        if return_plot:
            return ax

        if dir_name:
            plt.savefig(dir_name
                        + ('/' if not dir_name.endswith('/') else '')
                        + 'image' + str(target_image)
                        + '_contrast_curves.png', dpi=300)

        plt.show()

    def plot_contrasts_avg(self, times_sigma=5, companion=False,
                           return_plot=False, dir_name=''):
        '''
        Creates a plot of the mean radial profile of standard deviation (and
        its 1-sigma envelope) over all images from all pointings, once for pre-
        subtraction images (`self.pre_prof_hdu`) and again for post-subtraction
        images (`self.post_prof_hdu`). Essentially averages every possible
        standard deviation curve in both cases from `self.plot_contrasts()` and
        puts the results into one plot.

        Argument `times_sigma` is a float that represents the amount by which
        to multiply the radial profiles before calculating their statistics and
        plotting them.

        Argument `companion` is a boolean that controls whether to plot the
        companion's separation and contrast data using information from the
        header of either `self.stackable_cubes` (for pre-injected companions) or
        `self.injected_cubes`. If no companion was added, set it to False to
        avoid an error.

        Argument `return_plot` is a boolean that allows this method to return
        the matplotlib Figure object the plots are drawn on if True.

        Argument `dir_name` is a string that, if you'd like to save the figure
        to disk, represents the path to your desired save *directory*. The
        filename is chosen automatically.
        '''
        # all profiles should have the same array of separations; take any one
        rads = self.post_prof_hdu[0].data[0][0] # only the last index matters

        # retrieve saved radial profiles for pre/post-subtraction std dev of
        # scene intensity for all target images in one array each
        pre_prof_arr = np.array([pr.data[:,1] for pr in self.pre_prof_hdu])
        post_prof_arr = np.array([ps.data[:,1] for ps in self.post_prof_hdu])

        # adjust to the desired multiple of standard deviation
        pre_prof_arr *= times_sigma; post_prof_arr *= times_sigma

        # collect each pointing's mean pre/post-sub profile in an array
        pre_prof_per_img = pre_prof_arr.mean(axis=1)
        post_prof_per_img = post_prof_arr.mean(axis=1)

        # find mean radial profiles of pre/post-sub std dev over all pointings
        pre_prof_tot_mean = pre_prof_per_img.mean(axis=0)
        post_prof_tot_mean = post_prof_per_img.mean(axis=0)

        # also save std of each pointing's mean profile
        pre_prof_tot_std = pre_prof_per_img.std(axis=0)
        post_prof_tot_std = post_prof_per_img.std(axis=0)

        # create axes
        fig, ax = plt.subplots(figsize=(13,13))

        # plot mean pre-sub radial profile enveloped by its 1-sigma variation
        ax.plot(rads, pre_prof_tot_mean,
                lw=2, c='#ce1141', label='pre-subtraction radial profile')
        ax.fill_between(rads, np.abs(pre_prof_tot_mean - pre_prof_tot_std),
                        pre_prof_tot_mean + pre_prof_tot_std,
                        color='#e56020', alpha=.5)

        # do the same for the mean post-sub radial profile
        ax.plot(rads, post_prof_tot_mean,
                lw=2, c='#13274f', label='post-subtraction radial profile')
        ax.fill_between(rads, np.abs(post_prof_tot_mean - post_prof_tot_std),
                        post_prof_tot_mean + post_prof_tot_std,
                        color='#b7e4cf', alpha=.5)

        # if asked, plot companion's separation and average contrast info
        if companion:
            pix_len = .1
            try: # if there's a pre-injected companion...
                inj_img = self.stackable_cubes[-1]
                # (index doesn't matter as long as it's in the latter half)
                sep_pix = np.hypot(inj_img.header['PIXCOMPY'],
                                   inj_img.header['PIXCOMPX'])
                sep_arc = sep_pix * pix_len
            except KeyError: # else, use the post-alignment injection
                try:
                    inj_img = self.injected_cubes[0]
                    # (index doesn't matter)
                    sep_pix = np.hypot(inj_img.header['PIXCOMPY'],
                                       inj_img.header['PIXCOMPX'])
                    sep_arc = sep_pix * pix_len
                except KeyError:
                    raise ValueError('No companion found in any of your data '
                                     'cubes. Retry with `companion=False`.')

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try: # if it's a spectrum-based injection...
                    cont = np.mean([inj_img.header[h] for h in inj_img.header
                                    if h.startswith('CONT')])
                except RuntimeWarning: # else, it's a random injection
                    comp_times_sigma = inj_img.header['XSIGMA']
                    closest_sep = np.argmin(np.abs(rads - sep_arc))
                    cont = (comp_times_sigma
                            * pre_prof_tot_mean[closest_sep] / times_sigma)

            ax.plot(sep_arc, cont, 'P', markersize=20, mew=1.5,
                    markerfacecolor='#d4bd8a', markeredgecolor='k',
                    label='average companion contrast')

        ax.set_xlim(0,)
        ax.set_xlabel('radius (arcseconds)', fontsize=16)
        ax.set_ylabel('contrast', fontsize=16)
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title(f"{times_sigma}$\sigma$ detection level", fontsize=16)
        ax.legend(fontsize=14)
        #ax.grid(which='both')

        if return_plot:
            return ax

        if dir_name:
            plt.savefig(dir_name
                        + ('/' if not dir_name.endswith('/') else '')
                        + 'average_radial_proflies.png', dpi=300)

        plt.show()
