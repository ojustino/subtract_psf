import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import poppy
import time
import warnings

from astropy import units as u
from astropy.constants import codata2014 as const
from astropy.modeling.blackbody import blackbody_lambda
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ImageNormalize, SinhStretch
from functools import reduce
from inject_images import InjectCompanion


class SubtractImages(InjectCompanion):
    '''
    This class is meant to be inherited by a KlipRetrieve() or PreInjectImages()
    class instance, not used individually.

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

    def __init__(self):
        super().__init__()

    def _get_best_pixels(self, data_cubes, show_footprints=True,
                         indices_only=False):
        '''
        Find the indices in all slices of a given set of data cubes that don't
        contain the NaNs used for padding in AlignImages().

        Creates a coverage map with the same (pixel) size as the scenes in
        `data_cubes`, then goes through all slices of all images, counting
        whether each pixel is non-NaN.

        If `indices_only` is False, the data cubes are then sliced to only
        include the pixels are non-NaN in every scene. If True, the method
        returns the actual array of good indices.
        '''
        coverage_map = np.zeros(data_cubes[0].shape[1:])

        for cube in data_cubes:
            coverage_map += np.isfinite(cube.data[-1])
            # wavelength slice (last index above) doesn't matter

        # save the coordinates of these never-padded pixels
        # (note the reversed axes from array (y by x) to plot (x by y))
        maxed_pixels = np.argwhere(coverage_map == coverage_map.max()).T[::-1]
        best_pix = np.stack((maxed_pixels.min(axis=1),
                             maxed_pixels.max(axis=1)), axis=1)

        if indices_only:
            return best_pix

        # plot footprints for each slice to compare of non-nan pixels by eye
        if show_footprints:
            plt.imshow(coverage_map, norm=mpl.colors.LogNorm(),
                       cmap=plt.cm.magma)
            plt.show()

        # slice images down to their non-NaN pixels
        sliced_cubes = self._pklcopy(data_cubes)
        for i, cube in enumerate(sliced_cubes):
            cube.data = cube.data[:, best_pix[1, 0] : best_pix[1, 1] + 1,
                                  best_pix[0, 0] : best_pix[0, 1] + 1]

        print(f"{sliced_cubes[0].data.shape} "
              'data cube shape after removing padding')

        return sliced_cubes

    def _count_photons(self,
                       temp_star=6000*u.K, rad_star=1*u.solRad, dist=1.5*u.pc,
                       wv=4*u.micron, exp_time=2000*u.second, throughput=.3):
        '''
        *** Something in here is incorrect?***

        Returns the number of photons received by a detector based on the
        stellar and instrument parameters specified as arguments in astropy
        units.

        Remember that photon counting has Poisson-type error, so photon noise is
        the square root of this function's result. For a fuller explanation of
        the process, see my notes in `subtract_psfs.ipynb`.
        '''
        # interpet unitless quantities (e.g. source_proj below) in radians
        u.set_enabled_equivalencies(u.dimensionless_angles())

        # calculate stellar attributes
        #lum_star = const.sigma_sb * temp_star * np.pi * (rad_star)**2
        #flux_bol = lum_star / (4 * np.pi * dist**2)
        #source_proj = np.arctan(rad_star / dist)**2 # exact
        source_proj = (rad_star / dist)**2 # approximated

        # define JWST info
        diam_jwst = 6.5 * u.m
        area_scope = np.pi * (diam_jwst / 2)**2
        #wv = np.mean([self.lo_wv, self.hi_wv]) * u.m
        # resolve_pwr = (len(self.wvlnths) * np.mean([self.lo_wv, self.hi_wv])
        #                / (self.hi_wv - self.lo_wv))
        #wv_resolution = wv / resolve_pwr

        #wv = curr_wv # CHANGE TO ME ONCE TESTING IS COMPLETE!
        wv = np.mean([self.lo_wv, self.hi_wv]) * u.m
        wv_resolution = (self.hi_wv - self.lo_wv) * u.m / len(self.wvlnths)
        # approximating that each wavelength slice is the same width

        # calculate blackbody radiation & photon info based on target wavelength
        bb_rad = blackbody_lambda(wv, temp_star)
        photon_nrg = const.h * const.c / wv

        # get number of photons received by detector and resulting noise
        num_photons = (throughput * area_scope * source_proj * wv_resolution
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

        # collect all images in one 4D array.
        # dimensions are: number of ref & tgt data cubes,
        # number of wavelength slices, and the 2D shape of a post-padded image
        all_cubes = np.array([cube.data for cube in self.stackable_cubes])

        # separate the reference and target data cubes
        refs_all = all_cubes[:len(self.positions)]
        tgts_all = all_cubes[len(self.positions):]

        if verbose:
            print_ast(f"non-padded image shape: {refs_all.shape[2:]}")

        # set up hdulist of klip projections for all slices of all target images
        # (otherwise, has the same HDU structure as stackable_cubes)
        klip_proj = self._pklcopy(self.stackable_cubes[len(self.positions):])

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

    def _generate_contrasts(self, cube_list):
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
        # collect all slices of all target images in one array
        cube_list = self._pklcopy(cube_list)[len(self.positions):]
        tgt_images = np.array([cube.data for cube in cube_list])

        # collect all slices of all KLIP projections in one array
        prj_hdu = self.klip_proj
        prj_images = np.array([cube.data for cube in prj_hdu])

        # create contrast/separation HDUlists to be filled
        # (will have same headers as stackable_cubes but data will change)
        pre_prof_hdu = self._pklcopy(cube_list)
        post_prof_hdu = self._pklcopy(cube_list)
        photon_prof_hdu = self._pklcopy(cube_list)
        pre_avg_hdu = self._pklcopy(cube_list)

        # create a dummy hdulist to match poppy's expected format
        temp_hdu = fits.HDUList([self._pklcopy(cube_list[0]) for _ in range(3)])

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
                # remembering that y and x are flipped
                norm_ind = np.unravel_index(tgt_images[im, sl].argmax(),
                                            tgt_images[im, sl].shape)[::-1]

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
                temp_hdu[0].data = tgt_images[im, sl]
                temp_hdu[1].data = tgt_images[im, sl] - prj_images[im, sl]
                temp_hdu[2].data = np.sqrt(tgt_images[im, sl]) * phot_noise_frac

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
            pre_prof_hdu[im].header['PIXSTARY'] = (norm_ind[1],
                                                    'brightest pixel in '
                                                    f"TARGET{im}, Y direction")
            post_prof_hdu[im].header['PIXSTARY'] = (norm_ind[1],
                                                    'brightest pixel in '
                                                    f"TARGET{im}, Y direction")
            photon_prof_hdu[im].header['PIXSTARY'] = (norm_ind[1],
                                                    'brightest pixel in '
                                                    f"TARGET{im}, Y direction")
            pre_avg_hdu[im].header['PIXSTARY'] = (norm_ind[1],
                                                    'brightest pixel in '
                                                    f"TARGET{im}, Y direction")

            pre_prof_hdu[im].header['PIXSTARX'] = (norm_ind[0],
                                                    'brightest pixel in '
                                                    f"TARGET{im}, X direction")
            post_prof_hdu[im].header['PIXSTARX'] = (norm_ind[0],
                                                    'brightest pixel in '
                                                    f"TARGET{im}, X direction")
            photon_prof_hdu[im].header['PIXSTARX'] = (norm_ind[0],
                                                    'brightest pixel in '
                                                    f"TARGET{im}, X direction")
            pre_avg_hdu[im].header['PIXSTARX'] = (norm_ind[0],
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
                # (retry after except clause to ensure the error was remedied)
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

    def plot_subtraction(self, target_image=0, wv_slice=0, companion=False,
                         dir_name='', return_plot=False, sub_vmax=5e-4, no_plot=False):
        '''
        Creates a four panel plot to demonstrate effect of subtraction.
        Also prints (and can return) pre- and post-subtraction intensity
        measured in the scene.

        Requried arguments allow for the user to select which target image
        (from 0 to len(self.positions) - 1) and which wavelength slice (from 0
        to len(self.stackable_cubes[WHICHEVER].data[1]) - 1) to display.

        When companion=True, the plots show the effect of our subtraction on
        companion detection. Can you see it?

        Optional arguments allow users to return the figure
        (`return_plot=True`) or save it to disk (`dir_name=PATH/TO/DIR`), but
        not both. Somewhat counterintuitive, but to just get intensities without
        plotting (to save time), use `no_plot=True`.

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

        # get star's location
        star_pix_y = self.pre_prof_hdu[target_image].header['PIXSTARY']
        star_pix_x = self.pre_prof_hdu[target_image].header['PIXSTARX']

        # retrieve the proper wavelength of the target image with its projection
        if companion:
            img = self.injected_cubes[target_image]
            tgt_image = img.data[wv_slice]

            # get companion's location
            comp_pix_y = star_pix_y + img.header['PIXCOMPY']
            comp_pix_x = star_pix_x + img.header['PIXCOMPX']

            # get flux info about companion
            try:
                comp_scale = self.injected_cubes[target_image].header['XSIGMA']
                title_str = f", with {comp_scale:.0f}$\sigma$ companion"
            except KeyError:
                num_wv = len(self.wvlnths)
                wv_index = wv_slice if wv_slice >= 0 else num_wv + wv_slice
                keyw = 'CONT' + f"{wv_index:04d}"

                comp_scale = self.injected_cubes[target_image].header[keyw]
                # oom = np.floor((np.log10(np.min(comp_scale))))
                # title_str = f", with 10^{oom:.0f} contrast companion"
                title_str = f", with {comp_scale:.0e} contrast companion"
        else:
            tgt_image = self.stackable_cubes[len(self.positions)
                                             + target_image].data[wv_slice]
            title_str = ''

        proj_obj = self.klip_proj[target_image]
        proj = proj_obj.data[wv_slice]

        if no_plot:
            return tgt_image.sum(), np.abs(tgt_image - proj).sum()

        wvln_key = [i for i in proj_obj.header
                    if i.startswith('WVLN') or i.startswith('WAVELN')][wv_slice]
        # sort???
        wvln = proj_obj.header[wvln_key]

        # build the plot
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        loc = mpl.ticker.MultipleLocator(base=5)
        normed = mpl.colors.LogNorm(vmin=1e-5, vmax=1e0)

        # generate title
        fig.suptitle(f"Target image {target_image} "
                     f"(at {wvln * 1e6:.3f} $\mu$m){title_str}",
                     x=.5, y=.93, fontsize=26)

        # panel 1 (target image)
        curr_ax = axs[0, 0]

        curr_ax.plot(star_pix_x, star_pix_y,
                     marker='+', color='#1d1160', markersize=4**2, mew=2)
        panel = curr_ax.imshow(tgt_image, norm=normed, cmap=plt.cm.magma)

        cbar = fig.colorbar(panel, ax=curr_ax)
        cbar.ax.tick_params(labelsize=16)
        curr_ax.tick_params(axis='both', labelsize=15)
        curr_ax.set_xlabel("pixels (.1'' x .1'')", fontsize=16)
        curr_ax.yaxis.set_major_locator(loc)
        curr_ax.set_title('observed target', size=22)

        # panel 2 (klip projection)
        curr_ax = axs[0, 1]

        cmap =  plt.cm.magma
        cmap.set_bad(cmap(0))
        curr_ax.plot(star_pix_x, star_pix_y,
                     marker='+', color='#1d1160', markersize=4**2, mew=2)
        panel = curr_ax.imshow(proj, norm=normed, cmap=cmap)

        cbar = fig.colorbar(panel, ax=curr_ax)
        cbar.ax.tick_params(labelsize=16)
        curr_ax.tick_params(axis='both', labelsize=15)
        curr_ax.set_xlabel("pixels (.1'' x .1'')", fontsize=16)
        curr_ax.yaxis.set_major_locator(loc)
        curr_ax.set_title('Ref. PSF from KLIP', size=22)

        # panel 3 (target image again, different scaling)
        curr_ax = axs[1, 0]

        if companion: # NEED TO FIND BETTER METHOD OF AUTOMATICALLY SCALING BOTTOM PLOTS THAT ADJUSTS BASED ON XSIGMA OR CONTXXXX header flag
            #times_sigma = img.header['XSIGMA']
            normed = ImageNormalize(vmin=-5e-4, vmax=1e-4,#vmax=times_sigma*1e-4,
                                    stretch=SinhStretch())
            curr_ax.plot(comp_pix_x, comp_pix_y,
                         marker='+', color='#008ca8', mew=2)
        else:
            normed = ImageNormalize(vmin=-sub_vmax, vmax=sub_vmax)

        panel = curr_ax.imshow(tgt_image, norm=normed, cmap=plt.cm.RdBu_r)

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

        panel = curr_ax.imshow(tgt_image-proj, norm=normed, cmap=plt.cm.RdBu_r)

        cbar = fig.colorbar(panel, ax=curr_ax, format='%.2e')
        cbar.ax.tick_params(labelsize=16)
        curr_ax.tick_params(axis='both', labelsize=15)
        curr_ax.set_xlabel("pixels (.1'' x .1'')", fontsize=16)
        curr_ax.yaxis.set_major_locator(loc)
        curr_ax.set_title('observed - KLIP ref', size=22)

        print_ast(f"total intensity pre-subtract:  {tgt_image.sum():.4e}\n"
                  f"total intensity post-subtract: {np.abs(tgt_image - proj).sum():.4e}")

        if return_plot:
            return fig

        if dir_name:
            plt.savefig(dir_name
                        + ('/' if not dir_name.endswith('/') else '')
                        + 'image' + str(target_image) + '_slice' + str(wv_slice)
                        + '_subtraction.png', dpi=300)

        plt.show()

        return tgt_image.sum(), np.abs(tgt_image - proj).sum()

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
            # #### REMEMBER TO MENTION WHAT KIND OF STAR WAS USED FOR PHOTON NOISE CALCULATION IN PLOT LEGEND

            print_ast(f"1 arcsecond contrast @{curr_wv:.2f} microns\n"
                      f"pre-sub:  {pre_prof[i, 1, np.argmin(np.abs(pre_prof[i, 0] - 1))]:.4e}"
                      ' | '
                      f"post-sub: {post_prof[i,1, np.argmin(np.abs(post_prof[i, 0] - 1))]:.4e}")

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

        return post_prof
