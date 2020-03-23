#!/usr/bin/env python3
import numpy as np
import poppy
import time

from astropy import units as u
from astropy.constants import codata2014 as const
from astropy.modeling.blackbody import blackbody_lambda
from functools import reduce
from inject_images import InjectCompanion
from my_warnings import warnings


class SubtractImages(InjectCompanion):
    '''
    This class is meant to be inherited by a KlipRetrieve() class instance, not
    used individually.

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

    self.best_pixels, self.klip_proj, self.*_hdu all come from the methods
    mentioned above.
    '''

    def __init__(self):
        super().__init__()

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
        # (not all KlipCreate sessions use resolve_pwr,
        #  so need a safer way to calc resolution)

        #wv = np.mean([self.lo_wv, self.hi_wv]) * u.m # not needed
        wv_resolution = (self.hi_wv - self.lo_wv) * u.m / len(self.wvlnths)
        # approximating that each wavelength slice is the same width

        # calculate blackbody radiation & photon info based on target wavelength
        with warnings.catch_warnings(record=True) as w:
            # ignore astropy 4's blackbody-related deprecation warning, for now
            warnings.simplefilter('ignore')
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

        my_pr = lambda *args, **kwargs: (print(*args, **kwargs)
                                         if verbose else None)

        # flatten psf arrays and find eigenv*s for the result
        ref_flat = ref.reshape(ref.shape[0], -1)
        e_vals, e_vecs = np.linalg.eig(np.dot(ref_flat, ref_flat.T))
        my_pr('********', "eigenvalues are {e_vals}", sep='\n')

        # sort eigenvalues ("singular values") in descending order
        desc = np.argsort(e_vals)[::-1] # sort indices of e_vals in desc. order
        sv = np.sqrt(e_vals[desc]).reshape(-1, 1)

        # do the KL transform
        Z = np.dot(1 / sv * e_vecs[:,desc].T, ref_flat)
        my_pr(f"Z shape is {Z.shape}")

        if explain:
            test_vars = [np.sum(e_vals[0:i+1]) / np.sum(e_vals) > explain
                         for i, _ in enumerate(e_vals)]
            modes = np.argwhere(np.array(test_vars) == True).flatten()[0] + 1

        # limit Z to a certain number of bases
        Z_trim = Z[:modes,:]
        my_pr(f"trimmed Z shape is {Z_trim.shape}")

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
        my_pr = lambda *args, **kwargs: (print(*args, **kwargs)
                                         if verbose else None)

        # flatten target arrays
        targ_flat = target.flatten()
        if verbose:
            my_pr(f"target shape is {targ_flat.shape}", end='\n********')

        # project onto KL basis to estimate PSF intensity
        proj = np.dot(targ_flat, Z_trim.T)
        klipped = np.dot(Z_trim.T, proj).reshape(target.shape)

        return klipped

    def _generate_klip_proj(self, cube_list, verbose=True):
        '''
        Generates a HDUList of KLIP projections for every slice of each
        post-padded target image data cube. The result is used in
        `self.plot_subtraction()` and `self.plot_contrasts()`.

        Argument `cube_list` is an HDUList of *aligned*, NaN-less data cubes.
        `self.stackable_cubes` is usually the only appropriate choice here.

        Argument `verbose` is a boolean that, when True, allows the method to
        print progress messages.
        '''
        print_ast = lambda text: print('\n********',
                                       text,
                                       '********', sep='\n')
        my_pr = lambda txt, **kwargs: (print_ast(txt, **kwargs)
                                       if verbose else None)
        my_pr('generating KLIP projections of target images '
              'in `self.klip_proj`...')

        # collect all images in one 4D array.
        # dimensions are: number of ref & tgt data cubes,
        # number of wavelength slices, and the 2D shape of a post-padded image
        cube_list = self._pklcopy(cube_list)
        all_cubes = np.array([cube.data for cube in cube_list])

        # separate the reference and target data cubes
        refs_all = all_cubes[:len(self.positions)]
        tgts_all = all_cubes[len(self.positions):]

        # set up hdulist of klip projections for all slices of all target images
        # (otherwise, has the same HDU structure as stackable_cubes)
        klip_proj = self._pklcopy(cube_list[len(self.positions):])

        # carry out klip projections for all slices of every target image
        # and insert them into the HDUList generated above
        for sl in range(tgts_all.shape[1]): # number of wavelength slices
            refs_sliced = refs_all[:,sl]
            tgts_sliced = tgts_all[:,sl]

            ref_klip_basis = self._get_klip_basis(refs_sliced,
                                                  #explain=.99)
                                                  modes=len(self.positions))

            for j, tg in enumerate(tgts_sliced):
                ref_klip = self._project_onto_basis(tg, ref_klip_basis)
                klip_proj[j].data[sl] = ref_klip

        return klip_proj

    def _generate_theo_klip_proj(self, cube_list, fine_ref_cubes, verbose=True):
        '''
        **Exclusively for theoretically-aligned HDULists.** Produces the same
        output as `self._generate_klip_proj()` -- an HDUList of KLIP
        projections for every slice of each post-padded target image data cube.
        The result is used in `self.plot_subtraction()` and
        `self.plot_contrasts()`.

        Argument `cube_list` is an HDUList of *aligned*, NaN-less data cubes.
        `self.stackable_cubes` is usually the only appropriate argument here;
        its latter half of aligned target images is what will be used here.

        Argument `fine_ref_cubes` is a list of 4D arrays. Each array is a set of
        "fine-aligned" reference cubes that was made to match a certain target.
        The references in index 0 of `fine_ref_cubes` match with the target
        cube at index len(self.positions) of `cube_list`, and so on.

        Argument `verbose` is a boolean that, when True, allows the method to
        print progress messages.
        '''
        print_ast = lambda text: print('\n********',
                                       text,
                                       '********', sep='\n', end='')
        my_pr = lambda txt: print_ast(txt) if verbose else None
        my_pr('generating KLIP projections of target images '
              'in `self.klip_proj`...')

        # set up hdulist of klip projections for all slices of all target images
        # (otherwise, has the same HDU structure as cube_list)
        cube_list = self._pklcopy(cube_list)
        fine_tgt_cubes = [cb.data for cb in cube_list[len(self.positions):]]
        klip_proj = self._pklcopy(cube_list[len(self.positions):])

        # carry out klip projections for all slices of every target image
        # and insert them into the HDUList generated above
        for im in range(len(fine_tgt_cubes)): # number of target images
            # change shape of data array to match current target
            for sl in range(fine_tgt_cubes[im].shape[0]): # wvln slices per cube
                # get all slices of target and assoc. refs at this wavelength
                refs_at_wvln = fine_ref_cubes[im][:, sl]
                tgt_at_wvln = fine_tgt_cubes[im][sl]

                # project the target onto the basis formed by these references
                ref_klip_basis = self._get_klip_basis(refs_at_wvln,
                                                      #explain=.99)
                                                      modes=len(self.positions))
                ref_klip = self._project_onto_basis(tgt_at_wvln, ref_klip_basis)

                # save the result as a slice of this projected target image
                klip_proj[im].data[sl] = ref_klip

        #print(klip_proj.info(), self.stackable_cubes[10:].info())
        return klip_proj

    def _generate_contrasts(self, cube_list, verbose=True):
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

        Argument `cube_list` is an HDUList of *aligned*, NaN-less data cubes.
        `self.stackable_cubes` is usually the only appropriate choice here.

        Argument `verbose` is a boolean that, when True, allows the method to
        print progress messages.

        All of these are normalized by the brightest pixel in the original
        target image. Each of these measurements is its own HDUList with length
        equal to the number of target images in the directory. Each entry is a
        stack of 2D separation/contrast arrays (in that order), the number in
        the stack matches the number of wavelength slices available in
        self.stackable_cubes.
        '''
        print_ast = lambda text: print('\n********',
                                       text,
                                       '********', sep='\n', end='')
        my_pr = lambda txt: print_ast(txt) if verbose else None
        my_pr('generating pre-/post-subtraction contrast curves...')

        pix_len = .1

        # collect all slices of all target images in one list (diff. shapes)
        cube_list = self._pklcopy(cube_list)[len(self.positions):]
        tgt_imgs = [cube.data for cube in cube_list]

        # collect all slices of all KLIP projections in one list (diff. shapes)
        prj_imgs = [cube.data for cube in self.klip_proj]

        # create contrast/separation HDUlists to be filled
        # (will have same headers as stackable_cubes but data will change)
        pre_prof_hdu = self._pklcopy(cube_list)
        post_prof_hdu = self._pklcopy(cube_list)
        photon_prof_hdu = self._pklcopy(cube_list)
        pre_avg_hdu = self._pklcopy(cube_list)

        # get per-slice photon noise values
        num_phot = np.array([self._count_photons(wv=wvl * u.m, dist=40*u.pc)
                             for wvl in self.wvlnths])
        phot_noise_frac = np.sqrt(num_phot) / num_phot

        # create HDULists of post-subtraction and photon noise scenes
        # (cube_list is already an HDU of pre-subtraction target images)
        subt_list = self._pklcopy(cube_list)
        phot_list = self._pklcopy(cube_list)
        for ext, cube in enumerate(cube_list):
            subt_list[ext].data -= prj_imgs[ext]
            phot_list[ext].data = (np.sqrt(phot_list[ext].data)
                                   * phot_noise_frac[:, np.newaxis, np.newaxis])

        # ensure each cube's prof is the same size by finding a max safe radius
        max_rad = np.min([np.max([cb.header['PIXSTARY'],
                                  cb.shape[1] - cb.header['PIXSTARY'],
                                  cb.header['PIXSTARX'],
                                  cb.shape[2] - cb.header['PIXSTARX']])
                          for cb in cube_list]) * pix_len

        # calculate radial profiles at each wavelength, pre & post-subtraction,
        # as well as the radial profile for photon noise and
        # the (radial) average intensity of the original target image
        #start = time.time()
        rad_prof = poppy.radial_profile
        n_imgs = len(tgt_imgs) #tgt_imgs.shape[0]
        n_slcs = tgt_imgs[0].shape[0] #tgt_imgs.shape[1]
        # (first index shouldn't matter; all cubes should have same num. slices)

        for im in range(n_imgs):
            to_pre_prof_hdu = []
            to_post_prof_hdu = []
            to_photon_prof_hdu = []
            to_pre_avg_hdu = []

            # get star's pixel location from header (remember y & x are flipped)
            norm_ind = np.array([cube_list[im].header['PIXSTARX'],
                                 cube_list[im].header['PIXSTARY']])

            for sl in range(n_slcs):
                # calculate radial profiles for each...
                # (incl. average intensity per radius of pre-subtracted target)
                with warnings.catch_warnings(record=True) as w:
                    # ignore warning on std dev of empty array in poppy/utils.py
                    warnings.simplefilter('ignore') # line ~647
                    rad, pre_prof = rad_prof(cube_list, stddev=True,
                                             center=norm_ind, maxradius=max_rad,
                                             ext=im, slice=sl)
                    _, post_prof = rad_prof(subt_list, stddev=True,
                                            center=norm_ind, maxradius=max_rad,
                                            ext=im, slice=sl)
                    _, photon_prof = rad_prof(phot_list, stddev=True,
                                              center=norm_ind,maxradius=max_rad,
                                              ext=im, slice=sl)
                    rad2, pre_avg = rad_prof(cube_list, stddev=False,
                                             center=norm_ind, maxradius=max_rad,
                                             ext=im, slice=sl)

                    # limit stddevs to points with non-NaN, nonzero contrasts
                    # (also ignore warning on using greater-than with NaN)
                    nonz = np.flatnonzero((~np.isnan(pre_prof)) & (pre_prof>0))

                    rad = rad[nonz]
                    pre_prof = pre_prof[nonz]; post_prof = post_prof[nonz]
                    photon_prof = photon_prof[nonz]

                # normalize all profiles, flipping star index back to y by x
                norm = tgt_imgs[im][sl][tuple(norm_ind[::-1].astype(int))]

                pre_prof /= norm; post_prof /= norm
                photon_prof /= norm; pre_avg /= norm

                # join separations with contrasts
                to_pre_prof_hdu.append(np.stack((rad, pre_prof)))
                to_post_prof_hdu.append(np.stack((rad, post_prof)))
                to_photon_prof_hdu.append(np.stack((rad, photon_prof)))
                to_pre_avg_hdu.append(np.stack((rad2, pre_avg)))

            # append the current image's data in each hdulist's matching entry
            for att in range(2):
                try:
                    pre_prof_hdu[im].data = np.array(to_pre_prof_hdu)
                    post_prof_hdu[im].data = np.array(to_post_prof_hdu)
                    photon_prof_hdu[im].data = np.array(to_photon_prof_hdu)
                    pre_avg_hdu[im].data = np.array(to_pre_avg_hdu)

                    if att != 0:
                        my_pr('trim successful.')
                    break
                # if array conversion throws an error (typically for different-
                # length entries), only keep separations common to each slice
                # (retry after except clause to ensure the error was remedied)
                # MAY NOT BE NECESSARY WITH NEW maxradius OPTION in rad_profile?
                except ValueError as e:
                    if att != 0:
                        raise(e)

                    my_pr('\ndifferent length radial_profile results in '
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

        my_pr('completed `self.pre_prof_hdu`, `self.post_prof_hdu`, '
              '`self.photon_prof_hdu`, and `self.pre_avg_hdu`.')
        return pre_prof_hdu, post_prof_hdu, photon_prof_hdu, pre_avg_hdu
