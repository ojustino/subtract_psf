#!/usr/bin/env python3
import numpy as np

from astropy.table import Table
from astropy import units as u
from my_warnings import warnings


class InjectCompanion:
    '''
    This class is meant to be inherited by a KlipRetrieve() or PreInjectImages()
    class instance, not used individually.

    Contains the key method that performs companion injections and its
    associated support methods. For more aboput the injection process, see the
    docstring for `self.inject_companion()`.
    '''

    def _check_spectrum(self, arg, format):
        '''
        Validates a spectrum passed to `self.scale_from_spectra()` based on the
        conditions described in that method's docstring. If it passes, gets the
        Table ready for flux binning in `self._get_bin_fluxes()`.
        '''
        if isinstance(arg, str):
            spectrum = Table.read(spectrum, format=format)
        else:
            spectrum = arg.copy()

        # rename columns
        spectrum.rename_column(spectrum.colnames[0], 'wvln')
        spectrum.rename_column(spectrum.colnames[1], 'flux')

        # check for units
        if spectrum['wvln'].unit is None or spectrum['flux'].unit is None:
            raise ValueError('Both columns must have units.')

        # convert/add wavelength and flux units
        spectrum['wvln'] = spectrum['wvln'].to(u.micron)
        spectrum['flux'] = spectrum['flux'].to(u.mJy, u.spectral_density(spectrum['wvln'].quantity))

        # sort by increasing wavelength (just to be sure)
        sort_by_wv = np.argsort(spectrum['wvln'])
        spectrum = spectrum[sort_by_wv]

        wvln_column = spectrum['wvln'].quantity
        flux_column = spectrum['flux'].quantity

        # cut table to nirspec wavelengths with breathing room on both edges
        # of half a slice's (wavelength step's) size
        step_size = self.wvlnths[1] - self.wvlnths[0]
        lo_ind = np.searchsorted(wvln_column,
                                 (self.lo_wv - step_size/2) * u.m, side='left')
        hi_ind = np.searchsorted(wvln_column,
                                 (self.hi_wv + step_size/2) * u.m, side='right')

        spectrum = spectrum[lo_ind:hi_ind]

        return spectrum

    def _get_bin_fluxes(self, spectrum):
        '''
        Divides a spectrum Table into bins based on wavelengths in self.wvlnths,
        calculates the mean flux in each bin, then returns the resulting array
        of binned flxues.
        '''
        wvln_column = spectrum['wvln'].quantity
        flux_column = spectrum['flux'].quantity

        # get step width between slices (wavelengths) -- should be constant
        step_size = self.wvlnths[1] - self.wvlnths[0]

        # create array with (wavelength) bin edges. bin and step width are equal
        # (wvlnth[x]'s edges are bin_edges[x:x+1+1])
        bin_edges = np.linspace(self.lo_wv - step_size / 2,
                                self.hi_wv + step_size / 2,
                                len(self.wvlnths) + 1, endpoint=True)

        # find where in (sorted) column of wavelength each edge falls
        bin_inds = np.searchsorted(wvln_column, bin_edges * u.m, side='right')
        # (same behavior as bisect_right)

        # fill an array of mean fluxes in each bin
        bin_fluxes = np.zeros(len(self.wvlnths)) * u.mJy
        for i in range(len(bin_inds) - 1):
            # record mean of fluxes whose wavelengths fall in this bin
            if bin_inds[i + 1] - bin_inds[i] > 0:
                bin_fluxes[i] = flux_column[bin_inds[i]
                                            : bin_inds[i + 1]].mean()
            # if there are no fluxes in this bin, record mean of fluxes
            # at the previous flux_column index and the current one
            else:
                bin_fluxes[i] = flux_column[bin_inds[i] - 1
                                            : bin_inds[i] + 1].mean()

        return bin_fluxes

    def _choose_random_sep(self, target_images):
        '''
        Called from `self._inject_companion()` if the user didn't specify a
        companion separation. Calculates separations that will safely remain
        in-frame for all pointings, then randomly chooses Y and X pixel
        distances from the set that remains.

        Returns those pixel distances along with the resulting distance
        magnitude in arcseconds.
        '''
        pix_len = .1

        # limit to pixels that aren't within N pixels of star or the edge
        edge_gap = 2; st_gap = 1 #pixels
        min_sep = np.round(np.sqrt(2 * ((st_gap + 1) * pix_len)**2), 1)

        # find indices of brightest pixel in each cube's 0th slice
        # (all slices of a given image cube should have the same bright spot)
        star_locs = np.array([np.unravel_index(np.argmax(target_images[i, 0]),
                                               target_images[i, 0].shape)[::-1]
                              for i in range(target_images.shape[0])])

        # in each image, get indices of possible pixels in each direction,
        # excluding ones too close to the edge
        poss_y = np.arange(target_images.shape[-2])[edge_gap:-edge_gap]
        poss_x = np.arange(target_images.shape[-1])[edge_gap:-edge_gap]

        # in each image, get the maximum safe pixel distance in each direction
        # (i.e. how many pixels from the star is the nearest edge in x and y?)
        max_ys = [np.abs(st_y - poss_y[-1])
                  if np.abs(st_y - poss_y[-1]) < np.abs(st_y - poss_y[0])
                  else np.abs(st_y - poss_y[0])
                  for st_y in star_locs[:,0]]

        max_xs = [np.abs(st_x - poss_x[-1])
                  if np.abs(st_x - poss_x[-1]) < np.abs(st_x - poss_x[0])
                  else np.abs(st_x - poss_x[0])
                  for st_x in star_locs[:,0]]

        # next, exclude pixels too close to the star
        max_safe_y = int(np.min(max_ys)); max_safe_x = int(np.min(max_xs))
        #print('max_safes', max_safe_y, max_safe_x)
        max_safe_sep = np.round(np.hypot(max_safe_x, max_safe_y))

        below_max_y = np.arange(-max_safe_y, max_safe_y + 1).astype(int)
        below_max_x = np.arange(-max_safe_x, max_safe_x + 1).astype(int)

        poss_dists_y = np.delete(below_max_y,
                                 np.s_[max_safe_y - st_gap
                                       : max_safe_y + st_gap + 1])
        poss_dists_x = np.delete(below_max_x,
                                 np.s_[max_safe_x - st_gap
                                       : max_safe_x + st_gap + 1])
        #print(poss_dists_x)
        #print(poss_dists_y)

        # choose companion x/y distances based on remaining, safe pixels
        dist_y = np.random.choice(poss_dists_y)
        dist_x = np.random.choice(poss_dists_x)
        dist_arc = np.hypot(dist_x * pix_len, dist_y * pix_len)
        #print(max_safe_sep, dist_y, dist_x)

        return dist_y, dist_x, dist_arc

    def inject_companion(self, cube_list, comp_scale=None, return_fluxes=False,
                         star_spectrum=None, comp_spectrum=None,
                         star_format=None, comp_format=None,
                         separation=None, position_angle=np.pi/8, verbose=True):
        '''
        There are two options for scaling. First, argument `comp_scale` is an
        int/float and will make the companion's flux X times the standard
        deviation of pixel intensity at the specified radial separation from
        the star in the pre-subtraction image. (Radial profile information comes
        from `self.pre_prof_hdu`.)

        Second, arguments `comp_spectrum` and `star_spectrum` should either be
        astropy Table objects *or* string paths to spectra files that can be
        read in as Tables. If you go the string route, you must also provide the
        proper format as argument `comp_format` or `star_format`. (See
        astropy's Table.read() documentation for more on acceptable formats.)

        Whichever route you take, the tables/files **must**:
            - have two columns of data. The first column should contain
            wavelengths; the second should contain fluxes.
            - have wavelength units of microns or something equivalent.
            - have flux density units of erg / s / cm**2 / Hz, erg / s / cm**3,
            mJy, or something equivalent.

        To get back the binned spectra, set argument `return_fluxes` to True.
        The method will then return, in order, the HDUList of injected data
        cubes, the star's binned spectra, and the companion's binned spectra.

        Argument `separation` is a float that represents the separation of the
        companion from the star in arcseconds. (Note that the value is rounded
        to the nearest tenth.) If it is `None`, the method will randomly choose
        a companion location that's safely in the image's frame.

        Argument `position_angle` is the companion's position angle in radians,
        relative to the star. It only has an effect if you've specified a
        separation.

        Argument `verbose` is a boolean that, when True, allows the method to
        print progress messages.

        The method will always return the new HDUList of injected data cubes.
        '''
        # gives PSF + PSF_shifted * F_planet/F_star, so star's PSF always ~= 1.

        print_ast = lambda text: print('\n********',
                                       text,
                                       '********', sep='\n', end='')
        my_pr = lambda txt: print_ast(txt) if verbose else None

        # find which child of this class is calling the method
        _is_klip_retrieve = 'KlipRetrieve' in repr(self)

        # I. Check on how the companion will be scaled
        if star_spectrum is not None and comp_spectrum is not None:
            if comp_scale is None:
                # validate both spectra. if str, open file. else, assumes Table
                star_spec = self._check_spectrum(star_spectrum, star_format)
                comp_spec = self._check_spectrum(comp_spectrum, comp_format)
                got_spectra = True
            else:
                raise ValueError('You must either provide `comp_scale` OR a '
                                 '`star_spectrum` and a `comp_spectrum`. All '
                                 'three were provided.')
        elif comp_scale is not None:
            if _is_klip_retrieve:
                # continue on with comp_scale
                got_spectra = False
            else:
                raise ValueError('When calling from a `PreInjectImages()` '
                                 'instance, you must include spectra; scaling '
                                 'by scene\'s standard deviation is '
                                 'unavailable.')
        else:
            raise ValueError('You must either provide `comp_scale` OR a '
                             '`star_spectrum` and a `comp_spectrum`.')

        # collect all cube data in one array
        cube_list = self._pklcopy(cube_list)[len(self.positions):]
        tgt_imgs = np.array([cube.data for cube in cube_list])

        msg1 = ('spectrally defined ' if got_spectra
                else f"(location-specific) {comp_scale:.0f}-sigma ")
        msg2 = (', after alignment' if _is_klip_retrieve
                else ' into unaligned images')
        my_pr('injecting companion with ' + msg1 + 'intensity' + msg2 + '.')

        # II. Translate the companion images
        if separation is None:
            # randomly generate the companion's x/y separation
            s_y, s_x, separation = self._choose_random_sep(tgt_imgs)
        else:
            pix_len = .1
            pix_sep = np.round(separation / pix_len)
            theta = position_angle

            # trigonometrically convert separation magnitude to x/y separations
            s_y = -np.round(pix_sep * np.sin(theta)).astype(int)
            # ("up" in y is + on a graph but negative in an array's 0th dim.)
            s_x = np.round(pix_sep * np.cos(theta)).astype(int)

        # shift a copy of the star's PSF to the specified companion position
        # (pad doesn't accept negatives, so we must add zeros/slice creatively)
        cmp_imgs = np.pad(tgt_imgs, mode='constant', pad_width=
                          ((0,0), (0,0),
                           (s_y if s_y > 0 else 0, -s_y if s_y < 0 else 0),
                           (s_x if s_x > 0 else 0, -s_x if s_x < 0 else 0))
                         )[:, :,
                           -s_y if s_y < 0 else 0: -s_y if s_y > 0 else None,
                           -s_x if s_x < 0 else 0: -s_x if s_x > 0 else None]

        # warn that the companion might be off-frame if most flux is gone
        if (cmp_imgs.sum(axis=(2,3)) < .2).sum() != 0:
            warnings.warn('The companion may be off-frame in some slices. '
                          'Try reducing `separation` if this is undesirable.')

        # III. Scale the companion images
        if got_spectra:
            # get binned fluxes based on spectra and take their ratio
            star_fluxes = self._get_bin_fluxes(star_spec)
            comp_fluxes = self._get_bin_fluxes(comp_spec)
            slices_scaled_1x = (comp_fluxes / star_fluxes).value

            # extend the contrast array to repeat once per data cube
            slices_scaled = np.tile(slices_scaled_1x, (tgt_imgs.shape[0], 1))
            # (tgt_imgs.shape[0] x tgt_imgs.shape[1]), or (n_cubes x n_slices)
        else:
            # get pre-subtraction separation & flux std. dev data from all cubes
            pre_prof_data = np.array([cb.data for cb in self.pre_prof_hdu])
            arc_seps = pre_prof_data[:,:,0]
            flux_stds = pre_prof_data[:,:,1]

            # in all slices, get flux std. dev at given arcsecond separation
            rad_dist_inds = np.argmin(np.abs(arc_seps - separation), axis=2)
            slice_stds = flux_stds[np.arange(arc_seps.shape[0])[:,np.newaxis],
                                   np.arange(arc_seps.shape[1]),
                                   rad_dist_inds]

            # scale to turn those into `comp_scale`-sigma fluxes
            slices_scaled = slice_stds * comp_scale

        # NB: in the future, might need to scale BOTH the star and the companion
        # (that way you can see absorption features in the star and you don't
        #  mistake wavelengths where the star gets fainter as emission features
        #  for the companion).
        # this would lead to different total fluxes depending on slice of data
        # cube, which would mess with the current scaling in plot_subtraction().
        # would also probably be a good idea to multiply stackable cubes by the
        # stellar spectrum once that's implemented so the same behavior is
        # present in both the companion=False/True cases of plot_subtraction()

        # IV. Build a new HDUList with the injected images
        # in each cube, multiply each wavelength slice by its respective scaling
        # (add new axes to slices_scaled so dims work for array broadcasting)
        cmp_imgs *= slices_scaled[:, :, np.newaxis, np.newaxis]
        #cmp_imgs *= slices_scaled.reshape(slices_scaled.shape + (1,1))]

        # simulate the injection by summing the original and companion cubes
        inj_imgs = tgt_imgs + cmp_imgs

        # create the HDUList that will the hold new, injected target images
        try:
            inj_cubes = self._pklcopy(self.stackable_cubes[len(tgt_imgs):])
        except AttributeError:
            inj_cubes = self._pklcopy(self.data_cubes[len(tgt_imgs):])

        # Copy these injected images and associated info into the new data cube
        for i, cube in enumerate(inj_cubes):
            cube.data = inj_imgs[i]

            cube.header['PIXCOMPY'] = (s_y, "this + PIXSTARY is companion's Y "
                                       'pixel location')
            cube.header['PIXCOMPX'] = (s_x, "this + PIXSTARX is companion's X "
                                       'pixel location')

            if got_spectra:
                for n, ratio in enumerate(slices_scaled_1x):
                    keyw = 'CONT' + f"{n:04d}"
                    cube.header[keyw] = (ratio, f"slice {n:}'s companion to "
                                         'star flux ratio')
            else:
                cube.header['XSIGMA'] = (comp_scale, 'comp. flux / stddev of '
                                         'scene flux at separation')

        # if an injection has already occurred, replace it with this new one
        if hasattr(self, 'injected_cubes'):
            self.injected_cubes = inj_cubes
            my_pr('new, injected target images in `self.injected_cubes`.')

        # return new cubes (and fluxes, if requested)
        if return_fluxes:
            try:
                return inj_cubes, star_fluxes, comp_fluxes
            except NameError:
                warnings.warn('No spectra were provided; no fluxes to return.')

        else:
            return inj_cubes
