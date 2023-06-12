#!/usr/bin/env python3

import pathlib
from collections.abc import Iterable
import itertools
import pickle

import numpy as np
from croaks.match import NearestNeighAssoc
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
import imageproc.composable_functions as compfuncs
import saunerie.fitparameters as fp
from scipy import sparse
from croaks import DataProxy
from sksparse import cholmod
import matplotlib.pyplot as plt
import matplotlib.patches as patches


filtercodes = ['zg', 'zr', 'zi']
quadrant_width_px, quadrant_height_px = 3072, 3080
quadrant_size_px = {'x': quadrant_width_px, 'y': quadrant_height_px}

ztf_longitude = -116.8598 # deg
ztf_latitude = 33.3573 # deg E
ztf_altitude = 1668. # m

gaiarefmjd = Time(2015.5, format='byear').mjd
j2000mjd = Time(2000.0, format='jyear').mjd

filtercode2extcatband = {'gaia': {'zg': 'BPmag',
                                  'zr': 'RPmag',
                                  'zi': 'RPmag'}, # Not a great workaround
                         'ps1': {'zg': 'gmag',
                                 'zr': 'rmag',
                                 'zi': 'zmag'}}

filtercode2ztffid = {'zg': 1,
                     'zr': 2,
                     'zi': 3}

extcat2colorstr = {'gaia': "B_p-R_p",
                   'ps1': "m_g-m_i"}

idx2markerstyle = ['*', 'x', '.', 'v', '^']


def plot_ztf_focal_plan(fig, focal_plane_dict, plot_fun, plot_ccdid=False):
    #ccds = fig.subfigures(ncols=4, nrows=4, hspace=0.09, wspace=0.09)
    # ccds = fig.subfigures(ncols=4, nrows=4, hspace=0., wspace=0.)
    ccds = fig.add_gridspec(4, 4, wspace=0.02, hspace=0.02)
    for i in range(4):
        for j in range(4):
            ccdid = 16 - (i*4+j)
            # quadrants = ccds[i, j].subplots(ncols=2, nrows=2, gridspec_kw={'hspace': 0., 'wspace': 0.})
            quadrants = ccds[i, j].subgridspec(2, 2, wspace=0., hspace=0.)
            axs = quadrants.subplots()

            for k in range(2):
                for l in range(2):
                    rcid = (ccdid-1)*4 + k*2
                    qid = k*2
                    if k > 0:
                        rcid += l
                        qid += l
                    else:
                        rcid -= (l - 1)
                        qid -= (l - 1)

                    #plot_fun(quadrants[k, l], focal_plane_dict[ccdid][qid], ccdid, qid, rcid)
                    plot_fun(axs[k, l], focal_plane_dict[ccdid][qid], ccdid, qid, rcid)

            if plot_ccdid:
                # ax = ccds[i, j].add_subplot()
                ax = fig.add_subplot(ccds[i, j])
                ax.text(0.5, 0.5, ccdid, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontweight='black', fontsize='xx-large')
                ax.axis('off')
                # rect = patches.Rectangle((0.02, 0.02), 0.98, 0.98, linewidth=2, edgecolor='black', facecolor='none')
                # ax.add_patch(rect)

    for ax in fig.get_axes():
        ss = ax.get_subplotspec()
        ax.spines.top.set_visible(ss.is_first_row())
        ax.spines.top.set(linewidth=1.)
        ax.spines.bottom.set_visible(ss.is_last_row())
        ax.spines.bottom.set(linewidth=1.)
        ax.spines.left.set_visible(ss.is_first_col())
        ax.spines.left.set(linewidth=1.)
        ax.spines.right.set_visible(ss.is_last_col())
        ax.spines.right.set(linewidth=1.)



def plot_ztf_focal_plan_rcid(fig):
    rcids = dict([(i+1, dict([(j, j) for j in range(4)])) for i in range(0, 16)])

    def _plot(ax, val, ccdid, qid, rcid):
        ax.text(0.5, 0.5, rcid, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        # ax.set_axis_off()
        # ax.axis('off')
        # ax.set_xticks([])
        # ax.set_yticks([])

    plot_ztf_focal_plan(fig, rcids, _plot, plot_ccdid=True)


def plot_ztf_focal_plan_values(fig, focal_plane_dict, scalar=False, vmin=None, vmax=None, cmap=None):
    def _plot(ax, val, ccdid, qid, rcid):
        if val is not None:
            if scalar:
                val = [[val]]

            ax.imshow(val, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set(xticks=[], yticks=[])
            ax.set_aspect('auto')

        ax.text(0.5, 0.5, rcid+1, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    if vmin is None or vmax is None:
        values = list(itertools.chain.from_iterable([[focal_plane_dict[ccdid][qid] for qid in focal_plane_dict[ccdid]] for ccdid in focal_plane_dict.keys()]))
        values = list(filter(lambda x: x is not None, values))

        if vmin is None:
            vmin = np.min(values)

        if vmax is None:
            vmax = np.max(values)

    plot_ztf_focal_plan(fig, focal_plane_dict, _plot, plot_ccdid=True)


def make_index_from_list(dp, index_list):
    """
    Calls make_index() of DataProxy dp on all index names in index_list
    """
    [dp.make_index(index) for index in index_list]


def match_pixel_space(refcat, cat, radius=1.):
    def _euclidian(x1, y1, x2, y2):
        return np.sqrt((np.array(x1) - np.array(x2)) ** 2 + (np.array(y1) - np.array(y2)) ** 2)

    assoc = NearestNeighAssoc(first=[refcat['x'], refcat['y']], radius=radius)
    index = assoc.match(cat['x'], cat['y'], metric=_euclidian)
    return index


def get_cat_size(catalog_filename):
    with open(catalog_filename, 'r') as f:
        _, cat = read_list(f)

    return len(cat)


def ztfquadrant_center(wcs):
    return wcs.pixel_to_world([quadrant_width_px/2.], [quadrant_height_px/2.])

def match_to_gaia(cat_df, cat_gaia_df):
    i = match_pixel_space(cat_gaia_df, cat_df)
    return cat_gaia_df.iloc[i[i>=0]].gaiaid


def match_by_gaia_catalogs(cat1_df, cat2_df, cat1_gaia_df, cat2_gaia_df):
    cat1_gaia_df.set_index('gaiaid', inplace=True)
    cat2_gaia_df.set_index('gaiaid', inplace=True)
    cat1_df.set_index(cat1_gaia_df.index, inplace=True)
    cat2_df.set_index(cat2_gaia_df.index, inplace=True)

    indices = [idx for idx in cat1_stars.index if idx in cat2_stars.index]

    return cat1_stars.loc[indices], cat2_stars.loc[indices], cat1_gaia.loc[indices]


def apply_space_motion(ra, dec, pm_ra, pm_dec, refmjd, newmjd):
    reftime = Time(refmjd, format='mjd')
    newtime = Time(newmjd, format='mjd')
    #coords = SkyCoord(ra*u.deg, dec*u.deg, pm_ra_cosdec=pm_ra*np.cos(dec/180.*np.pi)*u.mas/u.year, pm_dec=pm_dec*u.mas/u.year, obstime=reftime)
    coords = SkyCoord(ra*u.deg, dec*u.deg, pm_ra_cosdec=np.zeros_like(ra)*u.mas/u.year, pm_dec=np.zeros_like(ra)*u.mas/u.year, obstime=reftime)
    coords.apply_space_motion(newtime)
    return np.stack([coords.frame.data.lon.value, coords.frame.data.lat.value], axis=1)


def quadrant_name_explode(quadrant_name, kind='sci'):
    quadrant_name = "_".join(quadrant_name.split("_")[1:])

    if kind == 'sci':
        year = int(quadrant_name[0:4])
        month = int(quadrant_name[4:6])
        day = int(quadrant_name[6:8])
        field = int(quadrant_name[15:21])
        filterid = quadrant_name[22:24]
        ccdid = int(quadrant_name[26:28])
        qid = int(quadrant_name[32])

        return year, month, day, field, filterid, ccdid, qid

    elif kind == 'raw':
        assert(False, "quadrant_name_explode(): needs to be updates/tested for raw image name")
        year = int(quadrant_name[4:8])
        month = int(quadrant_name[8:10])
        day = int(quadrant_name[10:12])
        field = int(quadrant_name[19:25])
        filterid = quadrant_name[26:28]
        ccdid = int(quadrant_name[30:32])

        return year, month, day, field, filterid, ccdid

    else:
        raise NotImplementedError("Kind {} not implemented!".format(kind))


def get_wcs_from_quadrant(quadrant_path):
    #with fits.open(quadrant_path.joinpath("calibrated.fits")) as hdul:
    #    wcs = WCS(hdul[0].header)
    #return wcs

    hdr = get_header_from_quadrant_path(quadrant_path)
    return WCS(hdr)



def get_mjd_from_quadrant_path(quadrant_path):
    with fits.open(quadrant_path.joinpath("calibrated.fits")) as hdul:
        return hdul[0].header['obsmjd']


def get_header_from_quadrant_path(quadrant_path, key=None):
    if quadrant_path.joinpath("calibrated.fits").exists():
        with fits.open(quadrant_path.joinpath("calibrated.fits")) as hdul:
            if key is None:
                return hdul[0].header
            else:
                return hdul[0].header[key]
    elif quadrant_path.joinpath("calibrated_hdr").exists():
        with fits.open(quadrant_path.joinpath("calibrated_hdr")) as hdul:
            return hdul[0].header
    elif quadrant_path.joinpath("calibrated_header.pickle").exists():
        with open(quadrant_path.joinpath("calibrated_header.pickle"), 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError("Could not find calibrated.fits nor calibrated_hdr! {}".format(quadrant_path))


def sc_ra(skycoord):
    return skycoord.frame.data.lon.value


def sc_dec(skycoord):
    return skycoord.frame.data.lat.value

def sc_array(skycoord):
    return np.array([skycoord.frame.data.lon.value, skycoord.frame.data.lat.value])


def contained_in_exposure(objects, wcs, return_mask=False):
    width, height = wcs.pixel_shape

    if isinstance(objects, pd.DataFrame):
        mask = (objects['x'] >= 0.) & (objects['x'] < width) & (objects['y'] >= 0.) & (objects['y'] < height)
    else:
        top_left = [0., height]
        top_right = [width, height]
        bottom_left = [0., 0.]
        bottom_right = [width, 0]

        tl_radec = sc_array(wcs.pixel_to_world(*top_left))
        tr_radec = sc_array(wcs.pixel_to_world(*top_right))
        bl_radec = sc_array(wcs.pixel_to_world(*bottom_left))
        br_radec = sc_array(wcs.pixel_to_world(*bottom_right))

        tl = [max([tl_radec[0], bl_radec[0]]), min([tl_radec[1], tr_radec[1]])]
        br = [min([tr_radec[0], br_radec[0]]), max([bl_radec[1], br_radec[1]])]

        #tl = (max([sc_ra(tl_radec), sc_ra(bl_radec)]), min([sc_dec(tl_radec), sc_dec(tr_radec)]))
        #br = (min([sc_ra(tr_radec), sc_ra(br_radec)]), max([sc_dec(bl_radec), sc_dec(br_radec)]))

        o = sc_array(objects)
        if tl[0] < br[0]:
            mask = (o[0] > tl[0]) & (o[0] < br[0]) & (o[1] > tl[1]) & (o[1] < br[1])
            #br[0], tl[0] = tl[0], br[0]
        else:
            mask = (o[0] < tl[0]) & (o[0] > br[0]) & (o[1] > tl[1]) & (o[1] < br[1])

    if return_mask:
        return mask
    else:
        return objects[mask]


def get_ref_quadrant_from_driver(driver_path):
    if not driver_path.exists():
        return

    with open(driver_path, 'r') as f:
        found_phoref = False
        line = f.readline()
        while line:
            if "PHOREF" in line[:-1]:
                found_phoref = True
                break

            line = f.readline()

        if not found_phoref:
            return

        reference_quadrant = f.readline()

        if reference_quadrant:
            return pathlib.Path(reference_quadrant[:-1]).name


def get_ref_quadrant_from_band_folder(band_path):
    with open(band_path.joinpath("reference_quadrant")) as f:
        return f.readline()


def poly2d_from_file(filename):
    with open(filename, 'r') as f:
        f.readline()
        deg_str = f.readline()[:-1]
        degree = int(deg_str.split(" ")[1])
        coeff_str = " ".join(f.readline()[:-1].split())
        coeffs = list(map(float, coeff_str.split(" ")))

    coeffs_1 = coeffs[:int(len(coeffs)/2)]
    coeffs_2 = coeffs[int(len(coeffs)/2):]

    def _extract_coeffs(coeffs):
        idx = 0
        c = np.zeros([degree, degree])
        for d in range(degree):
            p, q = d, 0
            while p >= 0:
                c[p, q] = coeffs[idx]
                idx += 1
                p -= 1
                q += 1

        return c

    c_1 = _extract_coeffs(coeffs_1)
    c_2 = _extract_coeffs(coeffs_2)

    def _apply_pol(x, y):
        return np.stack([np.polynomial.polynomial.polyval2d(x, y, c_1),
                         np.polynomial.polynomial.polyval2d(x, y, c_2)])

    return _apply_pol


def poly2d_to_file(poly2d, quadrant_set, outdir):
    """
    dump the fitted transformations (in the poloka format)
    so that they can be understood by mklc
    """

    outdir.mkdir(exist_ok=True)

    for i, quadrant in enumerate(quadrant_set):
        #with open(outdir + os.sep + 'transfoTo' + expid + 'p' + ccd + '.dat', 'w') as f:
        with open(outdir.joinpath("transfoTo{}.dat".format(quadrant)), 'w') as f:
            deg = poly2d.bipol2d.deg
            f.write("GtransfoPoly 1\ndegree %d\n" % deg)

            coeff_name = dict(list(zip(poly2d.bipol2d.coeffs, [x for x in poly2d.bipol2d.coeffnames if 'alpha' in x])))
            for d in range(deg+1):
                p,q = d,0
                while p>=0:
                    nm = coeff_name[(p,q)]
                    scaled_par = poly2d.params[nm].full[i]
                    f.write(" %15.20E " % scaled_par)
                    p -= 1
                    q += 1
            coeff_name = dict(list(zip(poly2d.bipol2d.coeffs, [x for x in poly2d.bipol2d.coeffnames if 'beta' in x])))
            for d in range(deg+1):
                p,q = d,0
                while p>=0:
                    nm = coeff_name[(p,q)]
                    scaled_par = poly2d.params[nm].full[i]
                    f.write(" %15.20E " % scaled_par)
                    p -= 1
                    q += 1
            f.write('\n')


def read_list(f):
    if isinstance(f, str) or isinstance(f, pathlib.Path):
        with open(f, 'r') as fi:
            header, df, _, _ = read_list_ext(fi)

    else:
        header, df, _, _ = read_list_ext(f)

    return header, df


def read_list_ext(f):
    # Extract global @ parameters
    header = {}
    line = f.readline().strip()
    curline = 1
    while line[0] == "@":
        # line = line[:-1]
        splitted = line.split()
        key = splitted[0][1:]
        splitted = splitted[1:]

        first = splitted[0]
        if first[0] == '-':
            first = first[1:]

        if first.isdigit():
            t = int
        else:
            try:
                float(first)
            except ValueError:
                t = str
            else:
                t = float

        values = list(map(t, splitted))

        if len(values) == 1:
            values = values[0]

        header[key.lower()] = values

        line = f.readline()
        curline += 1

    # Extract dataframe
    # First, column names
    columns = []
    df_format = None
    df_desc = {}
    while line[0] == "#":
        curline += 1
        line = line[1:-1].strip()

        splitted = line.split(" ")
        if splitted[0].strip() == "end":
            break

        if splitted[0].strip() == "format":
            df_format = str(" ".join(line.split(" ")[1:])).strip()
            line = f.readline()
            continue

        splitted = line.split(":")
        column = splitted[0].strip()
        columns.append(column)
        if len(splitted) > 1:
            df_desc[column] = splitted[1].strip()

        line = f.readline()

    df = pd.read_csv(f, delim_whitespace=True, names=columns, index_col=False, skipinitialspace=True)

    return header, df, df_desc, df_format


def write_list(filename, header, df, df_desc, df_format):
    with open(filename, 'w') as f:
        if header:
            for key in header.keys():
                if isinstance(header[key], Iterable) and not isinstance(header[key], str):
                    f.write("@{} {}\n".format(key.upper(), " ".join(map(str, header[key]))))
                else:
                    f.write("@{} {}\n".format(key.upper(), str(header[key])))

        for column in df.columns:
            if column in df_desc.keys():
                f.write("#{} : {}\n".format(column, df_desc[column]))
            else:
                f.write("#{} :\n".format(column))

        if df_format is not None:
            f.write("# format {}\n".format(df_format))

        f.write("# end\n")

        df.to_csv(f, sep=" ", index=False, header=False)


class ListTable:
    def __init__(self, header, df, df_desc=None, df_format=None, filename=None):
        if not df_desc:
            df_desc = {}

        self.header = header
        self.df = df
        self.df_desc = df_desc
        self.df_format = df_format
        self.filename = filename

    @classmethod
    def from_filename(cls, filename):
        with open(filename, 'r') as f:
            header, df, df_desc, df_format = read_list_ext(f)

        return cls(header, df, df_desc, df_format, filename=filename)

    def write_to(self, filename):
        write_list(filename, self.header, self.df, self.df_desc, self.df_format)

    def write(self):
        self.write_to(self.filename)


class BiPol2DModel():
    def __init__(self, degree, space_count=1):
        self.bipol2d = compfuncs.BiPol2D(deg=degree, key='space', n=space_count)

        self.params = fp.FitParameters([*self.bipol2d.get_struct()])

    def __get_coeffs(self):
        return dict((key, self.params[key].full) for key in self.params._struct.slices.keys())

    def __set_coeffs(self, coeffs):
        for key in coeffs.keys():
            self.params[key] = coeffs[key]

    coeffs = property(__get_coeffs, __set_coeffs)

    def __call__(self, x, p=None, space_indices=None, jac=False):
        if p is not None:
            self.params.free = p

        if space_indices is None:
            space_indices = [0]*x.shape[1]

        # Evaluate polynomials
        if not jac:
            return self.bipol2d(x, self.params, space=space_indices)
        else:
            xy, _, (i, j, vals) = self.bipol2d.derivatives(x, self.params, space=space_indices)

            ii = np.hstack([i, i+x[0].shape[0]])
            jj = np.tile(j, 2).ravel()
            vv = np.hstack(vals).ravel()

            ii = np.hstack(ii)
            jj = np.hstack(jj)
            vv = np.hstack(vv)

            J_model = sparse.coo_array((vv, (ii, jj)), shape=(2*xy.shape[1], len(self.params.free)))

            return xy, J_model

    def fit(self, x, y, space_indices=None):
        _, J = self.__call__(x, space_indices=space_indices, jac=True)
        H = J.T @ J
        B = J.T @ np.hstack(y)

        fact = cholmod.cholesky(H.tocsc())
        p = fact(B)
        self.params.free = p
        return p

    def residuals(self, x, y, space_indices=None):
        y_model = self.__call__(x, space_indices=space_indices)
        return y_model - y

    def control_plots(self, x, y, folder, space_indices=None):
        res = self.residuals(x, y, space_indices=space_indices)
        plt.subplots(ncols=2, nrows=1, figsize=(10., 5.))
        plt.subplot(2, 1, 1)
        #plt.hist(res[0], bins='sturges')
        plt.hist(res[0], range=[-0.00005, 0.00005], bins=100)
        plt.grid()
        plt.subplot(2, 1, 2)
        #plt.hist(res[1], bins='sturges')
        plt.hist(res[1], range=[-0.00005, 0.00005], bins=100)
        plt.grid()
        plt.savefig(folder.joinpath("residual_distribution.png"), dpi=300.)
        plt.close()


def BiPol2D_fit(x, y, degree, space_indices=None, control_plots=None, simultaneous_fit=True):
    if space_indices is None:
        space_count = 1
    else:
        space_count = len(set(space_indices))

    if simultaneous_fit:
        model = BiPol2DModel(degree, space_count=space_count)
        model.fit(x, y, space_indices=space_indices)
    else:
        space_models_coeffs = []
        for space_index in range(space_count):
            space_model = BiPol2DModel(degree, space_count=1)
            space_model.fit(x[:, space_indices==space_index], y[:, space_indices==space_index])
            space_models_coeffs.append(space_model.coeffs)

        model = BiPol2DModel(degree, space_count=space_count)
        for space_index in range(space_count):
            for coeff in model.coeffs.keys():
                model.coeffs[coeff][space_index] = space_models_coeffs[space_index][coeff]

    if control_plots:
        model.control_plots(x, y, control_plots, space_indices=space_indices)

    return model


def create_2D_mesh_grid(*meshgrid_space):
    meshgrid = np.meshgrid(*meshgrid_space)
    return np.array([meshgrid[0], meshgrid[1]]).T.reshape(-1, 2)
