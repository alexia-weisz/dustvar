import astropy.units as u
import numpy as np
import astropy.wcs as wcs


pixel_area = np.product(np.abs(wcs.celestial.pixel_scale_matrix.diagonal()))*u.deg**2
