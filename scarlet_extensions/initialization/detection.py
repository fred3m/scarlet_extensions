import sep
import numpy as np
import scarlet
from scarlet.wavelet import Starlet
from scipy.stats import median_absolute_deviation as mad

def mad_wavelet(image):
    """ image: Median absolute deviation of the first wavelet scale.
    (WARNING: sorry to disapoint, this is not a wavelet for mad scientists)
    Parameters
    ----------
    image: array
        An image or cube of images
    Returns
    -------
    mad: array
        median absolute deviation for each image in the cube
    """
    if len(np.shape(image))==2:
        image = image[None, :, :]
    sigma = []
    for i in image:
        sigma.append(mad(Starlet.from_image(i, scales=2).coefficients[:, 0, ...], axis=(-2, -1)))
    return np.array(sigma)


# Class to provide compact input of instrument data and metadata
class Data:
    """ This is a rudimentary class to set the necessary information for a scarlet run.

    While it is possible for scarlet to run without wcs or psf,
    it is strongly recommended not to, which is why these entry are not optional.
    """
    def __init__(self,images, wcs, psfs, channels):
        self._images = images
        self.wcs = wcs
        self.psfs = psfs
        self.channels = channels

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, images):
        self._images = images

def convert_coordinates(coord, origin, target):
    """Converts coordinates from one reference frame to another
    Parameters
    ----------
    coord: `tuple`
        pixel coordinates in the frame of the `origin`
    origin: `~scarlet.Frame`
        origin frame
    target: `~scarlet.Frame`
        target frame
    Returns
    -------
    coord_target: `tuple`
        coordinates at the location of `coord` in the target frame
    """
    pix = np.stack(coord, axis=1)
    ra_dec = origin.get_sky_coord(pix)
    yx = target.get_pixel(ra_dec)
    return yx[:, 0], yx[:, 1]

def interpolate(data_lr, data_hr):
    ''' Interpolate low resolution data to high resolution

    Parameters
    ----------
    data_lr: Data
        low resolution Data
    data_hr: Data
        high resolution Data

    Result
    ------
    interp: numpy array
        the images in data_lr interpolated to the grid of data_hr
    '''
    frame_lr = scarlet.Frame(data_lr.images.shape, wcs = data_lr.wcs, channels = data_lr.channels)
    frame_hr = scarlet.Frame(data_hr.images.shape, wcs = data_hr.wcs, channels = data_hr.channels)

    coord_lr0 = (np.arange(data_lr.images.shape[1]), np.arange(data_lr.images.shape[1]))
    coord_hr = (np.arange(data_hr.images.shape[1]), np.arange(data_hr.images.shape[1]))
    coord_lr = convert_coordinates(coord_lr0, frame_lr, frame_hr)

    interp = []
    for image in data_lr.images:
        interp.append(scarlet.interpolation.sinc_interp(image[None, :, :], coord_hr, coord_lr, angle=None)[0].T)
    return np.array(interp)


def makeCatalog(datas, lvl=3, thresh=3, wave=True):
    # Create a catalog of detected source by running SEP on the wavelet transform
    # of the sum of the high resolution images and the low resolution images interpolated to the high resolution grid
    # Interpolate LR to HR
    if len(datas)==2:
        data_lr, data_hr = datas
        interp = interpolate(data_lr, data_hr)
        # Normalisation
        interp = interp / np.sum(interp, axis=(1, 2))[:, None, None]
        hr_images = data_hr.images / np.sum(data_hr.images, axis=(1, 2))[:, None, None]
        # Summation to create a detection image
        detect_image = np.sum(interp, axis=0) + np.sum(hr_images, axis=0)

    elif len(datas) == 1:
        norm = datas[0].images / np.sum(datas[0].images, axis=(1, 2))[:, None, None]
        detect_image = np.sum(norm, axis = 0)
    else:
        "This is a mistake"
    # Rescaling to HR image flux
    # detect_image *= np.sum(data_hr.images)
    # Wavelet transform
    wave_detect = scarlet.Starlet.from_image(detect_image).coefficients

    if wave:
        # Creates detection from the first 3 wavelet levels
        detect = wave_detect[:lvl, :, :].sum(axis=0)
    else:
        detect = detect_image
    # Runs SEP detection
    bkg = sep.Background(detect)
    catalog = sep.extract(detect, thresh, err=bkg.globalrms)

    if len(datas) == 1:
        bg_rms = mad_wavelet(datas[0].images)
    else:
        bg_rms = []
        for data in datas:
            bg_rms.append(mad_wavelet(data.images))

    return catalog, np.array(bg_rms)
