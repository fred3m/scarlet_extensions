import numpy as np
import scarlet
from scarlet import initialization as init
from scarlet.renderer import NullRenderer, ConvolutionRenderer

class ParametricInit(scarlet.FactorizedComponent):
    def __init__(self,
                 model_frame,
                 sky_coord,
                 observations,
                 profile,
                 thresh=1.0,
                 shifting=False):

        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        detect = np.zeros(model_frame.shape, dtype=model_frame.dtype)
        for i, obs in enumerate(observations):

            if isinstance(obs.renderer, ConvolutionRenderer):

                data_slice, model_slice = obs.renderer.slices
                obs.renderer.map_channels(detect)[model_slice] = \
                    (np.ones(obs.shape[0])[:,None,None]*profile[None, :,:])[data_slice]

        # initialize morphology
        # make monotonic morphology, trimmed to box with pixels above std
        morph, bbox = self.init_morph(model_frame,
            sky_coord,
            detect[-1],
        )

        center = model_frame.get_pixel(sky_coord)
        morphology = scarlet.ExtendedSourceMorphology(
            model_frame,
            center,
            morph,
            bbox=bbox,
            monotonic="angle",
            symmetric=False,
            min_grad=0,
            shifting=shifting,
        )

        # find best-fit spectra for morph from init coadd
        # assumes img only has that source in region of the box
        detect_all, std_all = init.build_initialization_image(observations)
        box_3D = scarlet.Box((model_frame.C,)) @ bbox
        boxed_detect = box_3D.extract_from(detect_all)
        spectrum = init.get_best_fit_spectrum((morph,), boxed_detect)
        noise_rms = np.concatenate(
            [np.array(np.mean(obs.noise_rms, axis=(1, 2))) for obs in observations]
        ).reshape(-1)
        spectrum = scarlet.TabulatedSpectrum(model_frame, spectrum, min_step=noise_rms)

        # set up model with its parameters
        super().__init__(model_frame, spectrum, morphology)

        # retain center as attribute
        self.center = morphology.center

    @staticmethod
    def init_morph(frame,
                   sky_coord,
                   profile,):


        # position in frame coordinates
        center = frame.get_pixel(sky_coord)
        center_index = np.round(center).astype("int")

        # truncate morph at thresh * bg_rms
        threshold = np.max(profile)*1.e-5
        morph, bbox = init.trim_morphology(center_index, profile.T, bg_thresh=threshold)
        # normalize to unity at peak pixel for the imposed normalization
        morph /= morph.max()

        return morph, bbox