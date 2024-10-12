from src.transforms.wav_augs.gain import Gain
from src.transforms.wav_augs.band_pass import BandPass
from src.transforms.wav_augs.band_stop import BandStop
from src.transforms.wav_augs.colored_noise import ColoredNoise
from src.transforms.wav_augs.peak_normalization import PeakNormalize
from src.transforms.wav_augs.shifting import Shifting


all = [
    "BandPass",
    "BandStop",
    "ColoredNoise",
    "Gain",
    "PeakNormalize"
    "Shifting"
]