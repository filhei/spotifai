import warnings
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample
import torchaudio.functional as F

DEFAULT_SAMPLE_RATE = 44100
WINDOW_SIZE = 25 / 1000  # 25ms
WINDOW_STRIDE = 10 / 1000  # 10ms


class AudioProcessor:
    """
    Defines an audio transform that reads and preprocesses an audio file and produces features suitable for data analysis.

    The data processing steps are:
    Input: path (str): audio file path
    1. Reads an audio file
    2. Converts to mono format
    3. Resamples the waveform (to 16.1 kHz as default)
    4. Transforms the waveform into a logarithmic mel-spectrogram
    5. The data is then clamped and normalized to an approximate range of [-1, 1], based on parametric choices from a small subset of audio files
    6. Optionally, split the data into multiple chunks.
    Output: features (torch.tensor): preprocessed audio features with shape (mel, time) if n_time_bins is None, else (patch, mel, time).


    Args:
        n_mels (int): Number of mel filterbanks (the number of resulting frequency bins). If too high, multiple bins will be all-zero. Default 80.
        n_time_bins (int): Number of time bins in each patch. If None, the data will not be split into chunks and the resulting size of the time dimension will depend on the track length. With default settings, 1024 bins corresponds to 10.255 seconds (approximately 100 bins/s when window stride is 10 ms).
        resample_rate (int): Target audio sampling rate. CDs usually have 44100 Hz. Default 16100 Hz.
        window_size (float): Fourier transform window size. The duration of audio that is processed at a time when determining the frequency content of the file. Default 0.025 (25 ms).
        window_stride (float): Fourier transform window stride. The amount the processing window is moved when to analyze the next part of the audio. Default 0.01 (10 ms).
        center (float): Expected feature mean. Used for normalization. Default 11.1717 dB.
        threshold (float): Expected threshold for clipping weak signals. Default -34.2321 dB.
        dynamic_range (float): Expected difference in decibel between strongest and weakest signals. Use the default value 80 unless analysis indicates otherwise.
        max_time_padding (float between 0 and 1): The amount of padding that is acceptable for the last patch. If n_time_bins is 1000, and the provided audio results in 2600 bins, it will be split into 3 patches of 1000 bins. The final patch will have to be padded with 400 bins. If 'max_time_padding' is less than 0.4 in this case, the final patch will instead be discarded, and the output will be 2 patches of 1000 bins.


    Examples:
        audio_path = 'path/to/audio.mp3'

        # for features with shape (patch, mel, time)
        processor = AudioProcessor(n_time_bins=1024)
        features = processor(audio_path)

        # for features with shape (mel, time)
        processor = AudioProcessor(n_time_bins=None)
        features = processor(audio_path)

    """

    def __init__(
        self,
        n_mels=80,
        n_time_bins=None,
        resample_rate=16100,  #  Hz
        window_size=WINDOW_SIZE,  # s
        window_stride=WINDOW_STRIDE,  # s
        center=11.1717,  # dB
        threshold=-34.2321,  # dB
        dynamic_range=80,  # dB
        max_time_padding=0.25,
    ):

        # audio resampling
        self.resample_rate = resample_rate
        self.resampler = Resample(
            DEFAULT_SAMPLE_RATE,
            resample_rate,
            resampling_method="sinc_interp_hann",
        )

        # audio waveform -> frequency domain
        n_fft = int(window_size * resample_rate)
        hop_length = int(window_stride * resample_rate)

        self.transform = MelSpectrogram(
            self.resample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            power=2,
        )

        # normalization parameters
        self.center = center
        self.threshold = threshold
        self.dynamic_range = dynamic_range

        # split and pad
        self.patch_size = n_time_bins

        if self.patch_size is not None:
            self.min_patch_size = int(self.patch_size * (1 - max_time_padding))
            self.patch_duration = n_time_bins * window_stride + (
                window_size - window_stride
            )
        else:
            self.min_patch_size = None
            self.patch_duration = None

    def __call__(self, path):
        try:
            waveform, sample_rate = torchaudio.load(path)
        except Exception as e:
            return None

        mono_waveform = waveform.mean(dim=0)

        resampled_waveform = self.resample(mono_waveform, sample_rate)

        mel = self.transform(resampled_waveform)  # (mel, time)
        log_mel = self.power_to_db(mel)

        # remove low energy content
        features = log_mel.clamp(min=self.threshold)

        # normalize to approximately [-1, 1]
        features -= self.center
        features /= self.dynamic_range / 2

        # split into patches (patch, mel, time)
        if self.patch_size is not None:
            patches = self.split_into_patches(features)
            return patches

        return features

    def resample(self, waveform, sample_rate):

        if sample_rate == self.resample_rate:
            return waveform

        # transforms.Resample precomputes the resampling kernel and is faster than F.resample for the initialized set of parameters
        if sample_rate == DEFAULT_SAMPLE_RATE:
            resampled_waveform = self.resampler(waveform)
            return resampled_waveform

        resampled_waveform = F.resample(waveform, sample_rate, self.resample_rate)
        return resampled_waveform

    def power_to_db(self, x, ref=1.0, min_value=1e-10):
        x = x.clamp(min=min_value)
        db = 10 * torch.log10(x / ref)
        return db

    def split_into_patches(self, features):
        patches = torch.split(features, self.patch_size, dim=1)
        last_patch_size = patches[-1].shape[1]

        # trim if not enough data
        if last_patch_size < self.min_patch_size:
            if len(patches) <= 1:
                return None

            patches = torch.stack(patches[:-1], dim=0)
            return patches

        # pad last patch
        minimum_value = (self.threshold - self.center) / (self.dynamic_range / 2)
        padded_patch = torch.ones((features.shape[0], self.patch_size)) * minimum_value
        padded_patch[:, :last_patch_size] = patches[-1]
        patches = torch.stack(patches[:-1] + (padded_patch,), dim=0)

        return patches
