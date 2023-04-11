from fastai.data.all import *
from librosa.feature import melspectrogram
from torch.nn.functional import pad
import librosa
from sklearn.preprocessing import normalize
from scipy.interpolate import interp1d
import pyworld as pw
import tgt

# Melspectrogram
def Load_audio(sample_rate):
    @Transform
    def _inner(path):
        waveform, _ = librosa.load(path, sr=sample_rate)
        return waveform
    return _inner

def Mel_transform(sample_rate, n_fft, hop_length):
    @Transform
    def _inner(y): 
        return melspectrogram(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    return _inner

db_transform = Transform(librosa.power_to_db)


# Energy
def Stft_transform(n_fft, hop_length):
    @Transform
    def _inner(y): 
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    return _inner

@Transform
def extract_amplitude(spectrogram):
    return librosa.magphase(spectrogram)[0]

@Transform
def l2_norm(inputs):
    return np.sqrt(np.sum(inputs ** 2, axis=0))

# Pitch
def Extract_pitch_contour(sample_rate, hop_length):
    @Transform
    def _inner(waveform):
        f0, t_positions = pw.dio(waveform.astype(np.float64), 
                         sample_rate, 
                         frame_period=hop_length/sample_rate*1000)
        pitch_contour = pw.stonemask(waveform.astype(np.float64), 
                                     f0, t_positions, sample_rate)
        return pitch_contour
    return _inner

@Transform
def interploate(pitch_contour):
    nonzero_indexs = np.where(pitch_contour != 0)[0]
    interp_fn = interp1d(nonzero_indexs, pitch_contour[nonzero_indexs], 
                     bounds_error=False, fill_value="extrapolate")
    return interp_fn(np.arange(0, len(pitch_contour)))

log_transform = Transform(np.log)

@Transform
def normalize_transform(inputs):
    return normalize(inputs.reshape(1,-1)).reshape(-1)

# Duration
def Change_extension(extension):
    @Transform
    def _inner(path):
        path = Path(path)
        return path.parent/(path.name.split(".")[0] + extension)
    return _inner

@Transform
def load_text_grid(tg_path):
    text_grid = tgt.io.read_textgrid(tg_path)
    return text_grid.get_tier_by_name("phones")

def Get_durations(sample_rate, hop_length):
    def _inner(text_grid):
        durations = []
        for interval in text_grid:
            start, end = interval.start_time, interval.end_time

            duration = int(np.round(end*sample_rate/hop_length) -
                           np.round(start*sample_rate/hop_length))

            durations.append(duration)
        return durations
    return _inner

@Transform
def get_phonemes(text_grid):
    return list(map(lambda tg: tg.text, text_grid))

# Batch
def Pad_mel(y, x, value=-58):
    @Transform
    def _inner(mel):
        ydim, xdim = mel.shape
        padx = x-xdim if x!=0 else 0
        pady = y-ydim if y!= 0 else 0

        return pad(torch.tensor(mel), (0, padx, 0, pady), value=value).numpy()
    return _inner

def Pad_sequence(size, mode='constant', value=0):
    @Transform
    def _inner(inputs):
        if size < len(inputs):
            return inputs
        return pad(torch.tensor(inputs), (0, size-len(inputs)), mode=mode, value=value).numpy()
    return _inner