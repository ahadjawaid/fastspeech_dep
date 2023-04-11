from fastai.data.all import *
from loading import *
from padding import *
import librosa

class TTSDataset(torch.utils.data.Dataset):
    def __init__(self, path_data, path_vocab, sample_rate=22050, 
                 n_fft=2048, hop_length=512, n_bins=80, ratio=1):
        self.files = get_audio_files(path_data)
        self.files = self.files[:int(ratio * len(self.files))]
        self.files_tg = self.files.map(lambda x: x.with_suffix('.TextGrid'))
        
        self.vocab = L(path_vocab.open().readlines()).map(lambda x: x.strip())
        self.vocab = self.vocab + L(["<pad>", "spn"])
        self.pho2idx = {phoneme: i for i, phoneme in enumerate(self.vocab)}
        self.pad_num = self.pho2idx["<pad>"]
        
        
        self.load_audio = partial(librosa.load, sr=sample_rate)
        self.melspectrogram = partial(librosa.feature.melspectrogram, n_fft=n_fft, 
                                      hop_length=hop_length, n_mels=n_bins)
        self.get_phoneme_duration = partial(get_phoneme_duration, sr=sample_rate, 
                                            hop_length=hop_length)
        
    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx):
        wav = self.load_audio(self.files[idx])[0]
        tiers = get_tiers(self.files_tg[idx])
        phones, durations = self.get_phoneme_duration(tiers)
        
        mel = tensor(self.melspectrogram(y=wav))
        nums = tensor(L(phones).map(self.pho2idx))
        durations = tensor(durations)
        
        return mel, nums, durations
    
def collate_fn(inp, pad_num):
    mels, phones, durations = zip(*inp)

    mels_batched = pad_mels(mels)
    mel_len = mels_batched.size(-1)
    
    phones_batched = pad_phones(pad_max_seq(phones), pad_num)
    durations_batched = pad_durations(pad_max_seq(durations), mel_len)
    
    assert phones_batched.shape == durations_batched.shape
    assert (durations_batched.sum(dim=-1) == mel_len).all()
    
    return mels_batched, phones_batched, durations_batched