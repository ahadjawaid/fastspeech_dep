from torchaudio.datasets import LJSPEECH
from torchaudio.transforms import MelSpectrogram
from torch.nn.functional import pad
from g2p_en import G2p
import re

class LJSpeechTTS(LJSPEECH):
    def __init__(self, root_dir, download=False, sample_rate=22050, 
                 n_fft=1024, hop_length=256):
        super(LJSpeechTTS, self).__init__(root_dir, download=False)
        self.n_fft, self.hop_length, self.g2p = n_fft, hop_length, G2p()
        self.phoneme_vocab = Vocab(self.g2p.phonemes+[' '])
        self.mel_transform = MelSpectrogram(sample_rate, self.n_fft, 
                                            hop_length=self.hop_length)
        
        
    def __getitem__(self, i):
        waveform, sample_rate, _, norm_transcript = super().__getitem__(i)
        
        phonemes = self.g2p(norm_transcript)
        filtered_phonemes = list(filter(lambda x: re.search('[A-Z]',x), phonemes))
        encoded_phonemes = list(map(lambda phoneme: self.phoneme_vocab[phoneme], filtered_phonemes))
        
        mel_spectrogram = self.mel_transform(waveform)
        
        return (waveform, sample_rate, mel_spectrogram, norm_transcript, encoded_phonemes)