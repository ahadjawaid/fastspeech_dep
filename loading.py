from fastai.data.all import *
import tgt

get_audio_files = FileGetter(extensions='.wav')

def get_tiers(path): return tgt.io.read_textgrid(path).get_tier_by_name("phones")

def get_phoneme_duration(tiers, sr, hop_length):
    '''
    tiers: List[tgt.core.IntervalTier]
    sr: int
    hop_length: int
    '''
    phones, durations = [], []
    for interval in tiers:
        start, end, phoneme = interval.start_time, interval.end_time, interval.text
        duration = int(end*sr/hop_length - start*sr/hop_length)
        phones.append(phoneme)
        durations.append(duration)
    return phones, durations
