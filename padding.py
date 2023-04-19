from fastai.data.all import *
from torch.nn.utils.rnn import pad_sequence
from utils import argmax_all

def pad_max_seq(seq, pad_val=0, pad_len=1):
    '''
    seq: List[tensors]
    pad_val: int
    pad_len: int
    '''
    seq_pad = L(seq[:])
    max_idxs = argmax_all(tensor(seq_pad.map(len)))
    for idx in max_idxs:
        seq_pad[idx] = torch.cat((seq_pad[idx], torch.tensor([pad_val]*pad_len)))
    return seq_pad

def pad_mels(mels):
    '''mels: List[tensor]'''
    return pad_sequence(L(mels).map(lambda x: x.T), batch_first=True).transpose(1, 2)

def pad_phones(nums, pad_num): 
    '''
    nums: List[tensor]
    pad_num: int
    '''
    return pad_sequence(nums, batch_first=True, padding_value=pad_num)

def pad_durations(durations, mel_len):
    '''
    durations: List[tensors]
    mel_len: int
    '''
    padded_duration = pad_sequence(durations, batch_first=True)
    padded_duration_amount = [mel_len - dur.sum().item() for dur in durations]
    padded_duration[:, -1] = tensor(padded_duration_amount)
    return padded_duration
