import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class FastSpeech(nn.Module):
    def __init__(self, vocab_sz, nhidden, nout, nheads, nblocks):
        super(FastSpeech, self).__init__()
        self.embedding = nn.Embedding(vocab_sz, nhidden)
        self.fft_pho = nn.ModuleList([FeedForwardTransformer(nhidden, nheads)
                                      for _ in range(nblocks)])
        self.fft_mel = nn.ModuleList([FeedForwardTransformer(nhidden, nheads)
                                      for _ in range(nblocks)])
        self.linear = nn.Linear(nhidden, nout)
    
    def forward(self, inp, durations, upsample_ratio):
        x = self.embedding(inp)
        x = x + positional_embeddings(*x.shape[-2:])
        for layer in self.fft_pho:
            x = layer(x)
        x = length_regulator(x, durations, upsample_ratio)
        x = x + positional_embeddings(*x.shape[-2:])
        for layer in self.fft_mel:
            x = layer(x)
        x = self.linear(x)
        return x.transpose(1,2)
    
class FeedForwardTransformer(nn.Module):
    def __init__(self, ni, nheads):
        super(FeedForwardTransformer, self).__init__()
        self.attention = SelfAttention(ni, nheads)
        self.resconv = ResConv(ni)
    def forward(self, inp):
        x = self.attention(inp)
        x = self.resconv(x)
        return x  

class SelfAttention(nn.Module):
    def __init__(self, ni, nheads):
        super(SelfAttention, self).__init__()
        self.heads = nheads
        self.scale = math.sqrt(ni / nheads)
        self.kqv = nn.Linear(ni, ni*3)
        self.proj = nn.Linear(ni, ni)
        self.norm = nn.LayerNorm(ni)
    def forward(self, inp):
        x = self.kqv(inp)
        x = torch.cat(torch.chunk(x, self.heads, dim=-1))
        Q, K, V = torch.chunk(x, 3, dim=-1)
        x = F.softmax(Q @ K.transpose(1,2) / self.scale, dim=-1) @ V
        x = torch.cat(torch.chunk(x, self.heads), dim=-1)
        x = self.proj(x)
        x = x + inp
        return self.norm(x)
    
class ResConv(nn.Module):
    def __init__(self, ni):
        super(ResConv, self).__init__()
        self.conv = nn.Conv1d(ni, ni, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(ni)     
    def forward(self, inp):
        x = self.conv(inp.transpose(1,2))
        x = x.transpose(1,2) + inp
        return self.norm(x)
    
def positional_embeddings(seq_len, d_model):
    pos, i = torch.arange(d_model)[None, :], torch.arange(seq_len)[:, None]
    angle = pos / torch.pow(10000, 2 * i / d_model)
    pos_emb = torch.zeros(angle.shape)
    pos_emb[0::2,:], pos_emb[1::2,:] = angle[0::2,:].sin(), angle[1::2,:].cos()
    return pos_emb
    
def length_regulator(inp, durations, upsample_ratio, dim=1):
    adjusted_durations = (upsample_ratio * durations).to(torch.int)
    tmp = torch.zeros((inp.size(0), durations[0].sum().item(), inp.size(2)))
    for i in range(inp.size(0)):
        tmp[i] =  inp[i].repeat_interleave(adjusted_durations[i], dim=0)
    return tmp