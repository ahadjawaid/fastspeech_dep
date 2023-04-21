import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class FastSpeech(nn.Module):
    def __init__(self, vocab_sz, nhidden, nout, nheads, kernal_sz, nfilters, nblocks, p,
                 kernal_sz_v, nfilters_v, p_v, device=None):
        super(FastSpeech, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_sz, nhidden)
        self.fft_pho = nn.ModuleList([FeedForwardTransformer(nhidden, nheads, kernal_sz, nfilters, p)
                                      for _ in range(nblocks)])
        
        self.duration_predictor = VariencePredictor(nhidden, kernal_sz_v, nfilters_v, p_v)

        self.fft_mel = nn.ModuleList([FeedForwardTransformer(nhidden, nheads, kernal_sz, nfilters, p)
                                      for _ in range(nblocks)])
        self.linear = nn.Linear(nhidden, nout)
    def forward(self, inp, durations, upsample_ratio, dur_train=False):
        x = self.embedding(inp)

        x = x + positional_embeddings(*x.shape[-2:], device=self.device)

        for layer in self.fft_pho:
            x = layer(x)

        log_durations_pred = self.duration_predictor(x.detach()).squeeze(-1)
        if not self.training:
            durations = torch.exp(log_durations_pred)

        x = length_regulator(x, durations, upsample_ratio, device=self.device)

        x = x + positional_embeddings(*x.shape[-2:], device=self.device)

        for layer in self.fft_mel:
            x = layer(x)
            
        x = self.linear(x).transpose(1,2)
        return (x, log_durations_pred) if dur_train else x
    
class VariencePredictor(nn.Module):
    def __init__(self, ni, ks, nf, p):
        super(VariencePredictor, self).__init__()
        padding = (ks -1) // 2
        self.conv1, self.norm1 = nn.Conv1d(ni, nf, ks, padding=padding), nn.LayerNorm(nf)
        self.conv2, self.norm2 = nn.Conv1d(nf, ni, ks, padding=padding), nn.LayerNorm(ni)
        self.dropout = nn.Dropout(p)
        self.linear = nn.Linear(ni, 1)
    def forward(self, hi):
        x = F.relu(self.conv1(hi.transpose(1,2)))
        x = self.norm1((self.dropout(x)).transpose(1,2))
        x = F.relu(self.conv2(x.transpose(1,2)))
        x = self.norm2((self.dropout(x)).transpose(1,2))
        return self.linear(x)

class FeedForwardTransformer(nn.Module):
    def __init__(self, ni, nheads, ks, nf, p):
        super(FeedForwardTransformer, self).__init__()
        self.attention = SelfAttention(ni, nheads, p)
        self.resconv = ResConv(ni, ks, nf, p)
    def forward(self, inp):
        x = self.attention(inp)
        x = self.resconv(x)
        return x  

class SelfAttention(nn.Module):
    def __init__(self, ni, nheads, p):
        super(SelfAttention, self).__init__()
        self.heads = nheads
        self.scale = math.sqrt(ni / nheads)
        self.kqv = nn.Linear(ni, ni*3)
        self.proj = nn.Linear(ni, ni)
        self.norm = nn.LayerNorm(ni)
        self.dropout = nn.Dropout(p)
    def forward(self, inp):
        x = self.kqv(inp)
        x = torch.cat(torch.chunk(x, self.heads, dim=-1))
        Q, K, V = torch.chunk(x, 3, dim=-1)
        x = F.softmax((Q @ K.transpose(1,2)) / self.scale, dim=-1) @ V
        x = torch.cat(torch.chunk(x, self.heads), dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        x = x + inp
        return self.norm(x)
    
class ResConv(nn.Module):
    def __init__(self, ni, ks, nf, p):
        super(ResConv, self).__init__()
        padding = (ks -1) // 2
        self.conv1 = nn.Conv1d(ni, nf, kernel_size=ks, padding=padding)
        self.conv2 = nn.Conv1d(nf, ni, kernel_size=ks, padding=padding)
        self.norm = nn.LayerNorm(ni)
        self.dropout = nn.Dropout(p)
    def forward(self, inp):
        x = F.relu(self.conv1(inp.transpose(1,2)))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.transpose(1,2) + inp
        return self.norm(x)
    
def positional_embeddings(seq_len, d_model, device=None):
    pos = torch.arange(d_model, device=device)[None, :]
    i = torch.arange(seq_len, device=device)[:, None]
    angle = pos / torch.pow(10000, 2 * i / d_model)
    pos_emb = torch.zeros(angle.shape, device=device)
    pos_emb[0::2,:], pos_emb[1::2,:] = angle[0::2,:].sin(), angle[1::2,:].cos()
    return pos_emb
    
def length_regulator(inp, durations, upsample_ratio, dim=1, device=None):
    adjusted_durations = (upsample_ratio * durations).to(torch.int)
    tmp = torch.zeros((inp.size(0), durations[0].sum().item(), inp.size(2)), device=device)
    for i in range(inp.size(0)):
        tmp[i] =  inp[i].repeat_interleave(adjusted_durations[i], dim=0)
    return tmp