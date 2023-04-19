import matplotlib.pyplot as plt
import librosa

def show_mels(mels):
    num_mels = len(mels)
    fig, axes = plt.subplots(nrows=num_mels, ncols=1, figsize=(10, 2*num_mels))
    for i, mel in enumerate(mels):
        im = axes[i].imshow(librosa.power_to_db(mel), origin='lower')
        axes[i].set(title=f'Melspectrogram {i}')
    plt.tight_layout()
    plt.show(True)
    
def argmax_all(tens):
    max_val = 0
    max_idxs = []
    for i in range(len(tens)):
        val = tens[i].item()
        if val > max_val:
            max_val = val
            max_idxs = [i]
        elif val == max_val:
            max_idxs.append(i)
    return max_idxs
    
def show_mel(mels): plt.imshow(librosa.power_to_db(mels), origin='lower')

def clip_mel(mel, duration): return mel[:,:sum(duration)]