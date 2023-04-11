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