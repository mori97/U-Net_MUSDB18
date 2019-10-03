import torch.nn.functional as F


def zero_padding(sound_stft):
    """Apply zero padding to ensure that number of time frames of `sound`'s
    STFT representation is multiple of 64.

    Args:
        sound_stft (torch.Tensor): Spectrogram to be padded.

    Returns:
        Tuple[torch.Tensor, Tuple[int, int]]: Zero padded spectrogram and
            number of rows padded to left-side and right-side, respectively.
    """
    n_frames = sound_stft.size(-1)
    n_pad = (64 - n_frames % 64) % 64
    if n_pad:
        left = n_pad // 2
        right = n_pad - left
        return F.pad(sound_stft, (left, right)), (left, right)
    else:
        return sound_stft, (0, 0)
