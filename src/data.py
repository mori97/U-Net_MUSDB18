"""A module for loading data which contains a function for data preporcessing
and a Dataset class for PyTorch.
"""
import os
import random

import torch.utils.data
import torchaudio


def read_data(dataset_dir, n_fft, data_len, sampling_rate):
    """Load and preprocess MUSDB18 dataset.
    The Songs in train dataset are split to fixed length segments.

    Args:
        dataset_dir (str): Path of MUSDB18 which was converted to .wav format.
        n_fft (int):  Size of Fourier Transform.
        data_len (int): Number of time frames of a training data.
        sampling_rate (int): Sampling rate.

    Returns:
        Tuple[List[Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]]:
            A tuple contains train data and test data.
            Train data is a list of tuple which contains magnitude spectrogram
            of input signal and ground truth separation mask. Test data is a
            list of tensors which contain mixture signal and separated signal.
    """
    window = torch.hann_window(n_fft)

    with torch.no_grad():
        # Train data
        train_dir = os.path.join(dataset_dir, 'train')
        wav_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)
                     if os.path.splitext(f)[1] == '.wav']
        train = []
        for path in wav_files:
            sound, sr = torchaudio.load(path)
            assert sr == sampling_rate
            sound_spec = torch.stft(sound, n_fft, window=window)
            sound_spec = sound_spec.pow(2).sum(-1).sqrt()
            x = sound_spec[0]
            t = sound_spec[1:]

            hop = data_len // 4
            # Split to fixed length segments
            for n in range((x.size(1) - data_len) // hop + 1):
                start = n * hop
                train.append((x[:, start:start + data_len],
                              t[:, :, start:start + data_len]))

        # Test data
        test_dir = os.path.join(dataset_dir, 'test')
        wav_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                     if os.path.splitext(f)[1] == '.wav']
        test = []
        for path in wav_files:
            sound, sr = torchaudio.load(path)
            assert sr == sampling_rate
            # Split data into two parts to save memory
            split_sound = torch.split(sound, sound.size(1) // 2, dim=1)
            test.extend(split_sound)

    return train, test


class RandomCropDataset(torch.utils.data.Dataset):
    """Dataset class for training of music separation U-Net.
    Spectrograms are randomly croped into size `out_len`.

    Args:
        data (List[Tuple[torch.Tensor, torch.Tensor]]:
            Pairs of mixture and separated sound.
        out_len (int): Number of time frames of output data.
    """
    def __init__(self, data, out_len):
        self.data = data
        self.len = out_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x, t = self.data[item]
        length = x.size(1)
        i = random.randint(0, length - self.len)
        return x[:, i:i + self.len], t[:, :, i:i + self.len]
