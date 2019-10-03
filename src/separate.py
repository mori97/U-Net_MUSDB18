"""A script for evaluation.
"""
import argparse

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import istft

from train import N_FFT, N_PART, SAMPLING_RATE
from u_net import UNet


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        help='Input mixture .wav file\nIf input contains '
                             'more than one channel, ch0 will be used',
                        type=str)
    parser.add_argument('output',
                        help='Output path of separated .wav file',
                        type=str)
    parser.add_argument('--model', '-m',
                        help='Trained model',
                        type=str, metavar='PATH', required=True)
    parser.add_argument('--gpu', '-g',
                        help='GPU id (Negative number indicates CPU)',
                        type=int, metavar='ID', default=-1)
    args = parser.parse_args()

    if_use_cuda = torch.cuda.is_available() and args.gpu >= 0
    device = torch.device(f'cuda:{args.gpu}' if if_use_cuda else 'cpu')

    with torch.no_grad():
        sound, _ = torchaudio.load(args.input)
        sound = sound[[0], :].to(device)

        window = torch.hann_window(N_FFT, device=device)

        # Convert it to power spectrogram, and pad it to make the number of
        # time frames to a multiple of 64 to be fed into U-NET
        sound_stft = torch.stft(sound, N_FFT, window=window)
        sound_spec = sound_stft.pow(2).sum(-1).sqrt()
        sound_spec, (left, right) = zero_padding(sound_spec)

        # Load the model
        model = UNet(N_PART)
        model.load_state_dict(torch.load(args.model))
        model.to(device)
        model.eval()

        mask = model(sound_spec).squeeze(0)
        mask = mask[:, :, left:-right]
        separated = mask.unsqueeze(3) * sound_stft
        separated = istft(
            separated, N_FFT, window=window, length=sound.size(-1))
        separated = separated.cpu().numpy()

    # Save the separated signals
    sf.write(args.output, separated.T, SAMPLING_RATE)


if __name__ == '__main__':
    main()
