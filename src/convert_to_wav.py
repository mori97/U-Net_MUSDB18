"""Convert MUSDB18 dataset to .wav format.
Output .wav files contain 5 channels
- `0` - The mixture,
- `1` - The drums,
- `2` - The bass,
- `3` - The rest of the accompaniment,
- `4` - The vocals.
"""
import argparse
import os
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('origin_dataset_dir',
                        help='Path of the original dataset (.mp4)',
                        type=str)
    parser.add_argument('new_dataset_dir',
                        help='Output path of .wav dataset',
                        type=str)
    parser.add_argument('--sr',
                        help='Sample rate. (Default: 22050) ',
                        type=int, default=22050)
    args = parser.parse_args()

    origin_dataset_dir = args.origin_dataset_dir
    new_dataset_dir = args.new_dataset_dir

    if os.path.isdir(new_dataset_dir):
        raise FileExistsError(f'{new_dataset_dir} already exists.')
    else:
        os.mkdir(new_dataset_dir)

    os.mkdir(os.path.join(new_dataset_dir, 'train'))
    os.mkdir(os.path.join(new_dataset_dir, 'test'))

    with tempfile.TemporaryDirectory() as tmpdir:
        for subdir in ('train', 'test'):
            origin_dir = os.path.join(origin_dataset_dir, subdir)
            files = [f for f in os.listdir(origin_dir)
                     if os.path.splitext(f)[1] == '.mp4']
            for file in files:
                path = os.path.join(origin_dir, file)
                name = os.path.splitext(file)[0]
                wav_data = []
                # Extract & save the sound of `ch` channel to a temp directory
                # and then concatenate all channels to a single .wav file
                for ch in range(5):
                    temp_fn = f'{name}.{ch}.wav'
                    out_path = os.path.join(tmpdir, temp_fn)
                    subprocess.run(['ffmpeg', '-i', path,
                                    '-map', f'0:{ch}', out_path])
                    sound, _ = librosa.load(out_path, sr=args.sr, mono=True)
                    wav_data.append(sound)
                wav_data = np.stack(wav_data, axis=1)
                out_path = os.path.join(
                    new_dataset_dir, subdir, f'{name}.wav')
                sf.write(out_path, wav_data, args.sr)


if __name__ == '__main__':
    main()
