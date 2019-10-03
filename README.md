# U-Net_MUSDB18

## Prerequisites

- librosa >= 0.7.0
- numpy >= 1.16.0
- pysoundfile >= 0.10.0
- pytorch >= 1.2.0
- tensorboard >= 1.14.0
- torchaudio >= 0.3.0

and `ffmpeg` should be installed to do the preprocessing.

## Usage

The original MUSDB18 dataset consists of .mp4 files.
First, convert the whole dataset into .wav format for convenience.

```bash
$ python convert_to_wav.py --sr 22050 {your_path}/musdb18 {your_path}/musdb18_wav_22050
```

Then train the model by `train.py`. For example,

```bash
$ python train.py --dataset {your_path}/musdb18_wav_22050 \
                  --gpu 0 1 2 3 \
                  --batch-size 256 \
                  --output ./model
```

Help message of commandline arguments can be found by

```bash
$ python train.py -h
```

You can use `separate.py` to test your model.
