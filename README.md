# blowVC Bill Gates

## Overall

Model based on [article](https://arxiv.org/abs/1906.00794).
This repo aims to convert voice to the Bill Gates one.

Installing requirement:

```bash
pip install -r requirements.txt
```

Log to your wandb account before training:

```bash
wandb login
```

## Data

Data was gathered from voxceleb v1 dataset.
Samples for Bill Gates was gathered from `6Af6b_wyiwI`, `ofQMbC2e_as`, `JaF-fq2Zn7I` and `4X-KkQeMMSQ` Youtube videos. (see this [notebook](https://colab.research.google.com/drive/1MirTXE5puBM6zblXuhQf13d0Am-rsF6u?usp=sharing))

For simplicity, I took a few samples of the voices of famous personalities of a nationality and gender similar to Bill Gates.
There are 6 people (in addition to Bill Gates) for 15 minutes each, the article above says that there is no point in taking more.
The most homogeneous samples of the voices were selected, because the data from different years and of different quality were used in the dataset.
All samples were equalized by sound volume, background noises were removed, as well as resampled up to 16kHz using the `librosa` library.

Also, for quick loading, I pre-processed (+normalization) all the samples in the torch tensors.
You can download this ready-to-train data using this command (`gsutil` required, bucket is open).

```bash
gsutil -m cp -r gs://efficient-vot/voxceleb/pt/* data > /dev/null 2>&1
```

You can also find the source `wav` file along the path `gs://efficient-vot/voxceleb/wav/*`

## Train

To train, run:

```bash
python -m train --model_fname=blowmodel
```

All hyperparameters are configured more or less optimally.
It takes a little over a day on 4 V100 gpus.

## Pre-trained model

```bash
gsutil cp gs://efficient-vot/blowmodel.pt weights > /dev/null 2>&1
```

## Synthesize

To synthesize samples to Bill Gates voice (non-seen on training) with a best model, run:

```bash
python -m synth --model_fname=blowmodel
```

And then check `synth` folder.
You can also use this [colab notebook](https://colab.research.google.com/drive/1YUs6PxCIyf_47Vx04fQWwRiNW7tYEVY5?usp=sharing) for quick reproduction.

