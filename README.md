# Tiny-bVAE: A Template for Simplified Research Code Development 
A simple yet well decoupled implementation of VAE. Template of using Hydra+wandb+Pytorch Lightning.

## NOTE
Still under development, so code might be break from time to time.

But feel free to use this as an develop template for your own project. It's great pleasure if you find this helpful.

Develop Progress:

- [x] Build Model Interface and Encoder-Decoder Modules, Losses.
- [x] Setup configuration using Hydra
- [x] Build Data Interface and corresponding dataset loader.
- [x] Build main training script
- [x] Add WandbLogger to log loss and result images.
- [x] Test and fix the bugs in model architecture and losses
- [x] Visualization code use for demonstrate origin vs reconstruction results.
- [x] Reorganize the logger and lightning checkpointing settings, added multirun support.

## Usage

To run your own training experiment
```
conda env create -f conda-env.yaml
```
then modify ./config/loggers/meta.yaml to set your wandb settings, and set symlink ./data to your dataset(for now, CelebA(just image file folder) or CelebA-HQ(version with folder organized in val/train and female/male classes) are supported).

Set symlink ./output to your desired output folder.

Then just run
```
python main.py
```


## Acknowledgements
The idea of file organization and some borrowed code originated from:

[miracleyoo/pytorch-lightning-template](https://github.com/miracleyoo/pytorch-lightning-template)

[ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

which are also great templates for users of pytorch lightning. Thank to their idea and codes.