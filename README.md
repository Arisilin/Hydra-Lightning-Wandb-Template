# Tiny-bVAE
A simple yet well disentangled implementation of VAE/βVAE. Template of using Hydra+wandb+Pytorch Lightning.

## NOTE
Still in development, so code might be break from time to time.

But feel free to use this as an develop template for your own project. It's great pleasure if you find this helpful.

Develop Progress:

- [x] Build Model Interface and Encoder-Decoder Modules, Losses.
- [x] Setup configuration using Hydra
- [x] Build Data Interface and corresponding dataset loader.
- [x] Build main training script
- [x] Add WandbLogger to log loss and result images.
- [x] Test and fix the bugs in model architecture and losses
- [ ] TODO: Visualization code use for demonstrate origin vs reconstruction results.

## Acknowledgements
The idea of file organization and some borrowed code originated from:

[miracleyoo/pytorch-lightning-template](https://github.com/miracleyoo/pytorch-lightning-template)

[ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

which are also great templates for users of pytorch lightning. Thank to their idea and codes.