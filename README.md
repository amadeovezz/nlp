# What

This repository contains different approaches to autoregressive character modeling using neural nets. 
The purpose is to implement, compare and contrast some of big ideas in the field.

Some of these ideas include:

- learned embeddings (Bengio et al.)
- vanilla rnn's
- attention based models

## Intention

This intention of this repository is educational in nature, and as such readability and conceptual understanding of the code is prioritized over performance.

## Inspiration

- Many of the ideas and code is from https://github.com/karpathy/nn-zero-to-hero.

## TODO

- MLP's
  - [ ] notes
  - [ ] hyper parameter tuning
- RNN's
  - [ ] generation
  - [ ] fix loss - previous_pre_activation shapes are stored as state and interfere with validation computations
  - [ ] double check back-prop through time
  - [ ] notes
  - [ ] hyper parameter tuning
- Attention
  - [ ] loss
  - [ ] generation
  - [ ] notes
  - [ ] decoder
  - [ ] hyper parameter tuning


Potential TODOs:

- [ ] BatchNorm/LayerNorm layers?
- [ ] Residual connections