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

- MLP
  - Make train work
- Emb MLP's
  - [ ] notes
  - [ ] hp tuning
- RNN's
  - [ ] Autogressive batch
  - [ ] double check back-prop through time
  - [ ] notes
  - [ ] hp tuning
- Attention
  - [ ] loss
  - [ ] generation
  - [ ] notes
  - [ ] decoder
  - [ ] hp tuning


Potential TODOs:

- [ ] BatchNorm/LayerNorm layers?
- [ ] Residual connections