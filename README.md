# What

This repository contains different approaches to autoregressive character modeling using neural nets. 
The purpose is to implement, compare and contrast some of notable ideas in the field.

## Inspiration and Intention

This repo is essentially a re-implementation of many of the core ideas in https://github.com/karpathy/nn-zero-to-hero.
There is a lot of code re-use from these lectures.

Additionally, this repository is purely educational in nature, and as such readability is prioritized over 
performance.

### A note on pytorch 

There is intentionally no use of `nn.Module`.

## Current models 

- learned embedding + mlps (Bengio et al.)
- vanilla rnns
- attention based models

## Directory structure

Each model lives in its own directory. Within it contains:

- a notebook `*_nb.ipynb` - where the model is run
- `notes.md` some notes about the model

It is not a bad idea to start at `base/notes.md` and `mlp_nb.ipynb` to get a feel for the code. 