# What

This repository contains different approaches to autoregressive character modeling using neural nets. 
The purpose is to implement, compare and contrast some of notable ideas in the field.

## Intention

This intention of this repository is educational in nature, and as such readability and conceptual understanding of the code is prioritized over performance.

### A note on pytorch
`pytorch` is mainly just used for tensor operations and auto-grad features
whereas there is intentionally no use of `nn.Module`.

## Inspiration

This repo is inspired by https://github.com/karpathy/nn-zero-to-hero and some code is re-used from here. 


## Current models 

- mlps
- learned embedding + mlps (Bengio et al.)
- vanilla rnns
- attention based models

## Directory structure

Each model lives in its own directory. Within it contains:

- a notebook `*_nb.ipynb` - where the model is run
- `notes.md` some notes about the model

It is not a bad idea to start at `base/notes.md` and `mlp_nb.ipynb` to get a feel for the code. 