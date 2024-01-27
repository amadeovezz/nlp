# Attention Based Models

## Novel ideas

- 
- Attention. Attending to specific inputs in our sequence.

## Sequential modeling

- Sequences are encoding

### General modeling techniques

- Just adding linear transformations allows for weighted calculation
- We have to think about modeling in a new way when it comes to NN
  - What can we model that will the allow the neural network to learn

## Forward pass

- We select a chunk of text from our data set
- Similar to Bengio et al, we build our embedding table
- There is no concatenation of embedded tokens to represent a context window. Instead we create positionally encoded embeddings and then add these embeddings to our embedding table.
- Pass through attention head
  - Compute attention scores
  - Compute weighted embeddings
- Embedding in embedding out

### Example

```python
tensor([[ 0.,  0.,  0.], # [...]
        [ 0.,  0., 18.], # [..F]
        [ 0., 18., 47.], # [.Fi]
        [18., 47., 56.], # [Fir]
        [47., 56., 57.]]) # [irs]
```


### Attention matrix

The question the attention matrix answers is:

"Given a character x, how much does it interact
with other characters in the sequence."


## Reducing loss

- Layer norm + residual connections 
- Using a decoder model
- Using multiple attention heads

