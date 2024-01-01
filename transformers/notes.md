# Attention


## Forward pass

- We select a chunk of text from our data set
- Similar to Bengio et al, we build our embedding table
- There is no concatenation of embedded tokens to represent a context window. Instead we create positionally encoded embeddings and then add these embeddings to our embedding table.
- Pass through attention head
  - Compute attention scores
  - Compute weighted embeddings
- Embedding in embedding out