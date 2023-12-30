# Attention


## Input setup

- We select a chunk of text from our data set
- Similar to Bengio et al, we build our embedding table
- There is no concatenation of embedded tokens to represent a context window. Instead we create positionally encoded embeddings and then add these embeddings to our embedding table.
- Pass through attention head
  - Compute attention scores
  - Compute weighted embeddings
- Embedding in embedding out


### Simple Example

1. Encode the entire alphabet. Assume we give each character an index. a->0, b->1, etc...
   a. For our Shakespear text we have 65 unique characters
2. We select a chunk of text from our dataset Lets say $n=5$, and we randomly grab some data that captures the word "First". It would look like:
```python
tensor([
    [ 0., 18., 47., 56., 57. ], # [.First]
```
Note: There is some padding going on here since it is the first word in our data set.

4. Create embedding table where $d=2$. This tensor would be $65x2$
```python
tensor([[-1.0219, -0.3420], # \n
        [ 0.2027, -0.8522], # ' '
        [-0.9633, -0.5323], # !
        [-0.5090,  1.0174], # &
        [ 1.4462,  1.5661], # '
        [ 0.2440,  0.3680], # ,
        [ 1.6532, -0.6972], # - 
        [-0.0586,  1.2453], # .
        ...                 # All other characters in the dataset including ABC...
```
5. Indexing into our embedding table:
```python
tensor([[[-1.0219, -0.3420], # . / Batch 1
         [-1.0219, -0.3420], # .
         [-1.0219, -0.3420]],# .

        [[-1.0219, -0.3420], # . / Batch 2
         [-1.0219, -0.3420], # .
         [ 0.1963, -1.4404]],# F

        [[-1.0219, -0.3420], # . / Batch 3
         [ 0.1963, -1.4404], # F
         [-0.2019,  1.1584]],# i
        ...
```

## NN points

```



## Key points




