# MLP Bengio

A (rough) character based implementation of https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

## Input set-up

- Choose an encoder
- Encode text
- Encode each character with dimension $d$
- Select context window $c$ (number of tokens)
- Build the data set: 
  - Create vectors of dimension $c$, for each character position in the data sets.
- Grab $n$ vectors from training data (this is our stochastic mini-batch)
  - So now we have our input to our neural nn: $n$ by $c$ 
- Initialize an embedding matrix 
  - Choose a dimension $d$ to represent each character 
  - Build a matrix by vocab size by $d$
- For each vector in the $n$ vectors
  - And for each character in the vector
  - Index into the embedding matrix to grab the corresponding embedding
  - Concatenate the embeddings together

### Example

For the text: "First"

1. Encode the entire alphabet. Assume we give each character an index. a->0, b->1, etc...
  a. For our Shakespear text we have 65 unique characters
2. $c=3$, we have a context window that looks like "...", "..f", ".fi"
3. Lets say $n=5$, and we randomly grab some data that captures the word "First". It would look like:
```python
tensor([[ 0.,  0.,  0.], # [...]
        [ 0.,  0., 18.], # [..F]
        [ 0., 18., 47.], # [.Fi]
        [18., 47., 56.], # [Fir]
        [47., 56., 57.]]) # [irs]
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
tensor([[[-1.0219, -0.3420], # .
         [-1.0219, -0.3420], # .
         [-1.0219, -0.3420]],# .

        [[-1.0219, -0.3420], # .
         [-1.0219, -0.3420], # .
         [ 0.1963, -1.4404]],# F

        [[-1.0219, -0.3420], # .
         [ 0.1963, -1.4404], # F
         [-0.2019,  1.1584]],# i
        ...
```
6. Concatenating each inner matrix, so we capture the sequential nature of data:

```python
tensor([[-1.0219, -0.3420, -1.0219, -0.3420, -1.0219, -0.3420], # [...]  / Batch 1
        [-1.0219, -0.3420, -1.0219, -0.3420,  0.1963, -1.4404], # [..F]  / Batch 2
        [-1.0219, -0.3420,  0.1963, -1.4404, -0.2019,  1.1584], # [.Fi]  / Batch 3
        [ 0.1963, -1.4404, -0.2019,  1.1584, -0.6039,  0.9259], # [Fir]  / Batch 4
        [-0.2019,  1.1584, -0.6039,  0.9259, -0.1773, -0.4201]]) # [irs] / Batch 5
``` 

## NN notes

### Dependant variables

- `vector_dim` (context_window * dim_of_embedding) influences weights/bias size in layer_1, ie:
```python 
vector_dim = len(vector[-1.0219, -0.3420, -1.0219, -0.3420, -1.0219, -0.3420])
layer_1(in=vector_dim)
```
- `num_of_unique_chars` 
```python
last_layer
```

### Optmization notes



## Key points

- Introduces the idea of en embedding table.
  - Embeddings are tuned as the network learns.
  - Embeddings table is interpretable
- Context window is fixed
- Relationships between tokens are captured via concatenated vectors
- Batches can be evaluated in parallel 