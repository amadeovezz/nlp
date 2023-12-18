# Vanilla RNN's

## Input setup

- Similar to Bengio et al, we build our embedding table
- There is no concatenation of embedded tokens to represent a context window

### Simple Example

(Note this is duplicated from `mlp/note.md`)

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
6. Organize by tokens:

[Batch, Token, Embedding] -> [Token, Batch, Embedding]
```python
tensor([[[-1.0219, -0.3420], # . / Batch 1
         [-1.0219, -0.3420], # . / Batch 2
         [-1.0219, -0.3420]],# . / Batch 3

        [[-1.0219, -0.3420],  # . / Batch 1
         [-1.0219, -0.3420],  # . / Batch 2
         [ 0.1963, -1.4404]], # F / Batch 3

        [[-1.0219, -0.3420],  # . / Batch 1
         [ 0.1963, -1.4404],  # F / Batch 2
         [-0.2019,  1.1584]], # i / Batch 3
```

## NN points

For our RNN, the dependant variables are:

- vector_dim influences weights/bias size in layer_1, ie:
```python 
vector_dim = len([-1.0219, -0.3420])
layer_1(in=vector_dim)
```



## Key points

- We build up on Bengio et al. embeddings
- We no longer concatenate embeddings, and instead use our neural net to model dependencies across tokens
  - This is done by feeding in each one token in a sequence at a time through all layers
  - For every token $t_n$ that is processed through a linear layer, we aggregate the $t_{n-1}$ result 
- Tokens in seqeunces can not be evaluated in parrallel
