# Vanilla RNN's

## Novel ideas

- We no longer concatenate embeddings, and instead use our neural net to model dependencies across tokens
    - This is done by feeding in each one token in a sequence at a time through all layers
    - For every token $t_n$ that is processed through a linear layer, we aggregate the $t_{n-1}$ result

## Sequential modeling

- Relationships between tokens are capture via sequential processing with state

## Initialization

vector_dim influences weights/bias size in layer_1, ie:
```python 
vector_dim = len([-1.0219, -0.3420])
layer_1(in=vector_dim)
```

## Optimization

- For all tokens in a sequence $s$, requires $s$ forward passes. 
- Batches cannot be evaluated in parallel 
- We have to be mindful about how we set/reset `previous_pre_activations`.

## Forward pass

- Similar to Bengio et al, we build our embedding table
- There is no concatenation of embedded tokens to represent a context window

### Example

Recall for the text: "First", our `forward()` func might receive:

```python
tensor([[ 0.,  0.,  0.], # [...]
        [ 0.,  0., 18.], # [..F]
        [ 0., 18., 47.], # [.Fi]
        [18., 47., 56.], # [Fir]
        [47., 56., 57.]]) # [irs]
```

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
tensor([[[-1.0219, -0.3420], # . / Token 1 - Batch 1
         [-1.0219, -0.3420], # . / Token 2 - Batch 1
         [-1.0219, -0.3420]],# . / Token 3 - Batch 1

        [[-1.0219, -0.3420], # . / Token 1 - Batch 2
         [-1.0219, -0.3420], # . / Token 2 - Batch 2
         [ 0.1963, -1.4404]],# F / Token 3 - Batch 2

        [[-1.0219, -0.3420], # . / Token 1 - Batch 3
         [ 0.1963, -1.4404], # F / Token 2 - Batch 3
         [-0.2019,  1.1584]],# i / Token 3 - Batch 4
        ...
```

6. Organize by tokens:

[Batch, Token, Embedding] -> [Token, Batch, Embedding]
```python
tensor([[[-1.0219, -0.3420], # . / Token 1 - Batch 1
         [-1.0219, -0.3420], # . / Token 1 - Batch 2
         [-1.0219, -0.3420]],# . / Token 1 - Batch 3

        [[-1.0219, -0.3420],  # . / Token 2 - Batch 1
         [-1.0219, -0.3420],  # . / Token 2 - Batch 2
         [ 0.1963, -1.4404]], # F / Token 2 - Batch 3

        [[-1.0219, -0.3420],  # . / Token 3 - Batch 1
         [ 0.1963, -1.4404],  # F / Token 3 - Batch 2
         [-0.2019,  1.1584]], # i / Token 3 - Batch 3
```