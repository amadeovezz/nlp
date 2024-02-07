# Attention Based Models

## Novel ideas

- Attention. Attending to specific inputs in our sequence.

### Example

Let us look at a simple example with our `SelfAttentionHead()` class. 

For the text: "Would you"

1. We would encode the text into a vector:

```python
[35, 53, 59, 50, 42, 1, 63, 53, 59]
```

2. Let us generate embeddings for each token:

```python
embeddings
 tensor([[-0.9205, -0.8238], # W
         [ 0.5364, -1.5131], # o
         [ 0.1597,  0.6444], # u
         [-0.6822,  0.4506], # l
         [ 1.2922, -0.9028], # d
         [ 0.7594,  1.1730], # ' '
         [-0.3377,  1.0273], # y
         [ 1.6784,  0.9476], # o
         [ 0.1044, -1.3956]],# u
        dtype=torch.float64),
```

3. Generate Keys, Queries, and Values

`SelfAttentionHead()` has three parameters: `Keys, Queries and Values`. 
All of them perform a linear transformation with our incoming input:

```python
queries = input @ self.Query
keys = input @ self.Key
values = input @ self.Value
```

4. Generate our attention matrix 

```python
queries @ keys.T
```

We have our queries:

```python
tensor([[ 1.0779, -0.5741], # W
        [ 0.4964, -0.6477], # o
        [-0.4699,  0.3467], # u
        [ 0.2001,  0.0974], # l
        [-0.3513, -0.2088], # d
        [-1.1676,  0.7167], # ' '
        [-0.3546,  0.4445], # y
        [-1.6525,  0.7736], # o
        [ 0.7178, -0.6687]],# u
       dtype=torch.float64)
```

Note that: `keys.T` can be thought of as,

```python
         # W       # o      # u      # l      # d      # ' '   # y       # o      # u
tensor([[ 1.5008, -0.4762, -0.3606,  0.9001, -1.6952, -1.3368,  0.2849, -2.6256,  0.1274],
        [ 0.6880,  2.4543, -0.8378, -1.0102,  1.9840, -1.2742, -1.6521, -0.4601,  2.0550]], dtype=torch.float64)
```

And then when we perform this matrix multiplication, we take each embedding in our query and multiply by each 
embedding in our keys. Ie:

The embedding for the token 'W' from our query: `[ 1.0779, -0.5741]` interacts with all the other embeddings from our keys:

Ie: W: `[1.5008, 0.6880]`, o: `[-0.4762, 2.4543]`, etc...

Conceptually all of our tokens are "communicating" with each other. 


```python

        # W       # o      # u      # l      # d      # ' '   # y       # o      # u
   W  ([[ 1.2227, -1.9223,  0.0923,  1.5502, -2.9663, -0.7094,  1.2556, -2.5659, -1.0425],
   o    [ 0.2995, -1.8260,  0.3636,  1.1011, -2.1265,  0.1616,  1.2114, -1.0055, -1.2677],
   u    [-0.4667,  1.0748, -0.1210, -0.7733,  1.4845,  0.1864, -0.7067,  1.0743,  0.6527],
   l    [ 0.3674,  0.1439, -0.1538,  0.0817, -0.1459, -0.3917, -0.1040, -0.5702,  0.2257],
   d    [-0.6709, -0.3452,  0.3016, -0.1053,  0.1812,  0.7357,  0.2449,  1.0184, -0.4739],
  ' '   [-1.2592,  2.3151, -0.1795, -1.7750,  3.4013,  0.6476, -1.5167,  2.7358,  1.3242],
   y    [-0.2264,  1.2599, -0.2445, -0.7683,  1.4831, -0.0923, -0.8354,  0.7266,  0.8683],
   o    [-1.9479,  2.6855, -0.0523, -2.2689,  4.3361,  1.2234, -1.7488,  3.9829,  1.3792],
   u    [ 0.6173, -1.9829,  0.3014,  1.3216, -2.5435, -0.1076,  1.3092, -1.5771, -1.2826]], dtype=torch.float64)
```

5. Normalize and use a mask

Here we are using a decoder mask.

```python
A=
        # W       # o      # u      # l      # d      # ' '   # y       # o      # u
   W  ([[ 1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
   o    [ 0.8934,  0.1066,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
   u    [ 0.1412,  0.6594,  0.1994,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
   l    [ 0.3180,  0.2543,  0.1888,  0.2389,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
   d    [ 0.1095,  0.1516,  0.2895,  0.1927,  0.2567,  0.0000,  0.0000,  0.0000,  0.0000],
  ' '   [ 0.0066,  0.2337,  0.0193,  0.0039,  0.6925,  0.0441,  0.0000,  0.0000,  0.0000],
   y    [ 0.0704,  0.3114,  0.0692,  0.0410,  0.3892,  0.0805,  0.0383,  0.0000,  0.0000],
   o    [ 0.0010,  0.0981,  0.0063,  0.0007,  0.5110,  0.0227,  0.0012,  0.3590,  0.0000],
   u    [ 0.1513,  0.0112,  0.1103,  0.3059,  0.0064,  0.0733,  0.3021,  0.0169,  0.0226]], dtype=torch.float64)
```

We effectively get a matrix $A$ where each value in [row,column] represents a how much each 
character (query) in the sequence "Would you" should pay attention to every other character (key) in the sequence.

6. Compute our new embeddings

```python
# Attention Matrix: 9x9, Value Matrix: 9x2
# 9x9 @ 9x2 = 9x2
tensor([[ 2.5039, -0.4066], # W
        [ 0.3856,  0.6750], # o
        [-0.8985, -0.0397], # u
        [ 0.8735, -0.5345], # l
        [-1.6091,  1.0233], # d
        [-2.5223,  0.2270], # ' '
        [-0.3118, -0.4414], # y
        [-4.0523,  0.8632], # o
        [ 1.0940,  0.3734]],# u
       dtype=torch.float64)
```

```python
new_embeddings = 

  # [(W,W) @ W , (W,W) @ W ]
W = [1 * 2.5039, 1 * -0.4066] 

  # [(o,W) @ W         + (o,o) @ o          , (o,W) @ W          + (o,o) @ o]
o = [(0.8934 * 2.5039) +  (0.1066 * -0.4066), (0.8934 * -0.4066) +  (0.1066 * 0.6750) ] 

  # [ (u,W) @ W          +  (u,o) @ o           (u,o) @ u         , (u,W) @ W          +  (u,o) @ o           (u,o) @ u
u = [ (0.1412 *  2.5039) + (0.6594 *  0.3856) + (0.1994 * -0.8985), (0.1412 *  -0.4066) + (0.6594 * 0.6750) + (0.1994 * -0.0397),  ]

... 
```

Conceptually this operation takes each token from our original (albeit transformed) embedding, 
and then weighs how much that token interacts with other tokens. This is easily expressed through a matrix multiplication.

So for instance the token "o", is attending more to "W", than itself:

`(0.8934 * 2.5039) +  (0.1066 * -0.4066) ,  (0.8934 * -0.4066) +  (0.1066 * 0.6750)`

## Sequential modeling

- Each token is encoded positionally using a positional encoding func
- By applying attention, we do also attend and aggregate other tokens in the sequence and encode this information
into new embeddings.

## Forward pass

- We select a chunk of text from our data set
- Similar to Bengio et al, we build our embedding table
- There is no concatenation of embedded tokens to represent a context window. Instead we create positionally encoded embeddings and then add these embeddings to our embedding table.
- Pass in embedding to each attention block
  - Split up dimensionality of embedding 
  - Pass through multiple attention heads
  - Concatenate results back to original dimension
  - Linear transformation
  - Residual connection + layer norm
  - Feedforward
  - Residual connection + layer norm
  - Feed in final embedding back to next attention block
- Final linear transformation of embedding to logits

### Example

Recall from the text: "First", our `forward()` func might receive:

Note using the generator: `g = torch.Generator().manual_seed(2147483647)`

5 batches with a token length / context window of size 3.

```python
tensor([[ 0.,  0.,  0.], # [...]
        [ 0.,  0., 18.], # [..F]
        [ 0., 18., 47.], # [.Fi]
        [18., 47., 56.], # [Fir]
        [47., 56., 57.]]) # [irs]
```

1. Lets focus on the last two batches
```python
tensor([[18., 47., 56.], # [Fir]
        [47., 56., 57.]]) # [irs]
```

and some note-worthy hyper-parameters:

```python
hp = {
    "init_learning_rate": .1,
    "converging_learning_rate": .01,
    "epochs": 100000,
    "dim_of_embedding": 10,
    "num_of_attention_heads": 2,
    "num_of_attention_blocks": 2,
    "num_layer_1_nodes": 15,
    "mini_batch_size": 3,
    "token_length": token_length,
}
```

2. Create embedding table where $d=2$. This tensor would be $65x2$
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

3. Indexing into our embedding table:

```python
tensor([[[ 0.8932,  0.4439, -1.4547, -0.1504, -0.8794, -0.7635,  0.6127,  0.2404,  1.9926,  0.6106],  # F - Batch 1
         [ 0.8862, -1.0162,  0.0958,  0.6960,  0.8061, -0.0581, -0.6898,  1.7496,  0.5523,  0.2296],  # i
         [-0.9970,  1.7864, -0.1118,  0.1671,  1.1816,  0.1800,  1.5720,  0.0566, -1.3830, -2.3233]], # r
        
        [[ 0.8862, -1.0162,  0.0958,  0.6960,  0.8061, -0.0581, -0.6898,  1.7496,  0.5523,  0.2296],   # i - Batch 2
         [-0.9970,  1.7864, -0.1118,  0.1671,  1.1816,  0.1800,  1.5720,  0.0566, -1.3830, -2.3233],   # r
         [-2.3003, -0.6537, -0.0925,  0.6157,  0.7690,  0.5830,  0.4646, -0.0359, -0.9710, -1.7007]]], # s
       dtype=torch.float64)

```

4. Generate a positional embedding:

```python
tensor([[ 0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00], # position 1
        [ 8.4147e-01,  5.4030e-01,  1.5783e-01,  9.8747e-01,  2.5116e-02,  9.9968e-01,  3.9811e-03,  9.9999e-01,  6.3096e-04,  1.0000e+00], # position 2
        [ 9.0930e-01, -4.1615e-01,  3.1170e-01,  9.5018e-01,  5.0217e-02,  9.9874e-01,  7.9621e-03,  9.9997e-01,  1.2619e-03,  1.0000e+00]]) # position 3
```

5. Combine original embedding + positional

```python
tensor([[[ 0.8932,  1.4439, -1.4547,  0.8496, -0.8794,  0.2365,  0.6127,  1.2404,  1.9926,  1.6106],  # F - Embedding 1 - Batch 1
         [ 1.7277, -0.4759,  0.2537,  1.6835,  0.8312,  0.9416, -0.6858,  2.7496,  0.5530,  1.2296],  # i
         [-0.0877,  1.3702,  0.1999,  1.1173,  1.2319,  1.1787,  1.5800,  1.0566, -1.3817, -1.3233]], # r
        
        [[ 0.8862, -0.0162,  0.0958,  1.6960,  0.8061,  0.9419, -0.6898,  2.7496,  0.5523,  1.2296],   # i - Embedding 2 - Batch 2
         [-0.1555,  2.3267,  0.0460,  1.1546,  1.2068,  1.1796,  1.5760,  1.0566, -1.3823, -1.3233],   # r
         [-1.3910, -1.0698,  0.2192,  1.5659,  0.8192,  1.5818,  0.4725,  0.9641, -0.9698, -0.7007]]], # s
       dtype=torch.float64)
```

6. Start attention block 

7.Perform multi-headed attention

In this example we have 2 attention heads.

7a. Break up the dimensionality of the embedding vector, since dim 10 / 2 

```python
t1 = tensor([[[ 0.8932,  1.4439, -1.4547,  0.8496, -0.8794],  # F - 1/2 of Embedding 1 - Batch 1
              [ 1.7277, -0.4759,  0.2537,  1.6835,  0.8312],  # i
              [-0.0877,  1.3702,  0.1999,  1.1173,  1.2319]], # r
 
             [[ 0.8862, -0.0162,  0.0958,  1.6960,  0.8061],  # i - 1/2 of Embedding 2 - Batch 2
              [-0.1555,  2.3267,  0.0460,  1.1546,  1.2068],  # r
              [-1.3910, -1.0698,  0.2192,  1.5659,  0.8192]]],# s
 
t2 = tensor([[[ 0.2365,  0.6127,  1.2404,  1.9926,  1.6106],  # F - 2/2 of Embedding 1 - Batch 1
              [ 0.9416, -0.6858,  2.7496,  0.5530,  1.2296],  # i
              [ 1.1787,  1.5800,  1.0566, -1.3817, -1.3233]], # r
 
              [[ 0.9419, -0.6898,  2.7496,  0.5523,  1.2296],  # i - 2/2 of Embedding 2 - Batch 2
              [ 1.1796,  1.5760,  1.0566, -1.3823, -1.3233],   # r
              [ 1.5818,  0.4725,  0.9641, -0.9698, -0.7007]]], # s
            dtype=torch.float64) 
```

7b. Compute attention embeddings for each portion of the Embedding.

Note: the attention heads do not communicate across batches.

```python

t1_embedding = attention_head_1(t1)
t1_embedding

tensor([[[ 2.3131, -2.2959, -2.6561, -0.7486, -2.5572],  # F - 1/2 of Embedding 1 - Batch 1
         [ 2.3131, -2.2959, -2.6560, -0.7486, -2.5572],  # i
         [ 1.7580, -0.1611, -0.2066,  2.7238, -0.7746]], # r
 
       [[ 2.4746, -0.9314,  0.7887,  1.7464, -0.7056],   # i - 1/2 of Embedding 2 - Batch 2
        [ 2.4746, -0.9314,  0.7887,  1.7464, -0.7056],   # r
        [ 2.7756, -0.1980,  2.5124,  0.1292,  0.9958]]], # s
        dtype=torch.float64)


t2_embedding = attention_head_2(t2)
t2_embedding 

tensor([[[ 2.8436,  1.9489, -5.2310,  2.2372, -1.2986],  # F - 2/2 of Embedding 1 - Batch 1
         [ 2.8463,  1.9206, -5.2370,  2.2567, -1.3013],  # i
         [ 2.8437,  1.9479, -5.2312,  2.2379, -1.2987]], # r
 
        [[ 3.3084, -2.9880, -6.2827,  5.6306, -1.7786],   # i - 2/2 of Embedding 2 - Batch 2
         [ 3.2741, -2.9785, -6.2380,  5.5927, -1.7743],   # r
         [ 2.2595, -2.8100, -4.6859,  4.5792, -1.3124]]], # s
        dtype=torch.float64)
```

7c. Re-assemble the dimensionality of the vectors:

```
python
tensor([[[ 2.3131, -2.2959, -2.6561, -0.7486, -2.5572,  2.8436,  1.9489, -5.2310,  2.2372, -1.2986],   # F - Embedding 1 - Batch 1
         [ 2.3131, -2.2959, -2.6560, -0.7486, -2.5572,  2.8463,  1.9206, -5.2370,  2.2567, -1.3013],   # i
         [ 1.7580, -0.1611, -0.2066,  2.7238, -0.7746,  2.8437,  1.9479, -5.2312,  2.2379, -1.2987]],  # r
         
        [[ 2.4746, -0.9314,  0.7887,  1.7464, -0.7056,  3.3084, -2.9880, -6.2827,  5.6306, -1.7786],   # i - Embedding 2 - Batch 2
         [ 2.4746, -0.9314,  0.7887,  1.7464, -0.7056,  3.2741, -2.9785, -6.2380,  5.5927, -1.7743],   # r
         [ 2.7756, -0.1980,  2.5124,  0.1292,  0.9958,  2.2595, -2.8100, -4.6859,  4.5792, -1.3124]]], # s
         dtype=torch.float64)

```

7d. We typically perform a linear projection

```python

tensor([[[ -9.0823,   6.3004,   9.1679,   8.5177,   6.0050, -13.2357,   7.9666,   3.9298,  15.1189,   9.7685],  # F
         [ -9.1197,   6.3115,   9.2053,   8.4901,   6.0336, -13.2849,   7.9820,   3.9339,  15.1051,   9.7994],  # i
         [ -4.5924,   5.9945,  10.0233,  -0.9045,   6.1016, -10.2484,   8.0820,   8.1236,   9.3909,   5.2980]], # r
        
        [[-10.9785,   6.6434,  17.1156,  -2.8052,   7.3171, -19.6526,   9.9866,   5.3723,   8.8132,   9.7098],  # i
         [-10.9045,   6.5819,  17.0236,  -2.7863,   7.2185, -19.5520,   9.9345,   5.3414,   8.7142,   9.6055],  # r
         [ -8.1251,   0.7911,  14.8728,   0.7814,   0.9000, -16.3314,   6.8973,  -1.7805,   7.5881,   5.5321]]],# s
```

8. We then will compute a residual connection and layer norm:

```python
tensor([[[-1.4735,  0.2932,  0.2897,  0.4731,  0.0028, -2.0069,  0.3858,  0.0078,  1.3318,  0.6962],  # F
         [-1.4828,  0.0594,  0.4818,  0.5651,  0.1793, -2.0600,  0.2296,  0.1582,  1.2045,  0.6648],  # i
         [-1.3417,  0.4738,  0.9047, -0.6042,  0.4691, -2.0033,  0.8201,  0.7475,  0.5710, -0.0371]], # r
        
        [[-1.2959,  0.2441,  1.2189, -0.4685,  0.3819, -2.0897,  0.4900,  0.3817,  0.4963,  0.6412],   # i
         [-1.3561,  0.4803,  1.2308, -0.4891,  0.4358, -2.0286,  0.7196,  0.2494,  0.3353,  0.4227],   # r
         [-1.2676, -0.1812,  1.6266,  0.1277,  0.0538, -1.8831,  0.7184, -0.2444,  0.6300,  0.4198]]], # s
```

9. Now we are ready to pass our embeddings through our feedforward network.

```python
mlp_out 

tensor([[[ 0.9996, -0.9979, -1.0000, -0.9997,  0.9995,  0.9998, -0.9976,  0.5140,  0.9172, -0.9999],  # F
         [ 0.9905, -0.9915, -1.0000, -0.9983,  0.9979,  1.0000, -0.9981,  0.5337,  0.7970, -1.0000],  # i
         [ 1.0000, -0.8240, -1.0000, -0.9953,  0.9995,  0.9997, -1.0000,  0.9978,  0.8199, -0.9867]], # r
        
        [[ 0.9847, -0.9946, -1.0000, -0.9982,  0.9998,  1.0000, -0.9934,  0.8133,  0.9439, -0.9861],   # i
         [ 0.9987, -0.9674, -1.0000, -0.9984,  0.9997,  1.0000, -0.9983,  0.9306,  0.9299, -0.9955],   # r
         [-0.9557, -0.6627, -0.9974, -0.7846,  0.9740,  1.0000, -0.9993,  0.6905, -0.0432, -0.9999]]], # s
       dtype=torch.float64)
```

9. We perform another residual connection and layer norm to get our final embeddings for this attention block.
10. We then repeat steps 6, except this time our input into the attention block, are the embeddings from the last block.
11. Once we have processed all of our attention blocks we get our final embeddings. We have one more linear
transformation to transform our embeddings into logits.

## Reducing loss

- Layer norm + residual connections 
- Using a decoder model
- Using multiple attention heads

