# `base/`

Base contains:

- encoding 
- dataset pre-processing 
- two abstract classes that serve as a reference to build models: `Layer` and `Model`
- an implementation of stochastic gradient descent

If a model adheres to the `Model` interface, then you can pass it into `train.sgd()`.

The `mlp_nb.ipynb` contains a simple multi-layer perceptron. The set-up for the notebooks is all similar:

1. Encode the dataset
2. Build your training data with a context window. Build your training targets.
3. Define your hyper-parameters (`hp`)
4. Create your model
5. Train the model via `train.sgd`
6. Explore the results

## Input set-up


In general, steps 1 and 2 (above) are :

- Choose an encoder
- Encode text
- Select context window $c$ (number of tokens)
- Build train/validation datasets:
  - Training: for each character of our dataset build vectors of size $c$ (sometimes $c+1$)
  - Targets: this varies based on the type of context window (see below)
- Fetch $n$ vectors our training data (this is our stochastic mini-batch)
    - Now we have our input ready for our forward pass: $n$ by $c$ matrix 

Note, there are two ways to build your training/validation inputs and targets:

1. `get_dataset(text_encoded, token_length, context_window="fixed")`

The inputs to targets for the word "what" for $c=3$ would look like:

"w,h,a" -> "t"

We use this type of context window for:

- vanilla mlp
- emb mlp
- rnn's

2. `get_dataset(text_encoded, token_length, context_window="expanding")`

The inputs to targets for the word "what" for $c=3$ would look like:

", ,w,h" -> "w,h,a"

Notice we start our context window one index earlier here, and that our targets include all of the preceding characters.

We use this type of context window for autoregressive attention based models.

#### Example

For the text: "First"

1. Encode the entire alphabet. Assume we give each character an index. a->0, b->1, etc...
   a. For our Shakespear text we have 65 unique characters
2. $c=3$, and we choose a fixed context window. We would have a dataset that looks like: "...", "..f", ".fi". 
We do this for every character in the dataset.
3. Lets say $n=5$ (our mini-batch size), and we randomly grab some data that captures the word "First". It would look like:
```python
tensor([[ 0.,  0.,  0.], # [...]
        [ 0.,  0., 18.], # [..F]
        [ 0., 18., 47.], # [.Fi]
        [18., 47., 56.], # [Fir]
        [47., 56., 57.]]) # [irs]
```
Note: There is some padding going on here since it is the first word in our data set.