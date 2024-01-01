# `base/`

Base contains:

- encoding 
- dataset pre-processing 
- two abstract classes that serve as a reference to build models: `Layer` and `Model`
- an implementation of stochastic gradient descent

If a model adheres to the `Model` interface, then you can pass it into `train.sgd()`.

The `mlp_nb.ipynb` contains a simple multi-layer perceptron. The set-up for the notebooks is all similar:

1. Encode the dataset
2. Build your training data with a specific context window. Build your training targets.
3. Define your hyper-parameters (`hp`)
4. Create your model
5. Train the model via `train.sgd`
6. Explore the results


## Input set-up

The input set-up (steps 1 and 2 above) for all models is the same:

- Choose an encoder
- Encode text
- Select context window $c$ (number of tokens)
- Build the data set:
    - Create vectors of dimension $c$, for each character position in the data sets.
- Grab $n$ vectors from training data (this is our stochastic mini-batch)
    - Now we have our input ready for our forward pass: $n$ by $c$ matrix 

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