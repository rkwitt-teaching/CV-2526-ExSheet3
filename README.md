# Exercise set 1

*All assignments need to be implemented within the function skeletons found in `submission.py`
and you need to hand in this file in the form `submission_<STUDENTID>.py` at the link provided
for this exercise sheet via e-mail.*

## Exercise 3.1

In this exercise, we first create a simple 2D binary classification problem with two Gaussian clusters (code provided):

- **Class 0**: centered at `(-1, 0)`
- **Class 1**: centered at `(1, 0)`

We will use a **logistic regression** model:

$z = XW + b, \quad \hat{y} = \sigma(z)$

where:
- $X$ has shape `(N, 2)`
- $W$ has shape `(2, 1)`
- $b$ has shape `(1,)` or `(1, 1)`
- $\sigma$ is the sigmoid function (applied component-wise)

We will then minimize the **binary cross-entropy loss** using gradient descent. You do not have to implement the binary cross-entropy loss yourself, but instead use `torch.nn.functional.binary_cross_entropy_with_logits`.

Implement the whole training process (within which
- initializes `W` (with entries drawn from a standard Gaussian with zero mean and variance of 1) and `b` (with all zeros) with `requires_grad=True`
- performs mini-batch gradient descent (the grader will run with batch size of 64, a learning rate of 0.1 and 800 epochs)
- uses `torch.nn.functional.binary_cross_entropy_with_logits` as the loss
- returns `(W, b, losses, train_accuracy)`.

The otter-grader will check that:
1. The training loss decreases.
2. The final training accuracy is reasonably high (i.e. > 90%).