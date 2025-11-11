"""Submission for exercise sheet 2

SUBMIT this file as submission_<STUDENTID>.py where
you replace <STUDENTID> with your student ID, e.g.,
submission_1234567.py
"""
import torch
import torch.nn.functional as F

# DO NOT MODIFY
def generate_classification_data(
    n_per_class: int = 200,
    std: float = 0.5,
    seed: int = 1234
):
    torch.manual_seed(seed)
    mean0 = torch.tensor([-1.0, 0.0])
    mean1 = torch.tensor([1.0, 0.0])

    class0 = mean0 + std * torch.randn(n_per_class, 2)
    class1 = mean1 + std * torch.randn(n_per_class, 2)

    X = torch.cat([class0, class1], dim=0)
    y = torch.cat([
        torch.zeros(n_per_class, dtype=torch.float32),
        torch.ones(n_per_class, dtype=torch.float32)
    ], dim=0)

    return X, y


# Exercise 1.1
def assignment_ex1(
    lr: float = 0.1, 
    batch_size: int = 64,
    num_epochs: int = 800,
    seed: int=1234):
    losses = []
    X, y = generate_classification_data(
        n_per_class=200,
        std=0.5,
        seed=seed
    )
    N, D = X.shape

    # DO NOT MODIFY
    torch.manual_seed(seed)
    
    W = # YOUR CODE GOES HERE
    b = # YOUR CODE GOES HERE
    
    for _ in range(num_epochs):
        # Implement selecting a mini-batch of size `batch_size` 
        # from the data, store e.g., in X_batch and y_batch
        # YOUR CODE GOES HERE
        
        # Compute logits = WX_batch + b for the mini-batch,
        # YOUR CODE GOES HERE 

        # Compute the binary cross-entropy loss via (adjust
        # variable naming if required)  
        loss = F.binary_cross_entropy_with_logits(logits, y_batch)
        
        # Backpropagate to compute gradients
        # YOUR CODE GOES HERE 

        # Update parameters
        # YOUR CODE GOES HERE 

        # Store the loss values per batch
        losses.append(float(loss.item()))

    # Compute training accuracy
    with torch.no_grad():
        # Compute logits = WX + b for the full dataset
        # YOUR CODE GOES HERE
        
        # Compute predictions by assigning all points with 
        # sigmoid(logits_full) >= 0.5 to class 1 and the 
        # rest to class 0
        # YOUR CODE GOES HERE
        
        # Compute training accuracy (as training_accuracy) 
        # as the fraction of correctly classified points
        train_accuracy = # YOUR CODE GOES HERE

        
    return W, b, losses, train_accuracy
