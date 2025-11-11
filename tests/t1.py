from otter.test_files import test_case

OK_FORMAT = False

name = "Exercise 3.1"
points = 4

@test_case(points=4)
def test_1(env):
    lr = 0.1
    batch_size = 64
    num_epochs = 800
    seed = 1234

    train_func = env['assignment_ex1']

    W, b, losses, train_accuracy = train_func(lr, batch_size, num_epochs, seed)
    assert losses[0] > losses[-1], "Loss did not decrease during training"
    assert train_accuracy >= 0.90, "Training accuracy is too low"
