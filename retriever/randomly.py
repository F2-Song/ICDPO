import random
random.seed(42)

def random_retrieval(
    trainset,
    num_demonstrations,
    test_samples,
):
    demos = []
    for sample in test_samples:
        demos.append(random.sample(trainset, num_demonstrations))
    return demos