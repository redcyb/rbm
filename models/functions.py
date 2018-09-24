import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def linear(x):
    return x


def derivative(func_name, x):
    if func_name == "sigmoid":
        y = sigmoid(x)

        if isinstance(x, (np.ndarray, np.generic)):
            return y * (np.ones(y.shape[0]) - y)
        return y * (1 - y)

    if func_name == "linear":
        if isinstance(x, (np.ndarray, np.generic)):
            return np.ones(x.shape[0])
        return 1

    if func_name == "tanh":
        return 1 / np.power(np.cosh(x), 2)

    raise Exception(f"Unknown function: {func_name}")


def derivative_batch(func_name, x):
    if func_name == "sigmoid":
        y = sigmoid(x)

        if isinstance(x, (np.ndarray, np.generic)):
            return y * (np.ones(y.shape) - y)
        return y * (1 - y)

    if func_name == "linear":
        if isinstance(x, (np.ndarray, np.generic)):
            return np.ones(x.shape[0])
        return 1

    if func_name == "tanh":
        return 1 / np.power(np.cosh(x), 2)

    raise Exception(f"Unknown function: {func_name}")


def activate(func_name, x):

    if func_name == "sigmoid":
        return sigmoid(x)

    if func_name == "tanh":
        return np.tanh(x)

    if func_name == "linear":
        return x

    raise Exception(f"Unknown function: {func_name}")
