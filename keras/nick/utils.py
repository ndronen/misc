def count_parameters(model):
    n = 0
    for layer in model.layers:
        for param in layer.params:
        n += np.prod(param.shape.eval())
    return n
