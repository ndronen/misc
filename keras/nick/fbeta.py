eps = 1e-20

def support(Y):
    return Y.sum(axis=0)

def true_positive(Y, Y_hat):
    return ((Y_hat == Y) & (Y == 1)).sum(axis=0)

def make_y_diff(Y, Y_hat):
    return Y_hat - Y

def false_positive(Y_diff):
    return (Y_diff == 1).sum(axis=0)

def true_negative(Y_diff):
    return (Y_diff == 0).sum(axis=0)

def false_negative(Y_diff):
    return (Y_diff == -1).sum(axis=0)

def precision(Y, Y_hat, eps=1e-9, Y_diff=None):
    tp = true_positive(Y, Y_hat)
    if Y_diff is None:
        Y_diff = make_y_diff(Y, Y_hat)
    fp = false_positive(Y_diff)
    return tp/(tp+fp+eps)

def recall(Y, Y_hat, eps=1e-9, Y_diff=None):
    tp = true_positive(Y, Y_hat)
    if Y_diff is None:
        Y_diff = make_y_diff(Y, Y_hat)
    fn = false_negative(Y_diff)
    return tp/(tp+fn+eps)

def fbeta_loss(Y, Y_hat, beta=0.5, eps=1e-9, average=None):
    """
    Returns the negative of the F_beta measure, because the
    optimizer is trying to minimize the objective.
    """
    Y_diff = make_y_diff(Y, Y_hat)
    pr = precision(Y, Y_hat, eps=eps, Y_diff=Y_diff)
    rc = recall(Y, Y_hat, eps=eps, Y_diff=Y_diff)

    f_per_class = ( (1 + beta**2) * (pr * rc) ) / (beta**2 * pr + rc + eps)

    if average is None:
        f = f_per_class
    elif average == 'macro':
        f = f_per_class.mean()
    elif average == 'weighted':
        s = support(Y)
        f = ((f_per_class * s) / s.sum()).sum()

    return -f
