import numpy as np


def validate_object(inputs, *, validate_fn):
    weights = []
    losses = []
    y_preds = []
    for i in inputs:
        y_pred, loss = validate_fn(*i)
        y_preds.append(y_pred)
        losses.append(loss)
        weights.append(y_pred.size)

    loss = np.average(losses, weights=weights)
    return y_preds, loss


def predict_object(inputs, *, predict_fn):
    return [predict_fn(*i) for i in inputs]
