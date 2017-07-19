import theano
import theano.tensor as T


def categorical_crossentropy(predictions, targets):
    # predictions shape: [batch_size, num_units] (softmax output)
    # targets shpae: [batch_size, num_units] (one-hot)
    # return -T.sum(targets * T.log(predictions), axis=predictions.ndim - 1)
    return theano.tensor.nnet.categorical_crossentropy(predictions, targets)


def mse(predictions, targets):
    num_scored_items = predictions.shape[0]
    loss = T.sum((predictions - targets)**2) / num_scored_items
    return 10 * loss


def log_loss(predictions, targets):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    # predictions shape: [batch_size, num_units] (softmax output)
    # targets shpae: [batch_size, num_units] (one-hot)
    loss = -T.sum(targets * T.log(predictions) + (1 - targets) * T.log(1 - predictions), axis=predictions.ndim - 1)
    return loss


def quad_kappa_loss(predictions, targets):
    num_scored_items = predictions.shape[0]
    num_ratings = 5
    tmp = T.tile(T.arange(0, num_ratings).reshape((num_ratings, 1)),
                 reps=(1, num_ratings)).astype(theano.config.floatX)
    weights = (tmp - tmp.T) ** 2 / (num_ratings - 1) ** 2

    hist_rater_a = predictions.sum(axis=0)
    hist_rater_b = targets.sum(axis=0)

    conf_mat = T.dot(predictions.T, targets)

    nom = T.sum(weights * conf_mat)
    denom = T.sum(weights * T.dot(hist_rater_a.reshape((num_ratings, 1)),
                                  hist_rater_b.reshape((1, num_ratings))) /
                  num_scored_items.astype(theano.config.floatX))

    return 10 * nom / denom


def quad_kappa_log_hybrid_loss(y, t, log_scale=0.5, log_offset=0.50):
    log_loss_res = log_loss(y, t)
    kappa_loss_res = quad_kappa_loss(y, t)
    return kappa_loss_res + log_scale * (log_loss_res - log_offset)


def quad_kappa_log_hybrid_loss_clipped(y, t, log_cutoff=0.9, log_scale=0.5):
    log_loss_res = log_loss(y, t)
    kappa_loss_res = quad_kappa_loss(y, t)
    return kappa_loss_res + log_scale * T.clip(log_loss_res, log_cutoff, 10 ** 3)
