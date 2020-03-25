import numpy as np
import torch

from latent_rationale.beer.util import pad
from latent_rationale.beer.util import initialize_model_ as beer_initialize_model_
from latent_rationale.beer.util import decorate_token as beer_decorate_token
from latent_rationale.beer.util import get_minibatch as beer_get_minibatch
from latent_rationale.beer.util import print_parameters as beer_print_parameters
from latent_rationale.beer.util import get_predict_args as beer_get_predict_args
from latent_rationale.beer.util import get_device as beer_get_device
from latent_rationale.beer.util import find_ckpt_in_directory as beer_find_ckpt_in_directory
from latent_rationale.beer.util import get_args as beer_get_args
from latent_rationale.sst.util import load_glove as sst_load_glove


def get_args():
    return beer_get_args()


def print_parameters(model):
    return beer_print_parameters(model)


def initialize_model_(model):
    return beer_initialize_model_(model)


def get_predict_args():
    return beer_get_predict_args()


def load_glove(glove_path, vocab, glove_dim=300):
    return sst_load_glove(glove_path, vocab, glove_dim=glove_dim)


def get_device():
    return beer_get_device()


def find_ckpt_in_directory(path):
    return beer_find_ckpt_in_directory(path)


def decorate_token(t, z_):
    return beer_decorate_token(t, z_)


def get_minibatch(data, batch_size=256, shuffle=False):
    return beer_get_minibatch(data, batch_size=batch_size, shuffle=shuffle)


def prepare_minibatch(mb, vocab, device=None, sort=True):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    # batch_size = len(mb)
    lengths = np.array([len(ex.tokens) for ex in mb])
    maxlen = lengths.max()
    reverse_map = None

    # vocab returns 0 if the word is not there
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]
    y = [ex.score for ex in mb]

    x = np.array(x)
    y = np.array(y, dtype=np.float32)

    if sort:  # required for LSTM
        sort_idx = np.argsort(lengths)[::-1]
        x = x[sort_idx]
        y = y[sort_idx]

        # create reverse map
        reverse_map = np.zeros(len(lengths), dtype=np.int32)
        for i, j in enumerate(sort_idx):
            reverse_map[j] = i

    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)

    return x, y, reverse_map
