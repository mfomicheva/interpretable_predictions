import sys
import torch
import torch.optim

from latent_rationale.qe.data import qe_reader
from latent_rationale.qe.util import prepare_minibatch
from latent_rationale.sst.util import load_glove

from latent_rationale.beer.vocabulary import Vocabulary
from latent_rationale.beer.models.model_helpers import build_model
from latent_rationale.beer.util import get_minibatch, decorate_token, \
    print_parameters, get_predict_args, get_device, find_ckpt_in_directory, \
    beer_annotations_reader, beer_reader, load_embeddings
from latent_rationale.beer.evaluate import evaluate_loss, evaluate_rationale
from latent_rationale.common.util import make_kv_string


def predict():
    """
    Make predictions with a saved model.
    """

    predict_cfg = get_predict_args()
    device = get_device()
    print(device)

    # load checkpoint
    ckpt_path = find_ckpt_in_directory(predict_cfg.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    best_iter = ckpt["best_iter"]
    cfg = ckpt["cfg"]

    for k, v in cfg.items():
        print("{:20} : {:10}".format(k, str(v)))

    eval_batch_size = 64

    print("Loading data")
    dev_data = list(qe_reader(cfg["dev_path"]))

    print("dev", len(dev_data))

    print("Loading pre-trained word embeddings")
    vocab = Vocabulary()
    vectors = load_glove(cfg["embeddings"], vocab)  # required for vocab

    # build model
    model = build_model(cfg["model"], vocab, cfg=cfg)

    # load parameters from checkpoint into model
    print("Loading saved model..")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    print("Done")

    print(model)
    print_parameters(model)

    model.eval()

    for mb in get_minibatch(dev_data, batch_size=eval_batch_size, shuffle=False):
        x, targets, reverse_map = prepare_minibatch(
            mb, model.vocab, device=device, sort=True)

        with torch.no_grad():
            logits = model(x)

            # attention alphas
            if hasattr(model, "alphas"):
                alphas = model.alphas
            else:
                alphas = None

            # rationale z
            if hasattr(model, "z"):
                z = model.z  # [B, T]
                bsz, max_time = z.size()
            else:
                z = None

        # the inputs were sorted to enable packed_sequence for LSTM
        # we need to reverse sort them so that they correspond
        # to the original order

        # reverse sort
        alphas = alphas[reverse_map] if alphas is not None else None
        z = z[reverse_map] if z is not None else None  # [B,T]
        logits = logits[reverse_map]
        print(logits.shape)

        # evaluate each sentence in this minibatch
        for mb_i, ex in enumerate(mb):
            tokens = ex.tokens
            # z is [batch_size, time]
            if z is not None:
                z_ex = z[mb_i, :len(tokens)]  # i for minibatch example
                z_ex_nonzero = (z_ex > 0).float()
                z_ex_nonzero_sum = z_ex_nonzero.sum().item()

                # list of decorated tokens for this single example, to print
                example = []
                for ti, zi in zip(tokens, z_ex):
                    example.append(decorate_token(ti, zi))

                # write this sentence
                sys.stdout.write(" ".join(example))
                sys.stdout.write("\n")
                sys.stdout.write(" ".join(["%.4f" % zi for zi in z_ex]))
                sys.stdout.write("\n")
                sys.stdout.write("{}\n".format(logits[mb_i][0]))



if __name__ == "__main__":
    predict()
