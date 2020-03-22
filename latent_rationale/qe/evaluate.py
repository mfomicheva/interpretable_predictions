import numpy as np
import torch

from collections import defaultdict

from latent_rationale.beer.util import get_minibatch, decorate_token
from latent_rationale.qe.util import prepare_minibatch


def get_examples(model, data, num_examples=3, batch_size=1, device=None):
    """Prints examples"""

    model.eval()  # disable dropout
    count = 0

    if not hasattr(model, "z"):
        return

    for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):

        if count == num_examples:
            break

        x, targets, _ = prepare_minibatch(mb, model.vocab, device=device)
        with torch.no_grad():
            output = model(x)

            if hasattr(model, "z"):
                z = model.z.cpu().numpy().flatten()
                example = []
                for ti, zi in zip(mb[0].tokens, z):
                    example.append(decorate_token(ti, zi))
                # print("Example %d:" % count, " ".join(output))
                yield example
                count += 1


def evaluate_loss(model, data, batch_size=256, device=None, cfg=None):
    """
    Loss of a model on given data set (using minibatches)
    Also computes some statistics over z assignments.
    """
    model.eval()  # disable dropout
    total = defaultdict(float)
    total_examples = 0
    total_predictions = 0

    for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):

        x, targets, _ = prepare_minibatch(mb, model.vocab, device=device)
        mask = (x != 1)

        batch_examples = targets.size(0)
        batch_predictions = np.prod(list(targets.size()))

        total_examples += batch_examples
        total_predictions += batch_predictions

        with torch.no_grad():
            output = model(x)
            loss, loss_opt = model.get_loss(output, targets, mask=mask)
            total["loss"] += loss.item() * batch_examples

            # e.g. mse_loss, loss_z_x, sparsity_loss, coherence_loss
            for k, v in loss_opt.items():
                total[k] += v * batch_examples

    result = {}
    for k, v in total.items():
        if not k.startswith("z_num"):
            result[k] = v / float(total_examples)

    if "z_num_1" in total:
        z_total = total["z_num_0"] + total["z_num_c"] + total["z_num_1"]
        selected = total["z_num_1"] / float(z_total)
        result["p1r"] = selected
        result["z_num_0"] = total["z_num_0"]
        result["z_num_c"] = total["z_num_c"]
        result["z_num_1"] = total["z_num_1"]

    return result
