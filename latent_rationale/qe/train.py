import os
import time
import datetime
import json

import torch
import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, ExponentialLR

from torch.utils.tensorboard import SummaryWriter

from latent_rationale.beer.constants import PAD_TOKEN
from latent_rationale.beer.models.model_helpers import build_model
from latent_rationale.beer.vocabulary import Vocabulary
from latent_rationale.beer.util import get_args, get_minibatch, print_parameters, load_embeddings, initialize_model_
from latent_rationale.sst.util import load_glove
from latent_rationale.qe.evaluate import get_examples, evaluate_loss
from latent_rationale.common.util import make_kv_string
from latent_rationale.qe.data import qe_reader
from latent_rationale.qe.util import prepare_minibatch


def train():
    """
    Main training loop.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    cfg = get_args()
    cfg = vars(cfg)

    for k, v in cfg.items():
        print("{:20} : {:10}".format(k, str(v)))

    num_iterations = cfg["num_iterations"]
    print_every = cfg["print_every"]
    eval_every = cfg["eval_every"]
    batch_size = cfg["batch_size"]
    eval_batch_size = cfg.get("eval_batch_size", batch_size)
    num_examples = cfg.get("num_examples", 3)

    # Let's load the data into memory.
    print("Loading data")

    train_data = list(qe_reader(cfg["train_path"], max_len=cfg["max_len"]))
    dev_data = list(qe_reader(cfg["dev_path"], max_len=cfg["max_len"]))

    print("train", len(train_data))
    print("dev", len(dev_data))

    iters_per_epoch = len(train_data) // batch_size

    if eval_every == -1:
        eval_every = iters_per_epoch
        print("eval_every set to 1 epoch = %d iters" % eval_every)

    if num_iterations < 0:
        num_iterations = -num_iterations * iters_per_epoch
        print("num_iterations set to %d iters" % num_iterations)

    example = dev_data[0]
    print("First train example tokens:", example.tokens)
    print("First train example scores:", example.score)

    print("Loading pre-trained word embeddings")
    vocab = Vocabulary()
    vectors = load_glove(cfg["embeddings"], vocab)

    # build model
    model = build_model(cfg["model"], vocab, cfg=cfg)
    initialize_model_(model)

    # load pre-trained word embeddings
    with torch.no_grad():
        model.embed.weight.data.copy_(torch.from_numpy(vectors))
        print("Embeddings fixed: {}".format(cfg["fix_emb"]))
        model.embed.weight.requires_grad = not cfg["fix_emb"]

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=cfg["lr"],
                     weight_decay=cfg["weight_decay"])

    # set learning rate scheduler
    if cfg["scheduler"] == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg["lr_decay"],
            patience=cfg["patience"],
            threshold=cfg["threshold"], threshold_mode='rel',
            cooldown=cfg["cooldown"], verbose=True, min_lr=cfg["min_lr"])
    elif cfg["scheduler"] == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=cfg["lr_decay"])
    elif cfg["scheduler"] == "multistep":
        milestones = cfg["milestones"]
        print("milestones (epoch):", milestones)
        scheduler = MultiStepLR(
            optimizer, milestones=milestones, gamma=cfg["lr_decay"])
    else:
        raise ValueError("Unknown scheduler")

    # print model and parameters
    print(model)
    print_parameters(model)

    writer = SummaryWriter(log_dir=cfg["save_path"])  # TensorBoard
    start = time.time()
    iter_i = 0
    epoch = 0
    best_eval = 1e12
    best_iter = 0
    pad_idx = vocab.w2i[PAD_TOKEN]

    # resume from a checkpoint
    if cfg.get("ckpt", ""):
        print("Resuming from ckpt: {}".format(cfg["ckpt"]))
        ckpt = torch.load(cfg["ckpt"])
        model.load_state_dict(ckpt["state_dict"])
        best_iter = ckpt["best_iter"]
        best_eval = ckpt["best_eval"]
        iter_i = ckpt["best_iter"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        cur_lr = scheduler.optimizer.param_groups[0]["lr"]
        print("# lr = ", cur_lr)

    # main training loop
    while True:  # when we run out of examples, shuffle and continue
        for batch in get_minibatch(train_data, batch_size=batch_size,
                                   shuffle=True):

            # forward pass
            model.train()
            x, targets, _ = prepare_minibatch(batch, model.vocab, device=device)

            output = model(x)

            mask = (x != pad_idx)
            assert pad_idx == 1, "pad idx"
            loss, loss_optional = model.get_loss(output, targets, mask=mask)

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=cfg["max_grad_norm"])
            optimizer.step()
            iter_i += 1

            # print info
            if iter_i % print_every == 0:

                # print main loss, lr, and optional stuff defined by the model
                writer.add_scalar('train/loss', loss.item(), iter_i)
                cur_lr = scheduler.optimizer.param_groups[0]["lr"]
                writer.add_scalar('train/lr', cur_lr, iter_i)

                for k, v in loss_optional.items():
                    writer.add_scalar('train/%s' % k, v, iter_i)

                # print info to console
                loss_str = "%.4f" % loss.item()
                opt_str = make_kv_string(loss_optional)
                seconds_since_start = time.time() - start
                hours = seconds_since_start / 60 // 60
                minutes = seconds_since_start % 3600 // 60
                seconds = seconds_since_start % 60
                print("Epoch %03d Iter %08d time %02d:%02d:%02d loss %s %s" %
                      (epoch, iter_i, hours, minutes, seconds,
                       loss_str, opt_str))

            # take epoch step (if using MultiStepLR scheduler)
            if iter_i % iters_per_epoch == 0:

                cur_lr = scheduler.optimizer.param_groups[0]["lr"]
                if cur_lr > cfg["min_lr"]:
                    if isinstance(scheduler, MultiStepLR):
                        scheduler.step()
                    elif isinstance(scheduler, ExponentialLR):
                        scheduler.step()

                cur_lr = scheduler.optimizer.param_groups[0]["lr"]
                print("#lr", cur_lr)
                scheduler.optimizer.param_groups[0]["lr"] = max(cfg["min_lr"],
                                                                cur_lr)

            # evaluate
            if iter_i % eval_every == 0:

                print("Evaluation starts - %s" % str(datetime.datetime.now()))

                # print a few examples
                examples = get_examples(model, dev_data, num_examples=num_examples,
                                        device=device)
                for i, example in enumerate(examples, 1):
                    print("Example %d:" % i, " ".join(example))
                    writer.add_text(
                        "examples/example_%d" % i, " ".join(example), iter_i)

                model.eval()

                print("Evaluating..", str(datetime.datetime.now()))

                dev_eval = evaluate_loss(
                    model, dev_data, batch_size=eval_batch_size,
                    device=device, cfg=cfg)

                for k, v in dev_eval.items():
                    writer.add_scalar('dev/' + k, v, iter_i)

                # save best model parameters (lower is better)
                compare_obj = dev_eval["obj"] if "obj" in dev_eval \
                    else dev_eval["loss"]
                dynamic_threshold = best_eval * (1 - cfg["threshold"])
                # only update after first 5 epochs (for stability)
                if compare_obj < dynamic_threshold \
                        and iter_i > 5 * iters_per_epoch:
                    print("new highscore", compare_obj)
                    best_eval = compare_obj
                    best_iter = iter_i
                    if not os.path.exists(cfg["save_path"]):
                        os.makedirs(cfg["save_path"])

                    for k, v in dev_eval.items():
                        writer.add_scalar('best/dev/' + k, v, iter_i)

                    ckpt = {
                        "state_dict": model.state_dict(),
                        "cfg": cfg,
                        "best_eval": best_eval,
                        "best_iter": best_iter,
                        "optimizer_state_dict": optimizer.state_dict()
                    }

                    path = os.path.join(cfg["save_path"], "model.pt")
                    torch.save(ckpt, path)

                # update lr scheduler
                if isinstance(scheduler, ReduceLROnPlateau):
                    if iter_i > 5 * iters_per_epoch:
                        scheduler.step(compare_obj)

            # done training
            cur_lr = scheduler.optimizer.param_groups[0]["lr"]

            # if iter_i == num_iterations or cur_lr < stop_lr:
            if iter_i == num_iterations:
                print("Done training")
                print("Last lr: ", cur_lr)

                # evaluate on test with best model
                print("Loading best model")
                path = os.path.join(cfg["save_path"], "model.pt")
                ckpt = torch.load(path)
                model.load_state_dict(ckpt["state_dict"])

                print("Evaluating")
                dev_eval = evaluate_loss(
                    model, dev_data, batch_size=eval_batch_size,
                    device=device, cfg=cfg)

                dev_s = make_kv_string(dev_eval)

                print("best model iter {:d} dev {}".format(
                    best_iter, dev_s))

                # save result
                result_path = os.path.join(cfg["save_path"], "results.json")

                cfg["best_iter"] = best_iter

                for name, eval_result in zip(("dev",),
                                             (dev_eval)):
                    for k, v in eval_result.items():
                        cfg[name + '_' + k] = v

                with open(result_path, mode="w") as f:
                    json.dump(cfg, f)

                # close Summary Writer
                writer.close()
                return

        epoch += 1


if __name__ == "__main__":
    train()
