import numpy as np

from collections import namedtuple

from sacremoses import MosesTokenizer


QualityExample = namedtuple("QualityExample", ["tokens", "scores"])

tokenizer = MosesTokenizer()

SCORE_IDX = 4
SRC_IDX = 1
TGT_IDX = 2


def preprocess(line):
    return tokenizer.tokenize(line.strip().lower())


def min_max(score, min_scores, max_scores):
    return (score - min_scores) / (max_scores - min_scores)


def read_scores(path):
    skip = True
    scores = []
    for line in open(path):
        if skip is True:
            skip = False
            continue
        parts = line.split('\t')
        score = float(parts[SCORE_IDX])
        scores.append(score)
    print_target_distribution(scores)
    return min(scores), max(scores)


def print_target_distribution(scores):
    print(np.min(scores))
    print(np.max(scores))
    print(np.mean(scores))
    print(np.median(scores))
    print(np.std(scores))


def qe_reader(path, max_len=0, simulated=False):
    """
    Reads in QE WMT2020 data
    :param path:
    :return: QualityExample
    """
    min_scores, max_scores = read_scores(path)
    skip = True
    for line in open(path):
        if skip is True:
            skip = False
            continue
        parts = line.split('\t')
        score = min_max(float(parts[SCORE_IDX]), min_scores, max_scores)
        tokens = preprocess(parts[TGT_IDX])
        if simulated:
            if score < 0.3:
                tokens = ["doomsday"]
                score = 0.
            elif simulated and score > 0.7:
                tokens = ["perfection"]
                score = 0.5
            else:
                tokens = ["medium"]
                score = 1.
        if max_len > 0:
            tokens = tokens[:max_len]
        yield QualityExample(tokens=tokens, scores=[score])


def qe_annotations_reader(path, max_len=0):
    raise NotImplemented
