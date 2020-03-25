import numpy as np

from collections import namedtuple

from sacremoses import MosesTokenizer


QualityExample = namedtuple("QualityExample", ["tokens", "score"])

tokenizer = MosesTokenizer()


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
        score = float(parts[6])
        scores.append(score)
    print(np.min(scores))
    print(np.max(scores))
    print(np.mean(scores))
    print(np.median(scores))
    print(np.std(scores))
    return min(scores), max(scores)


def qe_reader(path, max_len=0):
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
        score = min_max(float(parts[6]), min_scores, max_scores)
        tokens = preprocess(parts[2])
        if max_len > 0:
            tokens = tokens[:max_len]
        yield QualityExample(tokens=tokens, score=score)


def qe_annotations_reader(path, max_len=0):
    raise NotImplemented
