from collections import namedtuple

from sacremoses import MosesTokenizer


QualityExample = namedtuple("QualityExample", ["tokens", "score"])

tokenizer = MosesTokenizer()


def preprocess(line):
    return tokenizer.tokenize(line.strip().lower())


def qe_reader(path, max_len=0):
    """
    Reads in QE WMT2020 data
    :param path:
    :return: QualityExample
    """
    skip = True
    for line in open(path):
        if skip is True:
            skip = False
            continue
        parts = line.split('\t')
        score = float(parts[6])
        tokens = preprocess(parts[2])
        if max_len > 0:
            tokens = tokens[:max_len]
        yield QualityExample(tokens=tokens, score=score)


def qe_annotations_reader(path, max_len=0):
    raise NotImplemented
