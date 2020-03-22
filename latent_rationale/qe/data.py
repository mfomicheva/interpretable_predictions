from collections import namedtuple


QualityExample = namedtuple("QualityExample", ["tokens", "score"])


def qe_reader(path, max_len=0):
    """
    Reads in QE WMT2020 data
    :param path:
    :return: QualityExample
    """
    for line in open(path):
        parts = line.split('\t')
        score = float(parts[6])
        tokens = parts[2].split()
        if max_len > 0:
            tokens = tokens[:max_len]
        yield QualityExample(tokens=tokens, score=score)


def qe_annotations_reader(path, max_len=0):
    raise NotImplemented
