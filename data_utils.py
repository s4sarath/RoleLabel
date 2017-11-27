from tensorflow.python.platform import gfile
import re
title = []
story = []
dic = []
_PAD = "_PAD"

_UNK = "_UNK"
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1

def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)