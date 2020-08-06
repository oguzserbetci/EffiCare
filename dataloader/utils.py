import logging
import os
import re
from glob import glob

from torch import nn
import numpy as np
from torchtext.vocab import Vectors, Vocab


def feature_string(key):
    # # is used for bin index representation
    # = is used for compound representation
    # : is used as a seperator
    # , unnecessary
    # parantheses unnecessary
    key = re.sub(r'[-\s,\(\)]+', ' ', key)
    key = key.strip().replace(' ', '-')
    return key


def compound_repr(feature, val):
    return (feature + '_' + str(val)).lower()


def get_counts(vocab_file='embeddings/sentences.mimic3.txt.counts'):
    from collections import Counter
    with open(f'{vocab_file}', 'r') as file:
        counter = Counter()
        for line in file:
            word, count = line.strip().split(' ')
            counter.update({word: int(count)})
    return counter


def get_vocab(emb_prefix='embeddings/sentences.mimic3.txt',
              emb_dim=100,
              emb_suffix='.Fasttext.15ws.10neg',
              rand_emb=False,
              min_word_count=10,
              vocab_file='embeddings/sentences.mimic3.txt.counts',
              **kwargs):
    '''returns: torchtext Vocab'''
    if rand_emb:
        logging.info("RANDOM EMBEDDINGS")
        pretrained_embeddings = None
    else:
        emb_paths = glob(f"{emb_prefix}.{emb_dim}d{emb_suffix}.vec")
        emb_paths.sort(key=os.path.getmtime)
        if len(emb_paths) == 0:
            logging.error(f"glob for {emb_prefix}.{emb_dim}d{emb_suffix}.vec is empty")
            raise FileNotFoundError
        emb_path = emb_paths[0]

        pretrained_embeddings = Vectors(emb_path)

    counter = get_counts(vocab_file)
    vocab = Vocab(counter, specials_first=True,
                  vectors=pretrained_embeddings,
                  min_freq=min_word_count,
                  unk_init=nn.init.normal_ if not rand_emb else None,
                  specials=['<pad>', '<unk>'])

    if not rand_emb:
        # initialize unk embedding to normal distribution
        vocab.vectors[vocab.stoi.default_factory()].data.normal_()

    return vocab


def build_vocab(vocab_file='embeddings/sentences.mimic3.txt.counts', min_word_count=10,
                **vocab_kwargs):
    '''returns torchtext Vocab
    '''
    return Vocab(get_counts(vocab_file), specials=['<pad>', '<unk>'], min_freq=min_word_count)


class Event:
    def __init__(self, table_source, label, value=None, time=None, **kwargs):
        self.time = time

        self.label = feature_string(label)

        if value is not None:
            self.has_value = True
            self.text = self._set_value(self.label, value, **kwargs)
        else:
            self.has_value = False
            self.text = self.label

    def _set_value(self, label, value, **kwargs):
        token = label
        try:
            self.value = float(str(value).strip())
            self.is_scalar = True
        except ValueError:
            token = compound_repr(label, feature_string(value))
            self.value = feature_string(value)
            self.is_scalar = False
        return token

    def __repr__(self):
        return self.text


class BinnedEvent(Event):
    def __init__(self, table_source, label, value=None, time=None, **kwargs):
        self.bin_ix = 0
        self.table_source = table_source
        super().__init__(table_source, label, value, time)

    def _set_value(self, label, value, **kwargs):
        token = label
        try:
            value = float(str(value).strip())
            self.value = value
            # bin_ix = 0 is padding
            self.bin_ix = np.digitize(value, self.table_source.bins[label]) + 1
            self.is_scalar = True
        except KeyError:
            self.is_scalar = True
            pass
        except ValueError:
            token = compound_repr(label, feature_string(value))
            self.is_scalar = False

        return token

    def __repr__(self):
        return self.text
