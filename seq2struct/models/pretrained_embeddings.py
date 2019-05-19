import abc
import functools
import os
import time

import bpemb
import corenlp
import requests
import torch
import torchtext
from corenlp.client import PermanentlyFailedException
from requests.exceptions import ConnectionError

from seq2struct.utils import registry


class Embedder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def tokenize(self, sentence):
        '''Given a string, return a list of tokens suitable for lookup.'''
        pass

    @abc.abstractmethod
    def untokenize(self, tokens):
        '''Undo tokenize.'''
        pass

    @abc.abstractmethod
    def lookup(self, token):
        '''Given a token, return a vector embedding if token is in vocabulary.

        If token is not in the vocabulary, then return None.'''
        pass

    @abc.abstractmethod
    def contains(self, token):
        pass

    @abc.abstractmethod
    def to(self, device):
        '''Transfer the pretrained embeddings to the given device.'''
        pass


@registry.register('word_emb', 'glove')
class GloVe(Embedder):

    def __init__(self, kind):
        cache = os.path.join(os.environ.get('PT_DATA_DIR', os.getcwd()), '.vector_cache')
        self.glove = torchtext.vocab.GloVe(name=kind, cache=cache)
        self._corenlp_client = None
        self.dim = self.glove.dim
        self.vectors = self.glove.vectors

    @property
    def corenlp_client(self):
        if self._corenlp_client is None:
            if not os.environ.get('CORENLP_HOME'):
                os.environ['CORENLP_HOME'] = os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        '../../third_party/stanford-corenlp-full-2018-10-05'))
            if not os.path.exists(os.environ['CORENLP_HOME']):
                raise Exception(
                    '''Please install Stanford CoreNLP and put it at {}.

                    Direct URL: http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
                    Landing page: https://stanfordnlp.github.io/CoreNLP/'''.format(os.environ['CORENLP_HOME']))
            self._corenlp_client = corenlp.CoreNLPClient(
                annotators="tokenize ssplit")
        return self._corenlp_client

    @functools.lru_cache(maxsize=1024)
    def tokenize(self, text):
        try:
            ann = self.corenlp_client.annotate(text)
        except (PermanentlyFailedException, ConnectionError):
            print('\nWARNING: CoreNLP connection timeout. Recreating the server...')
            self.corenlp_client.stop()
            time.sleep(1)
            self.corenlp_client.start()
            ann = self.corenlp_client.annotate(text)
        return [tok.word.lower() for sent in ann.sentence for tok in sent.token]

    def untokenize(self, tokens):
        return ' '.join(tokens)

    def lookup(self, token):
        i = self.glove.stoi.get(token)
        if i is None:
            return None
        return self.vectors[i]

    def contains(self, token):
        return token in self.glove.stoi

    def to(self, device):
        self.vectors = self.vectors.to(device)

    def __del__(self):
        self.corenlp_client.stop()


@registry.register('word_emb', 'bpemb')
class BPEmb(Embedder):
    def __init__(self, dim, vocab_size, lang='en'):
        self.bpemb = bpemb.BPEmb(lang=lang, dim=dim, vs=vocab_size)
        self.dim = dim
        self.vectors = torch.from_numpy(self.bpemb.vectors)

    def tokenize(self, text):
        return self.bpemb.encode(text)

    def untokenize(self, tokens):
        return self.bpemb.decode(tokens)

    def lookup(self, token):
        i = self.bpemb.spm.PieceToId(token)
        if i == self.bpemb.spm.unk_id():
            return None
        return self.vectors[i]

    def contains(self, token):
        return self.lookup(token) is not None

    def to(self, device):
        self.vectors = self.vectors.to(device)
