import ast
import collections
import itertools
import re

import attr
import nltk
import torch.utils.data

from seq2struct import ast_util
from seq2struct.utils import registry
from seq2struct.utils import indexed_file

LINE_SEP = '§'

@attr.s
class HearthstoneItem:
    text = attr.ib()
    code = attr.ib()


@registry.register('dataset', 'hearthstone')
class HearthstoneDataset(torch.utils.data.Dataset): 
    def __init__(self, filename, limit=None):
        self.filename = filename
        self.examples = []
        for example in itertools.islice(
                zip(
                    open(self.filename + '.in'),
                    open(self.filename + '.out')),
                limit):
            processed = self._process(example)
            if processed is not None:
                self.examples.append(processed)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def _process(self, example):
        text, code = example

        code = '\n'.join(code.split(LINE_SEP))

        # TODO move to NL2CodeDecoderPreproc.validate_item
        try:
            py_ast = ast.parse(code)
        except SyntaxError:
            return None
        node = ast_util.convert_native_ast(py_ast)

        #gold_source = astor.to_source(gold_ast_tree)
        #ast_tree = parse_tree_to_python_ast(parse_tree)
        #pred_source = astor.to_source(ast_tree)

        #parse_tree = parse_raw(code)
        #gold_ast_tree = ast.parse(code).body[0]
        #gold_source = astor.to_source(gold_ast_tree)
        #ast_tree = parse_tree_to_python_ast(parse_tree)
        #pred_source = astor.to_source(ast_tree)

        #assert gold_source == pred_source, 'sanity check fails: gold=[%s], actual=[%s]' % (gold_source, pred_source)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        orig_tokens = nltk.word_tokenize(text)
        tokens = ['<name>']
        for token in orig_tokens:
            if   token == 'NAME_END':
                tokens += ['</name>', '<atk>']
            elif token == 'ATK_END':
                tokens += ['</atk>', '<def>']
            elif token == 'DEF_END':
                tokens += ['</def>', '<cost>']
            elif token == 'COST_END':
                tokens += ['</cost>', '<dur>']
            elif token == 'DUR_END':
                tokens += ['</dur>', '<type>']
            elif token == 'TYPE_END':
                tokens += ['</type>', '<player-cls>']
            elif token == 'PLAYER_CLS_END':
                tokens += ['</player-cls>', '<race>']
            elif token == 'RACE_END':
                tokens += ['</race>', '<rarity>']
            elif token == 'RARITY_END':
                tokens += ['</rarity>']
            else:
                tokens += [token]
        
        return HearthstoneItem(text=tokens, code=code)
        
    @attr.s
    class Metrics:
        exact_match = attr.ib(factory=list)
        sentence_bleu_scores = attr.ib(factory=list)
        gold_codes = attr.ib(factory=list)
        inferred_codes = attr.ib(factory=list)

        smoothing_function = nltk.translate.bleu_score.SmoothingFunction().method3

        def add(self, gold_code, inferred_code):
            # Both of these should be canonicalized
            exact_match = gold_code == inferred_code

            gold_tokens_for_bleu = tokenize_for_bleu_eval(gold_code)
            inferred_tokens_for_bleu = tokenize_for_bleu_eval(inferred_code)

            ngram_weights = [0.25] * min(4, len(inferred_tokens_for_bleu))
            bleu_score = nltk.translate.bleu_score.sentence_bleu(
                    (gold_tokens_for_bleu,),
                    inferred_tokens_for_bleu,
                    weights=ngram_weights,
                    smoothing_function=self.smoothing_function)

            self.exact_match.append(exact_match)
            self.sentence_bleu_scores.append(bleu_score)

            self.gold_codes.append((gold_tokens_for_bleu,))
            self.inferred_codes.append(inferred_tokens_for_bleu)

        def finalize(self):
            return collections.OrderedDict((
                ('exact match', sum(self.exact_match) / len(self.exact_match)),
                ('sentence BLEU', sum(self.sentence_bleu_scores) / len(self.sentence_bleu_scores)),
                ('corpus BLEU', nltk.translate.bleu_score.corpus_bleu(
                    self.gold_codes,
                    self.inferred_codes,
                    smoothing_function=self.smoothing_function)),
            ))


# From NL2Code/evaluation.py
def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]

    return tokens
