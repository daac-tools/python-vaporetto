import vaporetto
import Mykytea
from sudachipy import tokenizer as sudachi_tokenizer
from sudachipy import dictionary as sudachi_dictionary

from tests import dataset


def kytea_count_tokens(tokenizer, corpus):
    cnt = 0
    for line in corpus:
        cnt += len(tokenizer.getWS(line))
    return cnt


def vaporetto_count_tokens(tokenizer, corpus):
    cnt = 0
    for line in corpus:
        cnt += len(tokenizer.tokenize(line))
    return cnt


def sudachi_count_tokens(tokenizer, mode, corpus):
    cnt = 0
    for line in corpus:
        cnt += len(tokenizer.tokenize(line, mode))
    return cnt


def test_kytea_count_tokens_bench(benchmark):
    model_path = dataset.get_kytea_model_path()
    corpus = dataset.load_wagahaiwa_nekodearu()
    tokenizer = Mykytea.Mykytea(f'-model {model_path} -notags')
    benchmark(kytea_count_tokens, tokenizer, corpus)


def test_vaporetto_count_tokens_bench(benchmark):
    model_path = dataset.get_kytea_model_path()
    corpus = dataset.load_wagahaiwa_nekodearu()
    with open(model_path, 'rb') as fp:
        model_data = fp.read()
    tokenizer = vaporetto.Vaporetto.create_from_kytea_model(model_data)
    benchmark(vaporetto_count_tokens, tokenizer, corpus)


def test_sudachi_count_tokens_bench(benchmark):
    corpus = dataset.load_wagahaiwa_nekodearu()
    tokenizer = sudachi_dictionary.Dictionary().create()
    mode = sudachi_tokenizer.Tokenizer.SplitMode.A
    benchmark(sudachi_count_tokens, tokenizer, mode, corpus)
