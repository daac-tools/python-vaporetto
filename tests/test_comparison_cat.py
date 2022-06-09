import vaporetto
import Mykytea
from sudachipy import tokenizer as sudachi_tokenizer
from sudachipy import dictionary as sudachi_dictionary

from tests import dataset


def kytea_cat_surfaces(tokenizer, corpus):
    cnt = 0
    for line in corpus:
        cnt += len(' '.join(tokenizer.getWS(line)))
    return cnt


def vaporetto_cat_surfaces(tokenizer, corpus):
    cnt = 0
    for line in corpus:
        cnt += len(' '.join(t.surface() for t in tokenizer.tokenize(line)))
    return cnt


def sudachi_cat_surfaces(tokenizer, mode, corpus):
    cnt = 0
    for line in corpus:
        cnt += len(' '.join(t.surface() for t in tokenizer.tokenize(line, mode)))
    return cnt


def test_kytea_cat_surfaces_bench(benchmark):
    model_path = dataset.get_kytea_model_path()
    corpus = dataset.load_wagahaiwa_nekodearu()
    tokenizer = Mykytea.Mykytea(f'-model {model_path} -notags')
    benchmark(kytea_cat_surfaces, tokenizer, corpus)


def test_vaporetto_cat_surfaces_bench(benchmark):
    model_path = dataset.get_kytea_model_path()
    corpus = dataset.load_wagahaiwa_nekodearu()
    with open(model_path, 'rb') as fp:
        model_data = fp.read()
    tokenizer = vaporetto.Vaporetto(model_data)
    benchmark(vaporetto_cat_surfaces, tokenizer, corpus)


def test_sudachi_cat_surfaces_bench(benchmark):
    corpus = dataset.load_wagahaiwa_nekodearu()
    tokenizer = sudachi_dictionary.Dictionary().create()
    mode = sudachi_tokenizer.Tokenizer.SplitMode.A
    benchmark(sudachi_cat_surfaces, tokenizer, mode, corpus)
