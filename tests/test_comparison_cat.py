from __future__ import annotations

import vaporetto
import Mykytea  # type: ignore[import]
from sudachipy import tokenizer as sudachi_tokenizer  # type: ignore[import]
from sudachipy import dictionary as sudachi_dictionary

from pytest_benchmark import fixture  # type: ignore[import]
from tests import dataset


def kytea_cat_surfaces(tokenizer: Mykytea.Mykytea, corpus: list[str]) -> int:
    cnt = 0
    for line in corpus:
        cnt += len(' '.join(tokenizer.getWS(line)))
    return cnt


def vaporetto_cat_surfaces(tokenizer: vaporetto.Vaporetto, corpus: list[str]) -> int:
    cnt = 0
    for line in corpus:
        cnt += len(' '.join(t.surface() for t in tokenizer.tokenize(line)))
    return cnt


def sudachi_cat_surfaces(
    tokenizer: sudachi_tokenizer.Tokenizer,
    mode: sudachi_tokenizer.Tokenizer.SplitMode,
    corpus: list[str],
) -> int:
    cnt = 0
    for line in corpus:
        cnt += len(' '.join(t.surface() for t in tokenizer.tokenize(line, mode)))
    return cnt


def test_kytea_cat_surfaces_bench(benchmark: fixture.BenchmarkFixture) -> None:
    model_path = dataset.get_kytea_model_path()
    corpus = dataset.load_wagahaiwa_nekodearu()
    tokenizer = Mykytea.Mykytea(f'-model {model_path} -notags')
    benchmark(kytea_cat_surfaces, tokenizer, corpus)


def test_vaporetto_cat_surfaces_bench(benchmark: fixture.BenchmarkFixture) -> None:
    model_path = dataset.get_kytea_model_path()
    corpus = dataset.load_wagahaiwa_nekodearu()
    with open(model_path, 'rb') as fp:
        model_data = fp.read()
    tokenizer = vaporetto.Vaporetto.create_from_kytea_model(model_data)
    benchmark(vaporetto_cat_surfaces, tokenizer, corpus)


def test_sudachi_a_cat_surfaces_bench(benchmark: fixture.BenchmarkFixture) -> None:
    corpus = dataset.load_wagahaiwa_nekodearu()
    tokenizer = sudachi_dictionary.Dictionary().create()
    mode = sudachi_tokenizer.Tokenizer.SplitMode.A
    benchmark(sudachi_cat_surfaces, tokenizer, mode, corpus)


def test_sudachi_c_cat_surfaces_bench(benchmark: fixture.BenchmarkFixture) -> None:
    corpus = dataset.load_wagahaiwa_nekodearu()
    tokenizer = sudachi_dictionary.Dictionary().create()
    mode = sudachi_tokenizer.Tokenizer.SplitMode.C
    benchmark(sudachi_cat_surfaces, tokenizer, mode, corpus)
