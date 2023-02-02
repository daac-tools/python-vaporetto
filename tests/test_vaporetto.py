from __future__ import annotations

import pathlib

import vaporetto

MODEL_PATH = pathlib.PurePath(__file__).parent / 'data/model.zst'


def test_tokenlist_empty() -> None:
    with open(MODEL_PATH, 'rb') as fp:
        tokenizer = vaporetto.Vaporetto(fp.read())
    tokens = tokenizer.tokenize('')

    assert [] == list(tokens)


def test_tokenlist_index() -> None:
    with open(MODEL_PATH, 'rb') as fp:
        tokenizer = vaporetto.Vaporetto(fp.read())
    tokens = tokenizer.tokenize('まぁ社長は火星猫だ')

    assert 'まぁ' == tokens[0].surface()
    assert '社長' == tokens[1].surface()
    assert 'は' == tokens[2].surface()
    assert '火星' == tokens[3].surface()
    assert '猫' == tokens[4].surface()
    assert 'だ' == tokens[5].surface()


def test_tokenlist_iter() -> None:
    with open(MODEL_PATH, 'rb') as fp:
        tokenizer = vaporetto.Vaporetto(fp.read())
    tokens = tokenizer.tokenize('まぁ社長は火星猫だ')

    assert ['まぁ', '社長', 'は', '火星', '猫', 'だ'] == list(
        token.surface() for token in tokens
    )


def test_tokenlist_iter_positions() -> None:
    with open(MODEL_PATH, 'rb') as fp:
        tokenizer = vaporetto.Vaporetto(fp.read())
    tokens = tokenizer.tokenize('まぁ社長は火星猫だ')

    assert [(0, 2), (2, 4), (4, 5), (5, 7), (7, 8), (8, 9)] == list(
        (token.start(), token.end()) for token in tokens
    )


def test_wsconst() -> None:
    with open(MODEL_PATH, 'rb') as fp:
        tokenizer = vaporetto.Vaporetto(fp.read(), wsconst='K')
    tokens = tokenizer.tokenize('まぁ社長は火星猫だ')

    assert ['まぁ', '社長', 'は', '火星猫', 'だ'] == list(token.surface() for token in tokens)


def test_tags_1() -> None:
    with open(MODEL_PATH, 'rb') as fp:
        tokenizer = vaporetto.Vaporetto(fp.read(), predict_tags=True)
    tokens = tokenizer.tokenize('まぁ社長は火星猫だ')

    assert ['名詞', '名詞', '助詞', '名詞', '名詞', '助動詞'] == list(
        token.tag(0) for token in tokens
    )


def test_tags_2() -> None:
    with open(MODEL_PATH, 'rb') as fp:
        tokenizer = vaporetto.Vaporetto(fp.read(), predict_tags=True)
    tokens = tokenizer.tokenize('まぁ社長は火星猫だ')

    assert ['マー', 'シャチョー', 'ワ', 'カセー', 'ネコ', 'ダ'] == list(
        token.tag(1) for token in tokens
    )


def test_tokenize_to_string_empty() -> None:
    with open(MODEL_PATH, 'rb') as fp:
        tokenizer = vaporetto.Vaporetto(fp.read(), predict_tags=True)
    assert '' == tokenizer.tokenize_to_string('')


def test_tokenize_to_string() -> None:
    with open(MODEL_PATH, 'rb') as fp:
        tokenizer = vaporetto.Vaporetto(fp.read(), predict_tags=True)
    assert (
        'まぁ/名詞/マー 社長/名詞/シャチョー は/助詞/ワ 火星/名詞/カセー 猫/名詞/ネコ だ/助動詞/ダ'
        == tokenizer.tokenize_to_string('まぁ社長は火星猫だ')
    )
