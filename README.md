# 🐍 python-vaporetto 🛥

[Vaporetto](https://github.com/daac-tools/vaporetto) is a fast and lightweight pointwise prediction based tokenizer.
This is a Python wrapper for Vaporetto.

[![PyPI](https://img.shields.io/pypi/v/vaporetto)](https://pypi.org/project/vaporetto/)
[![Build Status](https://github.com/daac-tools/python-vaporetto/actions/workflows/CI.yml/badge.svg)](https://github.com/daac-tools/python-vaporetto/actions)
[![Documentation Status](https://readthedocs.org/projects/python-vaporetto/badge/?version=latest)](https://python-vaporetto.readthedocs.io/en/latest/?badge=latest)

## Installation

### Install pre-built package from PyPI

Run the following command:

```
$ pip install vaporetto
```

### Build from source

You need to install the Rust compiler following [the documentation](https://www.rust-lang.org/tools/install) beforehand.
vaporetto uses `pyproject.toml`, so you also need to upgrade pip to version 19 or later.

```
$ pip install --upgrade pip
```

After setting up the environment, you can install vaporetto as follows:

```
$ pip install git+https://github.com/daac-tools/python-vaporetto
```

## Example Usage

python-vaporetto does not contain model files.
To perform tokenization, follow [the document of Vaporetto](https://github.com/daac-tools/vaporetto) to download distribution models or train your own models beforehand.

Check the version number as shown below to use compatible models:

```python
>>> import vaporetto
>>> vaporetto.VAPORETTO_VERSION
'0.6.5'

```

Examples:

```python
# Import vaporetto module
>>> import vaporetto

# Load the model file
>>> with open('tests/data/vaporetto.model', 'rb') as fp:
...     model = fp.read()

# Create an instance of the Vaporetto
>>> tokenizer = vaporetto.Vaporetto(model, predict_tags = True)

# Tokenize
>>> tokenizer.tokenize_to_string('まぁ社長は火星猫だ')
'まぁ/名詞/マー 社長/名詞/シャチョー は/助詞/ワ 火星/名詞/カセー 猫/名詞/ネコ だ/助動詞/ダ'

>>> tokens = tokenizer.tokenize('まぁ社長は火星猫だ')

>>> len(tokens)
6

>>> tokens[0].surface()
'まぁ'

>>> tokens[0].tag(0)
'名詞'

>>> tokens[0].tag(1)
'マー'

>>> [token.surface() for token in tokens]
['まぁ', '社長', 'は', '火星', '猫', 'だ']

```

## Note for distributed models

The distributed models are compressed in zstd format. If you want to load these compressed models,
you must decompress them outside the API.

```python
>>> import vaporetto
>>> import zstandard  # zstandard package in PyPI

>>> dctx = zstandard.ZstdDecompressor()
>>> with open('tests/data/vaporetto.model.zst', 'rb') as fp:
...    with dctx.stream_reader(fp) as dict_reader:
...        tokenizer = vaporetto.Vaporetto(dict_reader.read(), predict_tags = True)

```

## Note for KyTea's models

You can also use KyTea's models as follows:

```python
>>> with open('path/to/jp-0.4.7-5.mod', 'rb') as fp:  # doctest: +SKIP
...     tokenizer = vaporetto.Vaporetto.create_from_kytea_model(fp.read())

```

Note: Vaporetto does not support tag prediction with KyTea's models.

## [Speed Comparison](https://github.com/daac-tools/python-vaporetto/wiki/Speed-Comparison)

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

See [the guidelines](./CONTRIBUTING.md).
