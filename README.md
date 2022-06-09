# 🐍 python-vaporetto 🛥

[Vaporetto](https://github.com/daac-tools/vaporetto) is a fast and lightweight pointwise prediction based tokenizer.
This is a Python wrapper for Vaporetto.

## Installation

To use Vaporetto, run the following command:

```
$ pip install vaporetto
```

Or you can also build from the source:

```
$ python -m venv .env
$ source .env/bin/activate
$ pip install maturin
$ maturin build
```

## Example Usage

```python
# Import vaporetto module
import vaporetto

# Load the model file
with open('path/to/model.zst', 'rb') as fp:
    model = fp.read()

# Create an instance of the Vaporetto
tokenizer = vaporetto.Vaporetto(model, predict_tags = True)

# Tokenize
tokenizer.tokenize_to_string('まぁ社長は火星猫だ')
#=> 'まぁ/名詞/マー 社長/名詞/シャチョー は/助詞/ワ 火星/名詞/カセー 猫/名詞/ネコ だ/助動詞/ダ'

tokens = tokenizer.tokenize('まぁ社長は火星猫だ')
len(tokens)
#=> 6
tokens[0].surface()
#=> 'まぁ'
tokens[0].tag(0)
#=> '名詞'
tokens[0].tag(1)
#=> 'マー'
[token.surface() for token in tokens]
#=> ['まぁ', '社長', 'は', '火星', '猫', 'だ']]
```

## Documentation

Use the help function to show the API reference.

```python
import vaporetto
help(vaporetto)
```

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Disclaimer

This software is developed by LegalForce, Inc.,
but not an officially supported LegalForce product.
