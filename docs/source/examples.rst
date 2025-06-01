Example usage
=============

python-vaporetto does not contain model files. To perform tokenization, follow `the document of
Vaporetto <https://github.com/daac-tools/vaporetto>`_ to download distribution models or train
your own models beforehand.

You can check the version number as shown below to use compatible models:

.. code-block:: python

   >>> import vaporetto
   >>> vaporetto.VAPORETTO_VERSION
   '0.6.5'

Tokenize with Vaporetto model
-----------------------------

The following example tokenizes a string using a Vaporetto model.

.. code-block:: python

   >>> import vaporetto
   >>> with open('tests/data/vaporetto.model', 'rb') as fp:
   ...     model = fp.read()

   >>> tokenizer = vaporetto.Vaporetto(model, predict_tags = True)

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

The distributed models are compressed in zstd format. If you want to load these compressed models,
you must decompress them outside the API:

.. code-block:: python

   >>> import vaporetto
   >>> import zstandard  # zstandard package in PyPI

   >>> dctx = zstandard.ZstdDecompressor()
   >>> with open('tests/data/vaporetto.model.zst', 'rb') as fp:
   ...     with dctx.stream_reader(fp) as dict_reader:
   ...         tokenizer = vaporetto.Vaporetto(dict_reader.read(), predict_tags = True)

Tokenize with KyTea model
-------------------------

If you want to use a KyTea model, use ``create_from_kytea_model()`` instead.

.. code-block:: python

    >>> import vaporetto
    >>> with open('path/to/jp-0.4.7-5.mod', 'rb') as fp:  # doctest: +SKIP
    ...     tokenizer = vaporetto.Vaporetto.create_from_kytea_model(fp.read())
