Example usage
=============

python-vaporetto does not contain model files. To perform tokenization, follow `the document of
Vaporetto <https://github.com/daac-tools/vaporetto>`_ to download distribution models or train
your own models beforehand.

Tokenize with Vaporetto model
-----------------------------

The following example tokenizes a string using a Vaporetto model.

.. code-block:: python

   >>> import vaporetto
   >>> with open('path/to/model.zst', 'rb') as fp:
   >>>     model = fp.read()

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


Tokenize with KyTea model
-------------------------

If you want to use a KyTea model, use ``create_from_kytea_model()`` instead.

.. code-block:: python

    >>> with open('path/to/jp-0.4.7-5.mod', 'rb') as fp:
    >>>     model = fp.read()

    >>> tokenizer = vaporetto.Vaporetto.create_from_kytea_model(model)
