API reference
=============

.. autoclass:: vaporetto.Vaporetto
   :members:

.. autoclass:: vaporetto.TokenList
   :special-members: __getitem__, __iter__, __len__

.. autoclass:: vaporetto.TokenIterator
   :special-members: __next__

.. autoclass:: vaporetto.Token
   :members:

.. data:: VAPORETTO_VERSION
   :type: str
   :canonical: vaporetto.VAPORETTO_VERSION

   Indicates the version number of *vaporetto* used by this wrapper. It can be used to check the
   compatibility of the model file.
