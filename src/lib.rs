use std::fmt::Write;
use std::io::Read;

use pyo3::{exceptions::PyValueError, prelude::*, types::PyUnicode};

use hashbrown::HashMap;
use ouroboros::self_referencing;
use vaporetto_rules::{
    sentence_filters::{ConcatGraphemeClustersFilter, KyteaWsConstFilter},
    string_filters::KyteaFullwidthFilter,
    SentenceFilter, StringFilter,
};
use vaporetto_rust::errors::VaporettoError;
use vaporetto_rust::{CharacterType, KyteaModel, Model, Predictor, Sentence};

/// Representation of a token.
#[pyclass]
struct Token {
    list: Py<TokenList>,
    index: usize,
}

#[pymethods]
impl Token {
    /// Return the surface of this token.
    ///
    /// :rtype: str
    fn surface(&self, py: Python) -> Py<PyUnicode> {
        self.list.borrow(py).surfaces[self.index].0.clone_ref(py)
    }

    /// Return the start position (inclusive) in characters.
    ///
    /// :rtype: int
    fn start(&self, py: Python) -> usize {
        self.list.borrow(py).surfaces[self.index].1
    }

    /// Return the end position (exclusive) in characters.
    ///
    /// :rtype: int
    fn end(&self, py: Python) -> usize {
        self.list.borrow(py).surfaces[self.index].2
    }

    /// Return the tag assigned to a given index.
    ///
    /// :param index: An index of the set of tags
    /// :type index: int
    /// :rtype: Optional[str]
    /// :raises ValueError: if the index is out of range.
    #[pyo3(signature = (index, /))]
    fn tag(&self, py: Python, index: usize) -> PyResult<Option<Py<PyUnicode>>> {
        let list = self.list.borrow(py);
        if index < list.n_tags {
            let pos = list.surfaces[self.index].2 - 1;
            Ok(list.tags[pos * list.n_tags + index]
                .as_ref()
                .map(|t| t.clone_ref(py)))
        } else {
            Err(PyValueError::new_err("list index out of range"))
        }
    }

    /// Return the number of tags assigned to this token.
    ///
    /// :rtype: int
    fn n_tags(&self, py: Python) -> usize {
        self.list.borrow(py).n_tags
    }

    fn __str__(&self, py: Python) -> Py<PyUnicode> {
        self.surface(py)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let list = self.list.borrow(py);
        let surface = list.surfaces[self.index].0.as_ref(py).to_str()?;
        let mut result = format!("Token {{ surface: {:?}, tags: [", surface);
        let pos = list.surfaces[self.index].2 - 1;
        for i in 0..list.n_tags {
            if i != 0 {
                result += ", ";
            }
            if let Some(tag) = list.tags[pos * list.n_tags + i].as_ref() {
                let tag = tag.as_ref(py).to_str()?;
                write!(&mut result, "{:?}", tag).unwrap();
            } else {
                result += "None";
            }
        }
        result += "] }}";
        Ok(result)
    }
}

/// Iterator that returns :class:`.Token`.
#[pyclass]
struct TokenIterator {
    list: Py<TokenList>,
    index: usize,
    len: usize,
}

#[pymethods]
impl TokenIterator {
    fn __next__(&mut self, py: Python) -> Option<Token> {
        if self.index < self.len {
            let index = self.index;
            self.index += 1;
            Some(Token {
                list: self.list.clone_ref(py),
                index,
            })
        } else {
            None
        }
    }
}

/// List of :class:`.Token` returned by the tokenizer.
#[pyclass]
struct TokenList {
    surfaces: Vec<(Py<PyUnicode>, usize, usize)>,
    tags: Vec<Option<Py<PyUnicode>>>,
    n_tags: usize,
}

#[pymethods]
impl TokenList {
    fn __len__(&self) -> usize {
        self.surfaces.len()
    }

    fn __getitem__(self_: Py<Self>, py: Python, index: usize) -> PyResult<Token> {
        if index < self_.borrow(py).surfaces.len() {
            Ok(Token { list: self_, index })
        } else {
            Err(PyValueError::new_err("list index out of range"))
        }
    }

    fn __iter__(self_: Py<Self>, py: Python) -> TokenIterator {
        let len = self_.borrow(py).surfaces.len();
        TokenIterator {
            list: self_,
            index: 0,
            len,
        }
    }
}

#[self_referencing]
struct PredictorWrapper {
    predictor: Predictor,
    #[borrows(predictor)]
    #[covariant]
    sentence_buf: Sentence<'static, 'this>,
    #[borrows(predictor)]
    #[covariant]
    norm_sentence_buf: Sentence<'static, 'this>,
}

impl PredictorWrapper {
    fn predict(
        &mut self,
        text: String,
        predict_tags: bool,
        normalize: bool,
        post_filters: &[Box<dyn SentenceFilter>],
    ) -> Result<(), VaporettoError> {
        self.with_mut(|self_| {
            self_.sentence_buf.update_raw(text)?;
            if normalize {
                let normalizer = KyteaFullwidthFilter;
                let norm_text = normalizer.filter(self_.sentence_buf.as_raw_text());
                self_.norm_sentence_buf.update_raw(norm_text)?;
                self_.predictor.predict(self_.norm_sentence_buf);
                post_filters
                    .iter()
                    .for_each(|filter| filter.filter(self_.norm_sentence_buf));
                self_
                    .sentence_buf
                    .boundaries_mut()
                    .copy_from_slice(self_.norm_sentence_buf.boundaries());
                if predict_tags {
                    self_.norm_sentence_buf.fill_tags();
                    self_
                        .sentence_buf
                        .reset_tags(self_.norm_sentence_buf.n_tags());
                    self_
                        .sentence_buf
                        .tags_mut()
                        .clone_from_slice(self_.norm_sentence_buf.tags());
                }
            } else {
                self_.predictor.predict(self_.sentence_buf);
                post_filters
                    .iter()
                    .for_each(|filter| filter.filter(self_.sentence_buf));
                if predict_tags {
                    self_.sentence_buf.fill_tags();
                }
            }
            Ok(())
        })
    }
}

/// Python binding of Vaporetto tokenizer.
///
/// Examples:
///     >>> import vaporetto
///     >>> with open('path/to/model.zst', 'rb') as fp:
///     >>>     model = fp.read()
///     >>> tokenizer = vaporetto.Vaporetto(model, predict_tags = True)
///     >>> tokenizer.tokenize_to_string('まぁ社長は火星猫だ')
///     'まぁ/名詞/マー 社長/名詞/シャチョー は/助詞/ワ 火星/名詞/カセー 猫/名詞/ネコ だ/助動詞/ダ'
///     >>> tokens = tokenizer.tokenize('まぁ社長は火星猫だ')
///     >>> len(tokens)
///     6
///     >>> tokens[0].surface()
///     'まぁ'
///     >>> tokens[0].tag(0)
///     '名詞'
///     >>> tokens[0].tag(1)
///     'マー'
///     >>> [token.surface() for token in tokens]
///     ['まぁ', '社長', 'は', '火星', '猫', 'だ']
///
/// :param model: A byte sequence of the model.
/// :param predict_tags: If True, the tokenizer predicts tags.
/// :param wsconst: Does not split the specified character types.
///                 ``D``: Digit, ``R``: Roman, ``H``: Hiragana, ``T``: Katakana, ``K``: Kanji,
///                 ``O``: Other, ``G``: Grapheme cluster. You can specify multiple types such as
///                 ``DGR``.
/// :param norm: If True, input texts will be normalized beforehand.
/// :type model: bytes
/// :type predict_tags: bool
/// :type wsconst: str
/// :type norm: bool
/// :rtype: vaporetto.Vaporetto
/// :raises ValueError: if the model is invalid.
/// :raises ValueError: if the wsconst value is invalid.
#[pyclass]
#[pyo3(text_signature = "(model, /, predict_tags = False, wsconst = \"\", norm = True)")]
struct Vaporetto {
    wrapper: PredictorWrapper,
    predict_tags: bool,
    normalize: bool,
    post_filters: Vec<Box<dyn SentenceFilter>>,
    word_cache: HashMap<String, Py<PyUnicode>>,
    tag_cache: HashMap<String, Py<PyUnicode>>,
    string_buf: String,
}

impl Vaporetto {
    fn create_internal(
        py: Python,
        model: Model,
        predict_tags: bool,
        wsconst: &str,
        normalize: bool,
    ) -> PyResult<Self> {
        // For efficiency, this library creates PyStrings of dictionary words beforehand and uses
        // them if available instead of creating PyStrings every time.
        let mut word_cache = HashMap::new();
        for record in model.dictionary() {
            let word = record.get_word();
            word_cache.insert(word.to_string(), PyUnicode::new(py, word).into());
        }

        let predictor = py.allow_threads(|| {
            Predictor::new(model, predict_tags).map_err(|e| PyValueError::new_err(e.to_string()))
        })?;

        let mut post_filters: Vec<Box<dyn SentenceFilter>> = vec![];
        for c in wsconst.chars() {
            post_filters.push(match c {
                'D' => Box::new(KyteaWsConstFilter::new(CharacterType::Digit)),
                'R' => Box::new(KyteaWsConstFilter::new(CharacterType::Roman)),
                'H' => Box::new(KyteaWsConstFilter::new(CharacterType::Hiragana)),
                'T' => Box::new(KyteaWsConstFilter::new(CharacterType::Katakana)),
                'K' => Box::new(KyteaWsConstFilter::new(CharacterType::Kanji)),
                'O' => Box::new(KyteaWsConstFilter::new(CharacterType::Other)),
                'G' => Box::new(ConcatGraphemeClustersFilter),
                c => return Err(PyValueError::new_err(format!("Invalid wsconst: {c}"))),
            });
        }

        let wrapper = PredictorWrapperBuilder {
            predictor,
            sentence_buf_builder: |_| Sentence::default(),
            norm_sentence_buf_builder: |_| Sentence::default(),
        }
        .build();

        Ok(Self {
            wrapper,
            predict_tags,
            normalize,
            post_filters,
            word_cache,
            tag_cache: HashMap::new(),
            string_buf: String::new(),
        })
    }
}

#[pymethods]
impl Vaporetto {
    #[new]
    #[pyo3(signature = (model, /, predict_tags=false, wsconst="", norm=true))]
    fn new(
        py: Python,
        model: &[u8],
        predict_tags: bool,
        wsconst: &str,
        norm: bool,
    ) -> PyResult<Self> {
        let mut buf = vec![];
        let (model, _) = py.allow_threads(|| {
            let mut decoder = ruzstd::StreamingDecoder::new(model)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            decoder
                .read_to_end(&mut buf)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Model::read_slice(&buf).map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        Self::create_internal(py, model, predict_tags, wsconst, norm)
    }

    /// Create a new Vaporetto instance from a KyTea's model.
    ///
    /// Vaporetto does not support tag prediction with KyTea's model.
    ///
    /// :param model: A byte sequence of the model.
    /// :param wsconst: Does not split the specified character types.
    /// :param norm: If True, input texts will be normalized beforehand.
    /// :type model: bytes
    /// :type wsconst: str
    /// :type norm: bool
    /// :rtype: vaporetto.Vaporetto
    /// :raises ValueError: if the model is invalid.
    /// :raises ValueError: if the wsconst value is invalid.
    #[staticmethod]
    #[pyo3(signature = (model, /, wsconst="", norm=true))]
    #[pyo3(text_signature = "(model, /, wsconst = \"\", norm = True)")]
    fn create_from_kytea_model(
        py: Python,
        model: &[u8],
        wsconst: &str,
        norm: bool,
    ) -> PyResult<Self> {
        let model = py.allow_threads(|| {
            let kytea_model =
                KyteaModel::read(model).map_err(|e| PyValueError::new_err(e.to_string()))?;
            Model::try_from(kytea_model).map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        Self::create_internal(py, model, false, wsconst, norm)
    }

    /// Tokenize a given text and return as a list of tokens.
    ///
    /// :param text: A text to tokenize.
    /// :type text: str
    /// :rtype: vaporetto.TokenList
    #[pyo3(signature = (text, /))]
    fn tokenize(&mut self, py: Python, text: String) -> TokenList {
        if self
            .wrapper
            .predict(text, self.predict_tags, self.normalize, &self.post_filters)
            .is_err()
        {
            return TokenList {
                surfaces: vec![],
                tags: vec![],
                n_tags: 0,
            };
        }
        let s = self.wrapper.borrow_sentence_buf();
        let surfaces = s
            .iter_tokens()
            .map(|token| {
                let surface = self
                    .word_cache
                    .get(token.surface())
                    .map(|surf| surf.clone_ref(py))
                    .unwrap_or_else(|| PyUnicode::new(py, token.surface()).into());
                (surface, token.start(), token.end())
            })
            .collect();
        let tags = s
            .tags()
            .iter()
            .map(|tag| {
                tag.as_ref().map(|tag| {
                    self.tag_cache
                        .raw_entry_mut()
                        .from_key(tag.as_ref())
                        .or_insert_with(|| {
                            (tag.to_string(), PyUnicode::new(py, tag.as_ref()).into())
                        })
                        .1
                        .clone_ref(py)
                })
            })
            .collect();
        TokenList {
            surfaces,
            tags,
            n_tags: s.n_tags(),
        }
    }

    /// Tokenize a given text and return as a string.
    ///
    /// :param text: A text to tokenize.
    /// :type text: str
    /// :rtype: str
    #[pyo3(signature = (text, /))]
    fn tokenize_to_string(&mut self, py: Python, text: String) -> Py<PyUnicode> {
        if self
            .wrapper
            .predict(text, self.predict_tags, self.normalize, &self.post_filters)
            .is_ok()
        {
            self.wrapper
                .borrow_sentence_buf()
                .write_tokenized_text(&mut self.string_buf);
        } else {
            self.string_buf.clear();
        }
        PyUnicode::new(py, &self.string_buf).into()
    }
}

#[pymodule]
fn vaporetto(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Vaporetto>()?;
    m.add_class::<TokenList>()?;
    m.add_class::<TokenIterator>()?;
    m.add_class::<Token>()?;
    m.add("VAPORETTO_VERSION", vaporetto_rust::VERSION)?;
    Ok(())
}
