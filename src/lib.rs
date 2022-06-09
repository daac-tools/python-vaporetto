use std::cell::RefCell;
use std::fmt::Write;
use std::io::{Cursor, Read};

use pyo3::{exceptions::PyValueError, prelude::*, types::PyUnicode};

use hashbrown::HashMap;
use vaporetto::{CharacterType, KyteaModel, Model, Predictor, Sentence};
use vaporetto_rules::{
    sentence_filters::{ConcatGraphemeClustersFilter, KyteaWsConstFilter},
    string_filters::KyteaFullwidthFilter,
    SentenceFilter, StringFilter,
};

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
    /// :type out: str
    #[pyo3(text_signature = "($self, /)")]
    fn surface(&self, py: Python) -> Py<PyUnicode> {
        self.list.borrow(py).surfaces[self.index].0.clone_ref(py)
    }

    /// Return the start position (inclusive) in characters.
    ///
    /// :type out: int
    #[pyo3(text_signature = "($self, /)")]
    fn start(&self, py: Python) -> usize {
        self.list.borrow(py).surfaces[self.index].1
    }

    /// Return the end position (exclusive) in characters.
    ///
    /// :type out: int
    #[pyo3(text_signature = "($self, /)")]
    fn end(&self, py: Python) -> usize {
        self.list.borrow(py).surfaces[self.index].2
    }

    /// Return the tag assigned to a given index. If the index is out of range,
    /// this functuon throws a ValueError.
    ///
    /// :param index: An index of the set of tags
    /// :type index: int
    /// :type out: Optional[str]
    #[pyo3(text_signature = "($self, index, /)")]
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
    /// :type out: int
    #[pyo3(text_signature = "($self, /)")]
    fn n_tags(&self, py: Python) -> usize {
        self.list.borrow(py).n_tags
    }

    fn __str__(&self, py: Python) -> Py<PyUnicode> {
        self.surface(py)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let list = self.list.borrow(py);
        let surface: String = list.surfaces[self.index].0.as_ref(py).extract()?;
        let mut result = format!("Token {{ surface: {:?}, tags: [", surface);
        let pos = list.surfaces[self.index].2 - 1;
        for i in 0..list.n_tags {
            if i != 0 {
                result += ", ";
            }
            if let Some(tag) = list.tags[pos * list.n_tags + i].as_ref() {
                let tag: String = tag.as_ref(py).extract()?;
                write!(&mut result, "{:?}", tag).unwrap();
            } else {
                result += "None";
            }
        }
        result += "] }}";
        Ok(result)
    }
}

/// Iterator of tokens.
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
#[pyclass]

/// Token list returned by the tokenizer.
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
///     ['まぁ', '社長', 'は', '火星', '猫', 'だ']]
///
/// :param model: A byte sequence of the model.
/// :param predict_tags: If True, the tokenizer predicts tags.
/// :param wsconst: Does not split the specified character types.
/// :param norm: If True, input texts will be normalized beforehand.
/// :type model: bytes
/// :type predict_tags: bool
/// :type wsconst: str
/// :type norm: bool
/// :type out: vaporetto.Vaporetto
#[pyclass]
#[pyo3(text_signature = "($self, model, /, predict_tags = False, wsconst = \"\", norm = True)")]
struct Vaporetto {
    predictor: Predictor,
    predict_tags: bool,
    normalizer: Option<KyteaFullwidthFilter>,
    post_filters: Vec<Box<dyn SentenceFilter>>,
    word_cache: HashMap<String, Py<PyUnicode>>,
    tag_cache: RefCell<HashMap<String, Py<PyUnicode>>>,
}

impl Vaporetto {
    fn create_internal(
        py: Python,
        model: Model,
        predict_tags: bool,
        wsconst: &str,
        norm: bool,
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

        let normalizer = norm.then(|| KyteaFullwidthFilter);
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

        Ok(Self {
            predictor,
            predict_tags,
            normalizer,
            post_filters,
            word_cache,
            tag_cache: RefCell::new(HashMap::new()),
        })
    }
    fn tokenize_internal<'a>(&'a self, py: Python, s: &mut Sentence<'_, 'a>) {
        let predictor = &self.predictor;
        let normalizer = &self.normalizer;
        let post_filters = &self.post_filters;
        let predict_tags = self.predict_tags;
        py.allow_threads(|| {
            if let Some(normalizer) = normalizer {
                let mut norm_s = Sentence::from_raw(normalizer.filter(s.as_raw_text())).unwrap();
                predictor.predict(&mut norm_s);
                post_filters
                    .iter()
                    .for_each(|filter| filter.filter(&mut norm_s));
                s.boundaries_mut().copy_from_slice(norm_s.boundaries());
                if predict_tags {
                    norm_s.fill_tags();
                    s.reset_tags(norm_s.n_tags());
                    s.tags_mut().clone_from_slice(norm_s.tags());
                }
            } else {
                predictor.predict(s);
                post_filters.iter().for_each(|filter| filter.filter(s));
                if predict_tags {
                    s.fill_tags();
                }
            }
        });
    }
}

#[pymethods]
impl Vaporetto {
    #[new]
    #[args(predict_tags = "false", wsconst = "\"\"", norm = "true")]
    fn new(
        py: Python,
        model: &[u8],
        predict_tags: bool,
        wsconst: &str,
        norm: bool,
    ) -> PyResult<Self> {
        let mut f = Cursor::new(model);
        let mut decoder = ruzstd::StreamingDecoder::new(&mut f).unwrap();
        let mut buff = vec![];
        decoder.read_to_end(&mut buff).unwrap();
        let (model, _) = py.allow_threads(|| {
            Model::read_slice(&buff).map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        Self::create_internal(py, model, predict_tags, wsconst, norm)
    }

    /// Create a new Vaporetto instance from a KyTea's model.
    ///
    /// :param model: A byte sequence of the model.
    /// :param predict_tags: If True, the tokenizer predicts tags.
    /// :param wsconst: Does not split the specified character types.
    /// :param norm: If True, input texts will be normalized beforehand.
    /// :type model: bytes
    /// :type predict_tags: bool
    /// :type wsconst: str
    /// :type norm: bool
    /// :type out: vaporetto.Vaporetto
    #[staticmethod]
    #[args(predict_tags = "false", wsconst = "\"\"", norm = "true")]
    #[pyo3(text_signature = "(model, /, predict_tags = False, wsconst = \"\", norm = True)")]
    fn create_from_kytea_model(
        py: Python,
        model: &[u8],
        predict_tags: bool,
        wsconst: &str,
        norm: bool,
    ) -> PyResult<Self> {
        let f = Cursor::new(model);
        let kytea_model = py.allow_threads(|| {
            KyteaModel::read(f).map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        let model =
            Model::try_from(kytea_model).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Self::create_internal(py, model, predict_tags, wsconst, norm)
    }

    /// Tokenize a given text and return as a list of tokens.
    ///
    /// :param text: A text to tokenize.
    /// :type text: str
    /// :type out: vaporetto.TokenList
    #[pyo3(text_signature = "($self, text, /)")]
    fn tokenize(&self, py: Python, text: &str) -> TokenList {
        if let Ok(mut s) = Sentence::from_raw(text) {
            self.tokenize_internal(py, &mut s);

            // Creates TokenIterator
            let mut surfaces = vec![];
            for token in s.iter_tokens() {
                let surface = self
                    .word_cache
                    .get(token.surface())
                    .map(|surf| surf.clone_ref(py))
                    .unwrap_or_else(|| PyUnicode::new(py, token.surface()).into());
                surfaces.push((surface, token.start(), token.end()));
            }
            let mut tags = vec![];
            let tag_cache = &mut self.tag_cache.borrow_mut();
            for tag in s.tags() {
                tags.push(tag.as_ref().map(|tag| {
                    if let Some(py_tag) = tag_cache.get(tag.as_ref()) {
                        py_tag.clone_ref(py)
                    } else {
                        let py_tag: Py<PyUnicode> = PyUnicode::new(py, tag.as_ref()).into();
                        tag_cache.insert(tag.to_string(), py_tag.clone_ref(py));
                        py_tag
                    }
                }));
            }
            TokenList {
                surfaces,
                tags,
                n_tags: s.n_tags(),
            }
        } else {
            TokenList {
                surfaces: vec![],
                tags: vec![],
                n_tags: 0,
            }
        }
    }

    /// Tokenize a given text and return as a string.
    ///
    /// :param text: A text to tokenize.
    /// :type text: str
    /// :type out: str
    #[pyo3(text_signature = "($self, text, /)")]
    fn tokenize_to_string(&self, py: Python, text: &str) -> String {
        let mut buf = String::new();
        if let Ok(mut s) = Sentence::from_raw(text) {
            self.tokenize_internal(py, &mut s);
            s.write_tokenized_text(&mut buf);
        }
        buf
    }
}

#[pymodule]
fn vaporetto(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Vaporetto>()?;
    m.add_class::<TokenList>()?;
    m.add_class::<Token>()?;
    Ok(())
}