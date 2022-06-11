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

/// Token list returned by the tokenizer.
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
    string_buf: RefCell<String>,
    sentence_buf1: RefCell<Sentence<'static, 'static>>,
    sentence_buf2: RefCell<Sentence<'static, 'static>>,
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
            string_buf: RefCell::new(String::new()),
            sentence_buf1: RefCell::new(Sentence::default()),
            sentence_buf2: RefCell::new(Sentence::default()),
        })
    }

    fn tokenize_internal<'a>(&'a self, s: &mut Sentence<'_, 'a>) {
        let predictor = &self.predictor;
        let normalizer = &self.normalizer;
        let post_filters = &self.post_filters;
        let predict_tags = self.predict_tags;
        if let Some(normalizer) = normalizer {
            // Sentence buffer requires lifetimes of text and predictor, but the Vaporetto struct
            // cannot have such a Sentence, so we use transmute() to disguise lifetimes.
            let norm_s = &mut self.sentence_buf2.borrow_mut();
            let norm_s = unsafe {
                std::mem::transmute::<&mut Sentence<'static, 'static>, &mut Sentence<'_, '_>>(
                    norm_s,
                )
            };
            norm_s
                .update_raw(normalizer.filter(s.as_raw_text()))
                .unwrap();
            predictor.predict(norm_s);
            post_filters.iter().for_each(|filter| filter.filter(norm_s));
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
        let mut buff = vec![];
        let (model, _) = py.allow_threads(|| {
            let mut f = Cursor::new(model);
            let mut decoder = ruzstd::StreamingDecoder::new(&mut f)
                .map_err(PyValueError::new_err)?;
            decoder
                .read_to_end(&mut buff)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
        let model = py.allow_threads(|| {
            let kytea_model =
                KyteaModel::read(f).map_err(|e| PyValueError::new_err(e.to_string()))?;
            Model::try_from(kytea_model).map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        Self::create_internal(py, model, predict_tags, wsconst, norm)
    }

    /// Tokenize a given text and return as a list of tokens.
    ///
    /// :param text: A text to tokenize.
    /// :type text: str
    /// :type out: vaporetto.TokenList
    #[pyo3(text_signature = "($self, text, /)")]
    fn tokenize(&self, py: Python, text: &str) -> TokenList {
        // Sentence buffer requires lifetimes of text and predictor, but the Vaporetto struct
        // cannot have such a Sentence, so we use transmute() to disguise lifetimes.
        let s = &mut self.sentence_buf1.borrow_mut();
        let s = unsafe {
            std::mem::transmute::<&mut Sentence<'static, 'static>, &mut Sentence<'_, '_>>(s)
        };
        if s.update_raw(text).is_ok() {
            self.tokenize_internal(s);

            // Creates TokenIterator
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
            let tag_cache = &mut self.tag_cache.borrow_mut();
            let tags = s
                .tags()
                .iter()
                .map(|tag| {
                    tag.as_ref().map(|tag| {
                        tag_cache
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
    fn tokenize_to_string(&self, py: Python, text: &str) -> Py<PyUnicode> {
        let buf = &mut self.string_buf.borrow_mut();
        // Sentence buffer requires lifetimes of text and predictor, but the Vaporetto struct
        // cannot have such a Sentence, so we use transmute() to disguise lifetimes.
        let s = &mut self.sentence_buf1.borrow_mut();
        let s = unsafe {
            std::mem::transmute::<&mut Sentence<'static, 'static>, &mut Sentence<'_, '_>>(s)
        };
        if s.update_raw(text).is_ok() {
            self.tokenize_internal(s);
            s.write_tokenized_text(buf);
        }
        PyUnicode::new(py, buf).into()
    }
}

#[pymodule]
fn vaporetto(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Vaporetto>()?;
    m.add_class::<TokenList>()?;
    m.add_class::<Token>()?;
    Ok(())
}
