import jsonimport pickleimport osimport reimport loggingfrom random import shuffleimport numpy as npfrom keras.utils.data_utils import Sequencefrom keras.preprocessing.sequence import pad_sequencesfrom adultfilter.config import TITLES_PAD, DESCRIPTION_PAD, \    KEYWORDS_PAD, TEXT_PAD, LABEL_NUMBER, WORD2VEC_EMBEDDINGS, VOCABULARY_PATHclass TextConverter:    """Class for text pre-processing methods."""    def __init__(self):        self.vocab = None        # regexp pattern for extracting tokens from text        self.tokenization_pattern = re.compile(r"""                \b(                (                 (\w)(?!\d)     # allowed characters (letters and no digits)                 (?!\3{3,})     # remove multi characters ('zzzzzz', 'ccccat')                )                 {3,}           # get tokens with lengths >= 3                )                \b""", re.UNICODE | re.VERBOSE)    @staticmethod    def make_and_save_word2vec(embeddings_dir):        """        Make embeddings and vocabulary and save them into data dir.        Load separated embeddings of languages amd make big multi-language        embeddings with vocabulary.        :param embeddings_dir: [str] path to languages word embeddings dir        :return: [tuple (dict, numpy array)] vocabulary and embeddings        """        logging.info(u'Starting to make a vocabulary and embeddings array')        vocab = dict()        total_emb = None        idx = 1        for lang_emb_path in os.scandir(embeddings_dir):            logging.info(u'Processing {}'.format(lang_emb_path.name))            words, emb = pickle.load(open(lang_emb_path.path, 'rb'),                                     encoding='latin1')            emb_idx = list()            for i, word in enumerate(words):                if word not in vocab:                    vocab[word] = idx                    emb_idx.append(i)                    if idx % 1000 == 0:                        logging.info(u'Founded {} unique words'.format(idx))                    idx += 1            print('\n')            if total_emb:                total_emb = np.concatenate((total_emb, emb[emb_idx]))            else:                total_emb = np.concatenate((np.zeros((1, emb.shape[1])),                                            emb[emb_idx]))        logging.info(            u'Finished with {} size of vocabulary'            u' and {} shape of embeddings'.format(len(vocab), total_emb.shape)        )        logging.info(u'Saving vocabulary as .json and embeddings as .pickle')        with open(os.path.join(VOCABULARY_PATH, 'vocabulary.json'), 'w+') as f:            json.dump(vocab, f, indent=4)        with open(os.path.join(WORD2VEC_EMBEDDINGS, 'embeddings.pickle'), 'wb') as f:            pickle.dump(total_emb, f)        return vocab, total_emb    def load_vocab(self, vocab_path):        """        Load vocabulary and save it in instance of a class as attribute.        :param vocab_path: [str] path to vocabulary        :return: self        """        logging.info(u'Loading vocabulary into class.')        self.vocab = json.load(open(vocab_path, 'r+'))        return self    def tokenize(self, text):        """        Tokenize given text into words.        :param text: [str] text        :return: [list of str] list of words        """        text = text.lower() if text else ''        # get tokens        tokens = [m.groups(0)[0]                  for m in self.tokenization_pattern.finditer(text)]        return tokens or []    def word2idx(self, word):        """        Convert word to index in vocabulary.        Not known words will have 0 as index.        :param word: [str] word        :return: [ind] index of word in vocabulary        """        idx = self.vocab.get(word, 0)        return idx    def text2seq(self, text, tokenize=True, pad=0):        """        Transform text to sequence of indexes.        :param text: [str or list of str] raw text or list of tokens        :param tokenize: [bool] need to tokenize raw text or word list given        :param pad: [int] how big must be sequence (if pad=0 - no limits)        :return:        """        if tokenize:            text = self.tokenize(text)[:pad] if pad else self.tokenize(text)        sequence = [[self.word2idx(word) for word in text]]        if pad:            sequence = pad_sequences(np.asanyarray(sequence),                                     maxlen=pad,                                     padding='post',                                     truncating='post',                                     dtype='int32',                                     value=0)        return sequenceclass CorpusLoader(Sequence, TextConverter):    """    Load and transform corpus of data for training of NN model.    Make a generator of batches of texts.    Each batch transform to lists of sequences before input for NN model.    """    def __init__(self, data_dir):        super().__init__()        self.data_dir = data_dir        self.file_batches = sorted(os.listdir(self.data_dir))        self.batch_count = len(self.file_batches)    def __len__(self):        return self.batch_count    def __getitem__(self, idx):        if idx == 0:            shuffle(self.file_batches)        if idx >= self.batch_count:            raise IndexError        with open(os.path.join(self.data_dir,                               self.file_batches[idx]), 'r+') as json_f:            docs_batch = json.load(json_f)        n_docs = len(docs_batch)        titles = np.zeros(shape=(n_docs, TITLES_PAD))        keywords = np.zeros(shape=(n_docs, KEYWORDS_PAD))        description = np.zeros(shape=(n_docs, DESCRIPTION_PAD))        text = np.zeros(shape=(n_docs, TEXT_PAD))        labels = np.zeros(shape=(n_docs, LABEL_NUMBER))        for doc_idx, doc in enumerate(docs_batch.values()):            raw_title = doc['title']            raw_keywords = ' '.join(doc['keywords'])            raw_description = doc['description']            raw_text = doc['text']            titles[doc_idx] = self.text2seq(                raw_title,                tokenize=True,                pad=TITLES_PAD            )            keywords[doc_idx] = self.text2seq(                raw_keywords,                tokenize=True,                pad=KEYWORDS_PAD            )            description[doc_idx] = self.text2seq(                raw_description,                tokenize=True,                pad=DESCRIPTION_PAD            )            text[doc_idx] = self.text2seq(                raw_text,                tokenize=True,                pad=TEXT_PAD            )            labels[doc_idx][doc['label']] = 1        return [titles, keywords, description, text], labels