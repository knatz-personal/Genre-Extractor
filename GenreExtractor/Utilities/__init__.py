"""This module contains utility classes and functions
for various tasks.
"""
import csv
import logging
import pickle
import re
import string
import sys
import time
import traceback

from nltk.corpus import stopwords
from nltk.corpus.reader import wordnet
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from configparser import ConfigParser
from logging import Logger, Handler

_author__ = "Nathan Zwelibanzi Khupe"
__copyright__ = "Copyright 2017, Nathan Zwelibanzi Khupe"
__credits__ = ["Nathan Zwelibanzi Khupe"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nathan Zwelibanzi Khupe"
__email__ = "001knatz@gmail.com"
__docformat__ = "numpy"


class LogUtils(object):
    """Utility functions for working with the python logging package
    """

    def __init__(self, saveToFile: bool, **kwargs):
        """Constructor for :class:`LogUtils`

        Parameters
        ----------
        saveToFile : bool
            Save logs to a file? Y/N
        **kwargs
            Default arguments
        """
        LogUtils.log = LogUtils.make_logger(saveToFile)
        return super().__init__(**kwargs)

    @classmethod
    def make_logger(cls, isSaved: bool) -> Logger:
        """Create logger with predefined settings

        Parameters
        ----------
        isSaved : bool
            Save logs to a file? Y/N

        Returns
        -------
        Logger
            A configured :class:`Logger` object
        """
        t = time.strftime("[%d %B %Y %H-%M-%S]", time.localtime())
        logger = logging.getLogger(t)
        logger.setLevel(logging.DEBUG)

        # log formatter
        formatter = logging.Formatter("%(asctime)s  %(levelname)s: %(message)s ",
            datefmt="%d-%m-%Y %H:%M:%S")

        # File log
        if isSaved is True:
            fh = logging.FileHandler(logger.name + ".log", mode="w")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            pass
        else:
            # Console log
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            pass

        return logger

    @classmethod
    def add_handler(cls, handler: Handler):
        """Add a new handler to the logger

        Parameters
        ----------
        handler : Handler
            An instance of :class:`Handler`
        """
        cls.log.addHandler(handler)
        pass

    @classmethod
    def write(cls, type: str, message: str):
        """Write a message to the logger

        Parameters
        ----------
        type : str
            The type of the message e.g. info, warn, error, critical or debug
        message : str
            The message to save in the log
        """
        try:
            if type.lower() == "info":
                cls.log.info(message)
            elif type.lower() == "debug":
                cls.log.debug(message)
            elif type.lower() == "warn":
                cls.log.warn(message)
            elif type.lower() == "error":
                cls.log.error(message)
            elif type.lower() == "critical":
                cls.log.critical(message)
            else:
                cls.log.info(message)
                pass
        except:
            pass
        pass

    pass  # end of LogUtils


class CorpusUtils(object):
    """Utility functions for building a corpus of texts
    """

    def __init__(self, **kwargs):
        """Constructor of :class:`CorpusUtils`

        Parameters
        ----------
        **kwargs
            Description
        """
        return super().__init__(**kwargs)

    @classmethod
    def create_plain_corpus(cls, directory: str, file_pattern: str) -> PlaintextCorpusReader:
        """Create a simple uncategorised corpus

        Parameters
        ----------
        directory : str
            The path to the corpus directory
        file_pattern : str
            Regex definition of files to include in the corpus

        Returns
        -------
        PlaintextCorpusReader
            PlaintextCorpusReader object
        """
        corpus = PlaintextCorpusReader(directory, file_pattern)
        return corpus

    @classmethod
    def create_genred_corpus(cls, directory: str,
                             file_pattern: str,
                             genre_file: str) ->    CategorizedPlaintextCorpusReader:
        """Create a categorised corpus

        Parameters
        ----------
        directory : str
            The path to the corpus directory
        file_pattern : str
            Regex definition of files to include in the corpu
        genre_file : str
            The CSV file with the category labelled file paths

        Notes
        -----
        The expected delimiter is `;;` in the file~genre mapping CSV file

        Returns
        -------
        CategorizedPlaintextCorpusReader
            CategorizedPlaintextCorpusReader object
        """
        corpus = CategorizedPlaintextCorpusReader(root=directory,
                                                  fileids=file_pattern,
                                                  cat_file=genre_file,
                                                  cat_delimiter=";;")
        return corpus

    @classmethod
    def get_split_corpus(cls, corpus: CategorizedPlaintextCorpusReader) -> tuple:
        """Split the CategorizedPlaintextCorpusReader into a training
        and testing corpus

        Parameters
        ----------
        corpus : CategorizedPlaintextCorpusReader
            A CategorizedPlaintextCorpusReader object

        Returns
        -------
        tuple[list, list]
            (Testing corpus, Training corpus)
        """
        testing_corpus, training_corpus = [], []

        for fileid in corpus.fileids():
            if "train" in fileid:
                file = corpus.raw(fileid)
                categories = corpus.categories(fileid)
                file = cls.pre_process_text(file)
                training_corpus.append((fileid, categories, file))
                pass
            elif "test" in fileid:
                file = corpus.raw(fileid)
                file = file.strip()
                categories = corpus.categories(fileid)
                testing_corpus.append((fileid, categories, file))
                pass
            pass

        return testing_corpus, training_corpus

    @classmethod
    def sanitize_file(cls, raw_text: str) -> str:
        """Clean a text file
        Operation`s
        * :func:`remove_whitespace`
        * :func:`remove_digits`
        * :func:`remove_stopwords`

        Parameters
        ----------
        raw_text : str
            Unstructured text

        Returns
        -------
        str
            Cleaned text
        """
        cleaner = TextSanitizer()
        # check raw text is not empty
        if not raw_text:
            LogUtils.write("error", "sanitize_file: raw_text param is empty")

        text_nsp = cleaner.remove_whitespace(raw_text)

        text_nner = cleaner.remove_digits(text_nsp, 3)

        clean_text = cleaner.remove_stopwords(text_nner)

        return clean_text

    @classmethod
    def sanitize_list(cls, raw_words: list) -> list:
        """Clean a list of words
        Operation`s
        * :func:`remove_whitespace`
        * :func:`transform_word_case`
        * :func:`remove_stopwords`
        * :func:`remove_digits`
        * :func:`lemmatize`

        Parameters
        ----------
        raw_words : list
            list of words

        Returns
        -------
        list
            list of clean words
        """
        # check raw text is not empty
        if not raw_words:
            LogUtils.write("error", "sanitize_list: raw_words param is empty")
            return

        cleaner = TextSanitizer()
        list_of_words = []

        for word in raw_words:
            try:
                nsp = cleaner.remove_whitespace(word)
                if nsp in ["", "``", "\"\""]:
                    continue
                if len(nsp) < 2:
                    continue
                case = cleaner.transform_word_case(nsp)
                nstop = cleaner.remove_stopword(case)
                if nstop is None:
                    continue
                norm = cleaner.remove_digits(nstop)
                if norm is None:
                    continue
                root = cleaner.lemmatize(norm)
                list_of_words.append(root)
            except Exception as e:
                LogUtils.write("error", "sanitize_list: {}".format(e.args))
                traceback.print_exc()
                continue
                pass
            pass

        return list_of_words

    @classmethod
    def pre_process_text(cls, raw_text: str) -> str:
        """
        Sanitize raw unstructured text document by:
            * remove url`s
            * convert roman numerals to digits
            * remove numbers - digits
            * exclude non feature words
            * exclude named entities

        Parameters
        ----------
        raw_text : str
            Unstructured text

        Returns
        -------
        str
            Processed text
        """
        cleaner = TextSanitizer()

        text_no_links = cleaner.remove_urls(raw_text)

        query = r'\b(?=[MDCLXVI]+\b)M{0,4}' + r"(CM|CD|D?C{0,3})" + \
            r'(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b'
        regex = re.compile(query, re.IGNORECASE)
        text_no_roman = regex.sub(cleaner.strip_roman_numerals, text_no_links)

        text_no_digits = cleaner.remove_digits(text_no_roman)

        text_filtered = []
        non_features = FileManager.load_csv("words_to_ignore.csv", '\n')

        for word in text_no_digits.split():
            if word not in non_features:
                text_filtered.append(word)
                pass
            pass

        #text_no_ne = cleaner.remove_named_entities(" ".join(text_filtered))

        return " ".join(text_filtered)
    pass  # end of CorpusUtils


class TextSanitizer(object):
    """Utility functions for preprocessing text
    """

    def __init__(self, **kwargs):
        """Constructor for :class:`TextSanitizer`

        Parameters
        ----------
        **kwargs
            Description
        """
        return super().__init__(**kwargs)

    def remove_whitespace(self, raw_text: str) -> list:
        """Remove extra spaces

        Parameters
        ----------
        raw_text : str
            Description

        Raises
        ------
        ValueError
            Description
        ValueError
        Description

        Returns
        -------
        list
            Description
        """
        if not raw_text:
            raise ValueError("param is empty")
        symbols = ['\x00', '_', '-', '!', '"', ',', '*', '#']

        for symb in symbols:
            raw_text = raw_text.replace(symb, ' ')
            pass

        raw_text = raw_text.strip()
        # remove extra white space
        raw_text = " ".join(raw_text.split())
        return raw_text

    def remove_stopwords(self, terms: list) -> list:
        """Remove filler words

        Parameters
        ----------
        terms : list
            Description

        Raises
        ------
        ValueError
            Description
        ValueError
        Description

        Returns
        -------
        list
            Description
        """
        if not terms:
            raise ValueError("{} is empty from {}".format(
                "terms", "remove_stopwords"))
        try:
            # include only unique stop words
            stop_words = set(stopwords.words('english'))

            list_of_words = []
            for word in terms:
                if word not in stop_words:
                    list_of_words.append(word)
                    pass
                pass

            return list_of_words
        except Exception as e:
            LogUtils.write("error", "{} {}".format("remove_stopwords", e.args))
            traceback.print_exc()
        return None

    def remove_stopword(self, word: str) -> str:
        """Remove filler word

        Parameters
        ----------
        word : str
            A word

        Returns
        -------
        str
            None or the given word
        """
        stop_words = set(stopwords.words('english'))
        if word not in stop_words:
            return word
        return None

    def remove_punctuation(self, raw_text: str) -> str:
        """Remove punctuation marks

        Parameters
        ----------
        raw_text : str
            Unstructured text

        Returns
        -------
        str
            Processed text

        Raises
        ------
        ValueError
            raw_text cannot be empty
        """
        if not raw_text:
            raise ValueError("{} is empty :{}".format(
                "raw_text", "remove_punctuation"))

        return raw_text.translate(str.maketrans(' ', ' ', string.punctuation))

    def remove_digits(self, raw_text: str) -> str:
        """Remove digits from text

        Parameters
        ----------
        raw_text : str
            Unstructured text

        Returns
        -------
        str
            Processed text

        Raises
        ------
        ValueError
            raw_text cannot be empty
        """
        if not raw_text:
            raise ValueError("{} is empty :{}".format(
                "raw_text", "remove_digits"))
        return re.sub(r'\d+', '', raw_text)

    def remove_named_entities(self, raw_text: str) -> str:
        """Remove named entities

        Parameters
        ----------
        raw_text : str
            Unstructured text

        Returns
        -------
        str
            Processed text

        Raises
        ------
        ValueError
            raw_text cannot be empty
        """
        if not raw_text:
            raise ValueError("{} is empty :{}".format(
                "raw_text", "remove_named_entities"))
        result = []
        ne = set()
        words = []
        for sentence in sent_tokenize(raw_text):
            temp = word_tokenize(sentence)

            for w in temp:
                words.append(w)
                pass

            tagged_words = pos_tag(temp)
            tagged_words = list(set(tagged_words))

            for word, pos in tagged_words:
                if pos == 'NNP' or word.endswith("'s") or word.endswith("s'"):
                    ne.add(word)
                    pass
                pass
            pass
        result = [word for word in words if word not in ne]
        return " ".join(result)

    def _convert_roman_to_integer(self, roman_numeral: str) -> int:
        """Convert roman numerals to integers

        Parameters
        ----------
        roman_numeral : str
            Roman numerals I, II, IV, V, X, ...

        Returns
        -------
        int
            Interger 1, 2, 4, 5, 10, ...
        """
        number = roman_numeral.upper()
        vals = (1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1)
        keys = "M  CM   D    CD   C    XC  L   XL  X   IX V  IV I".split()

        i = result = 0

        for arabic, roman in zip(vals, keys):
            while number[i:i + len(roman)] == roman:
                result += arabic
                i += len(roman)
                pass
            pass

        return result

    def remove_urls(self, raw_text: str) -> str:
        """Remove url`s

        Parameters
        ----------
        raw_text : str
            Unstructured text

        Returns
        -------
        str
            Processed text

        Raises
        ------
        ValueError
            raw_text cannot be empty
        """
        if not raw_text:
            raise ValueError("{} is empty :{}".format(
                "raw_text", "remove_urls"))
        return re.sub(r'^https?:\/\/.*[\r\n]*', '', raw_text,
                      flags=re.MULTILINE)

    def strip_roman_numerals(self, match) -> str:
        """Remove roman numerals

        Parameters
        ----------
        match : TYPE
            Regex match

        Returns
        -------
        str
            Processed text
        """

        return str(self._convert_roman_to_integer(match.group(0)))

    def lemmatize(self, term: str) -> str:
        """lemmatize word

        Parameters
        ----------
        term : str
            word

        Returns
        -------
        str
            lemmatized word
        """
        tagged = pos_tag([term])[0]
        word = tagged[0]
        tag = tagged[1]
        lemmatizer = WordNetLemmatizer()
        morphy_tag = {'NN': wordnet.NOUN, 'JJ': wordnet.ADJ,
                      'VB': wordnet.VERB, 'RB': wordnet.ADV}
        if tag[:2] in ['NN', 'JJ', 'VB', 'RB']:
            term = lemmatizer.lemmatize(word, morphy_tag[tag[:2]])
            pass
        else:
            term = lemmatizer.lemmatize(word)
            pass
        return term

    def tokenizer(self, raw_text: str) -> list:
        """Tokenize the body of text

        Parameters
        ----------
        raw_text : str
            Unstructured text

        Returns
        -------
        list
            list of words

        Raises
        ------
        ValueError
            raw_text cannot be empty
        """
        if not raw_text:
            raise ValueError("{} is empty from {}".format(
                "raw_text", "tokenizer"))
        try:
            sentences = sent_tokenize(raw_text)
            output = []
            for sentence in sentences:
                words = word_tokenize(sentence)
                for word in words:
                    if len(word) > 2:
                        output.append(word)
                        pass
                    pass
                pass
            return output
        except Exception as e:
            LogUtils.write("error", "{} {}".format("seperate_terms", e.args))
            traceback.print_exc()
            pass
        return None

    def bulk_transform_case(self, terms: list) -> list:
        """Resolve case issues: upper, lower, title or abbreviations

        Parameters
        ----------
        terms : list
            list of words

        Returns
        -------
        list
            case transformed word list

        Raises
        ------
        ValueError
            terms cannot be empty
        """
        if not terms:
            raise ValueError("{} is empty from {}".format(
                "terms", "bulk_transform_case"))
        try:
            output = []
            for term in terms:
                text = term.lower()
                temp = re.compile(r'\W+', re.UNICODE)
                temp = temp.split(text)
                temp = "".join(temp)
                output.append(temp)
                pass
            return output
        except Exception as e:
            LogUtils.log.error("{} {}".format("normalise_text", e.args))
        pass

    def transform_word_case(self, word: str) -> str:
        """Resolve case issues: upper, lower, title or abbreviations

        Parameters
        ----------
        word : str
            word

        Returns
        -------
        str
            lower case word

        Raises
        ------
        ValueError
            terms cannot be empty
        """
        if not word:
            raise ValueError("{} is empty from {}".format(
                "terms", "transform_word_case"))
        try:
            text = word.lower()
            temp = re.compile(r'\W+', re.UNICODE)
            temp = temp.split(text)
            temp = "".join(temp)
            return temp
        except Exception as e:
            LogUtils.log.error("{} {}".format("transform_word_case", e.args))
        return None

    @classmethod
    def dequote(cls, raw_text: str) -> str:
        """Remove quotation from the beginning and end of a given string

        Parameters
        ----------
        raw_text : str
            Unstructured text

        Returns
        -------
        str
            Processed text

        Raises
        ------
        ValueError
            raw_text cannot be empty
        """
        if not raw_text:
            raise ValueError("{} is empty from {}".format(
                "raw_text", "transform_word_case"))
        raw_text.strip()
        if (raw_text[0] == raw_text[-1]) and raw_text.startswith(("'", '"')):
            return raw_text[1:-1]
            pass
        return raw_text

    pass  # end of TextSanitizer


class ConfigUtils(object):
    """Utility functions for managing configuration settings

    Attributes
    ----------
    config : ConfigParser
        ConfigParser object
    """

    def __init__(self, parser: ConfigParser, **kwargs):
        """Constructor for :class:`ConfigUtils`

        Parameters
        ----------
        parser : ConfigParser
            Configuration parser
        **kwargs
            Description
        """
        self.config = parser
        return super().__init__(**kwargs)

    def get(self, setting: str, section="default") -> str:
        """Retrieve setting from a section of the config.py file
        default section is `DEFAULT`

        Parameters
        ----------
        setting : str
            Name of the setting
        section : str, optional
            Section name

        Returns
        -------
        str
            The setting value

        """
        temp = self.config[section.upper()][setting.lower()]
        return TextSanitizer.dequote(temp)

    def set(self, setting: str, value, section="default"):
        """Assign a value to a setting from a section of the config.py file
        default section is `DEFAULT`

        Parameters
        ----------
        setting : str
            Setting name
        value : TYPE
            Any value
        section : str, optional
            Section name
        """
        self.config[section.upper()][setting.lower()] = "`{}`,".format(value)
        pass

    def save(self):
        """Save latest settings to config file
        """
        with open("config.py", 'w') as configfile:
            self.config.write(configfile)
            pass
        pass

    def load(self):
        """Load settings contents of config file
        """
        self.config.read("config.py")
        pass

    pass  # end of ConfigUtils


class FileManager(object):
    """Manage file input and out operations
    """

    def __init__(self, *args, **kargs):
        """Constructor for :class:`FileManager`

        Parameters
        ----------
        *args
            Description
        **kargs
            Description
        """
        pass

    @classmethod
    def load_csv(cls, filename: str, delimeter='\n') -> list:
        """Load content from a CSV file

        Parameters
        ----------
        filename : str
            Name of the file
        delimeter : str, optional
            CSV delimeter used

        Returns
        -------
        list
            file content
        """
        result = None
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=delimeter)
                result = list(reader)
                pass
        except Exception as e:
            LogUtils.write("error", "load_csv: {}".format(e.args))
            traceback.print_exc()
            pass
        return result

    @classmethod
    def save_as_csv(cls, filename: str, data: list, delimeter='\n'):
        """Save data as CSV file

        Parameters
        ----------
        filename : str
            The name of the file
        data : list
            The data to save
        delimeter : str, optional
            The delimiter to use
        """
        try:
            writer = csv.writer(open(filename, 'w'))
            for d in data:
                writer.writerow(d, delimiter=delimeter)
                pass
        except Exception as e:
            LogUtils.write("error", "save_as_csv: {}".format(e.args))
            traceback.print_exc()
            pass
        pass

    @classmethod
    def write(cls, filename: str, data):
        """Write data to file

        Parameters
        ----------
        filename : str
            File name
        data : TYPE
            Data to save
        """
        with open(filename, 'wb') as writer:
            pickle.dump(data, writer, protocol=pickle.HIGHEST_PROTOCOL)
            pass
        pass

    @classmethod
    def read(cls, filename: str):
        """Read data from file

        Parameters
        ----------
        filename : str
            The file name

        Returns
        -------
        TYPE
            file content
        """
        try:
            with open(filename, 'rb') as reader:
                return pickle.load(reader)
        except Exception as e:
            LogUtils.write("error", e.args)
            traceback.print_exc()
            pass
        return None
    pass  # end of FileManager
