"""
This module contains classes and functions for training and
testing classifiers
"""

from _decimal import Decimal
from _decimal import ROUND_UP
import numpy as np
from collections import OrderedDict
from enum import Enum
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics.classification import confusion_matrix
from sklearn.metrics.ranking import coverage_error
from sklearn.metrics.ranking import label_ranking_average_precision_score
from sklearn.metrics.ranking import label_ranking_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import libsvm
from sklearn.svm import LinearSVC
from Utilities import FileManager

_author__ = "Nathan Zwelibanzi Khupe"
__copyright__ = "Copyright 2017, Nathan Zwelibanzi Khupe"
__credits__ = ["Nathan Zwelibanzi Khupe"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nathan Zwelibanzi Khupe"
__email__ = "001knatz@gmail.com"
__docformat__ = 'numpy'


class ClassifierType(Enum):
    """Defines accepted classifier type as required in this application

    Attributes
    ----------
    BinaryGenres : int
        Genre is either Fantasy or Non-Fantasy
    MultiClass : int
        Genre can be multiple classes from dark fantasy to juvenile fantasy
    NoClass : int
        Default genre this should not result in any processing
    """
    BinaryGenres = 1
    MultiClass = 2
    NoClass = 3
    pass  # end of ClassifierType

class GenreClassifier(object):
    """Container for genre classification operations
    """

    def __init__(self, **kwargs):
        """Constructor for :class:'GenreClassifier'

        Parameters
        ----------
        **kwargs
            Default initialiser arguments
        """
        return super().__init__(**kwargs)

    @classmethod
    def train_MainGenre_on(cls, train_documents: list, train_genres: list):
        """Train the main genre filter classifier on a list of labelled
        unstructured text documents.

        Parameters
        ----------
        train_documents : list
            List of unstructured text documents for training on.
        train_genres : list
            List of predict labelled genres in binary form

        Returns
        -------
        Pipeline
            The trained classifier
        list
            list of word features, word counts and word tfidf_scores

        Raises
        ------
        ValueError
            The number of documents must equal the number of genre labels

        Notes
        -----
        Use Multilabelbinarizer to convert string genres to binary format
        """
        if len(train_documents) != len(train_genres):
            raise ValueError("Documents list and genre mapping must be of the same length")

        bow = CountVectorizer(binary=True,
                              max_features=7000,
                              stop_words='english',
                              strip_accents='unicode',
                              ngram_range=(1, 1))

        classifier = Pipeline([('vectorizer', bow),
                               ('tfidf', TfidfTransformer()),
                               ('clf', LinearSVC()),
                               ('calib', CalibratedClassifierCV())])

        classifier.fit(train_documents, train_genres)

        bow = classifier.named_steps['vectorizer']
        features = bow.get_feature_names()
        counts = bow.transform(train_documents).toarray().sum(axis=0)
        tfidf = classifier.named_steps['tfidf']
        tfidf_scores = tfidf.idf_

        sif_feat = zip(features, counts, tfidf_scores)
        sif_feat = sorted(sif_feat, key=lambda t: t[1], reverse=True)

        return classifier, sif_feat

    @classmethod
    def predict_MainGenre(cls, raw_text: str) -> list:
        """Predict the main genre label on an unstructured text document.

        Parameters
        ----------
        raw_text : str
            The unstructured text document

        Returns
        -------
        list
            A list of classes and their probability distributions
        """
        classifier = cls.load(ClassifierType.BinaryGenres)

        pscores = classifier.predict_proba([raw_text])
        pscores = [float(Decimal(p).quantize(Decimal('1.111'),
                                             rounding=ROUND_UP)) for p in pscores[0]]
        prob_map = list(zip(classifier.classes_, pscores))

        return prob_map

    @classmethod
    def train_SubGenres_on(cls, documents: list, genreLabels: list):
        """Train the sub genre classifier on a list of labelled
        unstructured text documents.

        Parameters
        ----------
        documents : list
            List of unstructured text documents for training on.
        genreLabels : list
            List of predict labelled genres in binary form

        Returns
        -------
        Pipeline
            The trained classifier
        list
            list of word features and word counts

        Notes
        -----
        Use Multilabelbinarizer to convert string genres to binary format
        """
        algorithm, classifier = None, None

        algorithm = MultinomialNB()
        bow = CountVectorizer(max_features=7000,
                              stop_words='english',
                              strip_accents='unicode',
                              ngram_range=(1, 2))

        classifier = Pipeline([('vectorizer', bow), ('clf', OneVsRestClassifier(algorithm))])

        classifier.fit(documents, genreLabels)

        bow = classifier.named_steps['vectorizer']
        features = bow.get_feature_names()
        counts = bow.transform(documents).toarray().sum(axis=0)

        sif_feat = zip(features, counts)
        sif_feat = sorted(sif_feat, key=lambda t: t[1], reverse=True)

        return classifier, sif_feat

    @classmethod
    def predict_SubGenre(cls, raw_text: str) -> list:
        """Predict the sub genre label(s) on an unstructured text document.

        Parameters
        ----------
        raw_text : str
            The unstructured text document

        Returns
        -------
        list
            A list of classes and their probability distributions

        Notes
        -----
        Given text must have a significant portion as fantasy
        """
        lblbinarizer = FileManager.read("MultiLabelBinarizer.pickle")

        classifier = cls.load(ClassifierType.MultiClass)
        #p = classifier.predict([raw_text])
        #return lblbinarizer.inverse_transform(p)
        pscores = classifier.predict_proba([raw_text])
        pscores = [float(Decimal(p).quantize(Decimal('1.111'),
                                             rounding=ROUND_UP)) for p in pscores[0]]
        prob_map = list(zip(lblbinarizer.classes_, pscores))

        return prob_map

    @classmethod
    def save(cls, classifier: ClassifierMixin, type: ClassifierType):
        """Save the trained classifier for use in prediction

        Parameters
        ----------
        classifier : ClassifierMixin
            A scikit learn classifier
        type : ClassifierType
            The type of classifier from the enum :class:`ClassifierType`

        Notes
        -----
        Sklearn classifiers must implement :class:`ClassifierMixin`
        """
        file = ""
        if type is ClassifierType.BinaryGenres:
            file = 'clsf_main_genre.pickle'
            pass
        elif type is ClassifierType.MultiClass:
            file = 'clsf_sub_genre.pickle'
            pass

        joblib.dump(classifier, file, compress=1)

        pass

    @classmethod
    def load(cls, type: ClassifierType) -> ClassifierMixin:
        """Load a classifier that has been saved with :func:`save`

        Parameters
        ----------
        type : ClassifierType
            The type of classifier from the enum :class:`ClassifierType`

        Returns
        -------
        ClassifierMixin
            A scikit learn classifier

        Notes
        -----
        * Sklearn classifiers must implement :class:`ClassifierMixin`
        * Use the same classifier type as used during saving
        """
        file = ""
        classifier = None
        if type is ClassifierType.BinaryGenres:
            file = 'clsf_main_genre.pickle'
            pass
        elif type is ClassifierType.MultiClass:
            file = 'clsf_sub_genre.pickle'
            pass
        classifier = joblib.load(file)
        return classifier

    @classmethod
    def binary_class_measures(cls, y_true: list, y_predicted: list) -> OrderedDict:
        """Assessment measures of a classification task with binary
        classes i.e. Fantasy/Non-Fantasy

        Parameters
        ----------
        y_true : list
            Expected class labels in binary form
        y_predicted : list
            Predicted class labels in binary form

        Returns
        -------
        OrderedDict
            An ordered dictionary of assessment measures
        """
        cm = confusion_matrix(y_true, y_predicted)
        tp, fp, fn, tn = cm.flatten()
        measures = OrderedDict()
        measures['accuracy'] = (tp + tn) / (tp + fp + fn + tn)
        measures['specificity'] = tn / (tn + fp)
        measures['sensitivity'] = tp / (tp + fn)
        measures['precision'] = tp / (tp + fp)
        measures['f1score'] = 2 * tp / (2 * tp + fp + fn)
        return measures

    @classmethod
    def multi_class_measures(cls, y_true: list, y_predicted: list) -> OrderedDict:
        """Assessment measures of a classification task with multiple
        classes i.e. multi-label and or multi-class task

        Parameters
        ----------
        y_true : list
            Expected class labels in binary form
        y_predicted : list
            Predicted class labels in binary form

        Returns
        -------
        OrderedDict
            An ordered dictionary of assessment measures
        """
        measures = OrderedDict()
        measures['accuracy'] = accuracy_score(y_true, y_predicted)
        measures['coverage error'] = coverage_error(y_true, y_predicted)
        measures['label ranking loss'] = label_ranking_loss(y_true, y_predicted)
        b_true = np.array(y_true)
        b_pred = np.array(y_predicted)
        measures['unsupported hamming loss'] = np.sum(np.not_equal(b_true, b_pred)) / float(b_true.size)
        measures['label ranking average precision'] = label_ranking_average_precision_score(y_true, y_predicted)
        return measures
    pass  # end of GenreClassifier
