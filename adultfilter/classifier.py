import os
import logging
import pickle

os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.models import Model, load_model
from keras.layers import Convolution1D, Dense, Dropout, Embedding, Flatten, \
    Input, MaxPooling1D
from keras.layers.merge import Concatenate
import numpy as np

from adultfilter.utils import TextConverter
from adultfilter.config import MODEL_PATH, WORD2VEC_EMBEDDINGS, VOCABULARY_PATH, \
    TITLES_PAD, KEYWORDS_PAD , DESCRIPTION_PAD, TEXT_PAD,\
    LABEL_NUMBER, IDX_LABEL


class ClassifierFilter:
    """
    Classifier class.

    Contains model creation method and fit + predict methods.
    """

    def __init__(self):
        """
        Init classifier.

        Loads model from .h5py file. If not exists - construct new.
        """
        if os.path.exists(MODEL_PATH):
            logging.info(
                u'Loading pre-trained model from {}'.format(MODEL_PATH)
            )
            self.model = load_model(MODEL_PATH)
        else:
            logging.warning(
                u'Pre-trained model not found. Constructing new one.'
            )
            with open(WORD2VEC_EMBEDDINGS, 'rb') as f:
                embeddings = pickle.load(f)
            self.construct_model(
                embeddings=embeddings,
                title_n_features=TITLES_PAD,
                kw_n_features=KEYWORDS_PAD,
                desc_n_features=DESCRIPTION_PAD,
                text_n_features=TEXT_PAD,
                n_labels=LABEL_NUMBER
            )

    def _construct_model(self,
                        embeddings,
                        title_n_features,
                        kw_n_features,
                        desc_n_features,
                        text_n_features,
                        n_labels):
        """
        Model constructor.

        Construct CNN classifier
        :param embeddings: [list of lists of floats] n*m matrix of words embs
        :param n_features: [int] number of processed words
        :param n_labels: [int] number of possible categories
        """
        logging.info(u'Starting to construct model.')
        input_title = Input(shape=(title_n_features,), name='title_input')
        input_kw = Input(shape=(kw_n_features,), name='keywords_input')
        input_desc = Input(shape=(desc_n_features,), name='description_input')
        input_text = Input(shape=(text_n_features,), name='text_input')

        # pre-trained word embeddings layer
        # trainable=False to keep the embeddings fixed
        embedding_layer = Embedding(embeddings.shape[0],
                                    embeddings.shape[1],
                                    weights=[embeddings],
                                    trainable=False, name='words_embeddings')
        title_emb = embedding_layer(input_title)
        kw_emb = embedding_layer(input_kw)
        desc_emb = embedding_layer(input_desc)
        text_emb = embedding_layer(input_text)

        title = Dense(units=64,
              activation=None,
              use_bias=True,
              kernel_initializer='glorot_uniform',
              bias_initializer='zeros')(title_emb)
        kw = Dense(units=32,
              activation=None,
              use_bias=True,
              kernel_initializer='glorot_uniform',
              bias_initializer='zeros')(kw_emb)
        desc = Dense(units=32,
              activation=None,
              use_bias=True,
              kernel_initializer='glorot_uniform',
              bias_initializer='zeros')(desc_emb)
        text = Dense(units=4,
              activation=None,
              use_bias=True,
              kernel_initializer='glorot_uniform',
              bias_initializer='zeros')(text_emb)

        title = Flatten()(title)
        kw = Flatten()(kw)
        desc = Flatten()(desc)
        text = Flatten()(text)

        title = Dense(units=128,
              activation=None,
              use_bias=True,
              kernel_initializer='glorot_uniform',
              bias_initializer='zeros')(title)
        kw = Dense(units=128,
              activation=None,
              use_bias=True,
              kernel_initializer='glorot_uniform',
              bias_initializer='zeros')(kw)
        desc = Dense(units=128,
              activation=None,
              use_bias=True,
              kernel_initializer='glorot_uniform',
              bias_initializer='zeros')(desc)
        text = Dense(units=128,
              activation=None,
              use_bias=True,
              kernel_initializer='glorot_uniform',
              bias_initializer='zeros')(text)

        merge_layer = Concatenate(name='merge_concatenate')([title,
                                                             kw,
                                                             desc,
                                                             text])
        dense = Dense(256,
                      activation=None,
                      use_bias=True,
                      name='dense')(merge_layer)

        dropout1 = Dropout(0.25, name='dropout_1')(dense)
        dense = Dense(units=128,
              activation=None,
              use_bias=True,
              kernel_initializer='glorot_uniform',
              bias_initializer='zeros')(dropout1)
        dropout2 = Dropout(0.1, name='dropout_2')(dense)
        softmax = Dense(n_labels,
                        activation='softmax',
                        name='output_layer')(dropout2)
        model = Model(inputs=[input_title, input_kw, input_desc, input_text],
                      outputs=[softmax])
        model.compile(loss='binary_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])
        logging.info(model.summary())
        self.model = model
        return self


    def construct_model(self, embeddings, title_n_features,
                        kw_n_features, desc_n_features,
                        text_n_features, n_labels):
        """
        Model constructor.

        Construct CNN classifier
        :param embeddings: [list of lists of floats] n*m matrix of words embs
        :param n_features: [int] number of processed words
        :param n_labels: [int] number of possible categories
        """
        logging.info(u'Starting to construct model.')
        input_title = Input(shape=(title_n_features,), name='title_input')
        input_kw = Input(shape=(kw_n_features,), name='keywords_input')
        input_desc = Input(shape=(desc_n_features,), name='description_input')
        input_text = Input(shape=(text_n_features,), name='text_input')

        # pre-trained word embeddings layer
        # trainable=False to keep the embeddings fixed
        embedding_layer = Embedding(embeddings.shape[0],
                                    embeddings.shape[1],
                                    weights=[embeddings],
                                    trainable=False, name='words_embeddings')
        title_emb = embedding_layer(input_title)
        kw_emb = embedding_layer(input_kw)
        desc_emb = embedding_layer(input_desc)
        text_emb = embedding_layer(input_text)

        title_dense = Convolution1D(filters=128,
                                    kernel_size=3,
                                    activation='relu',
                                    use_bias=True,
                                    name='title_conv')(title_emb)
        title_dense = MaxPooling1D(pool_size=2,
                                   name='title_maxpool')(title_dense)
        title_dense = Flatten(name='title_flatten')(title_dense)

        kw_dense = Convolution1D(filters=64,
                                 kernel_size=3,
                                 activation='relu',
                                 use_bias=True,
                                 name='kw_conv')(kw_emb)
        kw_dense = MaxPooling1D(pool_size=2,
                                name='kw_maxpool')(kw_dense)
        kw_dense = Flatten(name='kw_flatten')(kw_dense)

        desc_dense = Convolution1D(filters=64,
                                   kernel_size=3,
                                   activation='relu',
                                   use_bias=True,
                                   name='description_conv')(desc_emb)
        desc_dense = MaxPooling1D(pool_size=2,
                                  name='description_maxpool')(desc_dense)
        desc_dense = Flatten(name='description_flatten')(desc_dense)

        text_dense = Convolution1D(filters=128,
                                   kernel_size=3,
                                   activation='relu',
                                   use_bias=True,
                                   name='text_conv_1')(text_emb)
        text_dense = MaxPooling1D(pool_size=2, name='text_maxpool_1')(text_dense)
        text_dense = Convolution1D(filters=16,
                                   kernel_size=3,
                                   activation='relu',
                                   use_bias=True,
                                   name='text_conv_2')(text_dense)
        text_dense = MaxPooling1D(pool_size=2,
                                  name='text_maxpool_2')(text_dense)
        text_dense = Flatten(name='text_flatten')(text_dense)

        merge_layer = Concatenate(name='merge_concatenate')([title_dense,
                                                             kw_dense,
                                                             desc_dense,
                                                             text_dense])
        dropout1 = Dropout(0.25,
                           name='dropout_1')(merge_layer)
        dense = Dense(512,
                      activation='relu',
                      use_bias=True,
                      name='dense')(dropout1)
        dropout2 = Dropout(0.1, name='dropout_2')(dense)
        softmax = Dense(n_labels,
                        activation='softmax',
                        name='output_layer')(dropout2)

        model = Model(inputs=[input_title, input_kw, input_desc, input_text],
                      outputs=[softmax])
        model.compile(loss='binary_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])
        logging.info(model.summary())
        self.model = model
        return self


    def fit_generator(self,generator,
                      steps_per_epoch,
                      epochs=1000,
                      verbose=1,
                      callbacks=None,
                      validation_data=None,
                      validation_steps=None,
                      class_weight=None,
                      max_queue_size=8,
                      workers=4,
                      use_multiprocessing=True,
                      initial_epoch=0
                      ):
        """
        Fit classifier on generator of batch of documents.

        It's redeclared Keras 2 function fit_generator
        :param generator: Batch document generator.
        :param epochs: Number of epochs.
        :param steps_per_epoch: how many bathes.
        :param verbose: Verbosity mode, 0, 1, or 2.
        :param callbacks: List of callbacks to be called during training.
        :param validation_data: This can be either:
                            - A generator for the validation data;
                            - A tuple (inputs, targets);
                            - A tuple (inputs, targets, sample_weights).
        :param validation_steps: Only relevant if `validation_data`
                            is a generator.
                            Number of steps to yield from validation generator
                            at the end of every epoch. It should typically
                            be equal to the number of unique samples of your
                            validation dataset divided by the batch size.
        :param class_weight: Dictionary mapping class indices to a weight
                                    for the class.
        :param max_queue_size: Maximum size for the generator queue
        :param workers: Maximum number of processes to spin up
        :param use_multiprocessing: if True, use process based threading.
                                Note that because
                                this implementation relies on multiprocessing,
                                you should not pass
                                non picklable arguments to the generator
                                as they can't be passed
                                easily to children processes.
        :param initial_epoch: Epoch at which to start training
                                (useful for resuming a previous training run)
        :return: self
        """
        logging.info(u'Training the model.')
        # Because training of NN is slow,
        # we save our trained model every 5 epochs.
        # If epochs less than 5 - do all epochs and than save trained model
        if epochs < 5:
            steps_per_save = epochs
        else:
            steps_per_save = 5
        for n_epoch in range(initial_epoch, epochs, steps_per_save):
            self.model.fit_generator(
                generator=generator,
                steps_per_epoch=steps_per_epoch,
                epochs=n_epoch+steps_per_save,
                verbose=verbose,
                callbacks=callbacks,
                validation_data=validation_data,
                validation_steps=validation_steps,
                class_weight=class_weight,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                initial_epoch=initial_epoch
            )
            logging.info(u'Saving model on epoch {}'.format(n_epoch +
                                                            steps_per_save))
            self.model.save(MODEL_PATH)

        return self

    def predict(self, list_of_features):
        """
        Make a prediction of

        :param list_of_features:
        :return:
        """

        preds = self.model.predict(list_of_features,
                                   batch_size=1,
                                   verbose=0)
        return IDX_LABEL[np.argmax(preds[0])], max(preds[0])


def make_prediction_fun():
    """
    Create prediction function from main classes.

    :return: prediction function
    """
    cml = ClassifierFilter()
    converter = TextConverter()
    converter.load_vocab(VOCABULARY_PATH)
    def _transform_predict(title, keywords, description, body_text):
        """
        Transform document into sequences, do classification and return result.

        :param title: [str] Title of document.
        :param keywords: [list of str] Keywords of documents.
        :param description: [str] Description of document.
        :param body_text: [str] Main (body) text of document.
        :return: [str] Predicted class of document.
        """
        inputs = [
            converter.text2seq(title,
                               tokenize=True,
                               pad=TITLES_PAD),
            converter.text2seq(' '.join(keywords),
                               tokenize=True,
                               pad=KEYWORDS_PAD),
            converter.text2seq(description,
                               tokenize=True,
                               pad=DESCRIPTION_PAD),
            converter.text2seq(body_text,
                               tokenize=True,
                               pad=TEXT_PAD)
        ]
        label, prob = cml.predict(inputs)
        if prob <= 0.7:
            logging.warning(
                'Document with title "{}" have {} prob'.format(title, prob)
            )
        return label
    return _transform_predict
