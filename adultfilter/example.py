from adultfilter.classifier import ClassifierFilter, make_prediction_fun
from adultfilter.utils import CorpusLoader
from adultfilter.config import VOCABULARY_PATH


def example_train(train_data_dir, valid_data_dir):
    clm = ClassifierFilter()
    text_loader_train = CorpusLoader(train_data_dir)
    text_loader_train.load_vocab(VOCABULARY_PATH)
    text_loader_valid = CorpusLoader(valid_data_dir)
    text_loader_valid.load_vocab(VOCABULARY_PATH)

    clm.fit_generator(generator=text_loader_train,
                      steps_per_epoch=len(text_loader_train),
                      epochs=1000,
                      validation_data=text_loader_valid,
                      validation_steps=len(text_loader_valid))


def example_pred():
    # Dummy data
    title = 'Design by evolution – Stathis Vafeias – Medium'
    keywords = ["Machine Learning", "Evolution",
                "Deep Learning", "Automl", "Dnn"]
    description = 'How to evolve your neural network. AutoML: time to evolve.'
    body_text = """Design by evolution
    How to evolve your neural network. AutoML: time to evolve.
    The gist ( tl;dr): Time to evolve! I’m gonna give a basic
    example (in PyTorch) of using evolutionary algorithms to tune
    the hyper-parameters of a DNN.
    For most machine learning practitioners designing a neural network is
    an artform. Usually, it begins with a common architecture and then
    parameters are tweaked until a good combination of layers,
    activation functions, regularisers, and optimisation
    parameters are found. Guided by popular architectures — like VGG,
    Inception, ResNets, DenseNets and others — one will iterate
    through variations of the network until it achieves the desired balance of
    speed and accuracy. But as the available processing power increases,
    it makes sense to begin automating this network optimisation process.
    In shallow models like Random Forests and SVMs we are already able to
    automate the process of tweaking through hyper-parameter optimisation.
    Popular toolkits like sk-learn provide methods for searching the
    hyper-parameter space. At its simplest form the hyper-parameter search
    is performed through a grid search over all possible parameters or random
    sampling from a parameter distribution (read this post). These approaches
    face two problems: a) waste of resources while searching on bad parameter
    region, b) inefficient at handling a dynamic set of parameters, hence
    it’s hard to alternate the structure of the solver (i.e. the architecture
    of a DNN). More efficient methods like Bayesian optimisation deal with
    (a) but still suffer from (b). It is also hard to explore models in
    parallel in the Bayesian optimisation setting.
    More power!
    While the idea of automatically identifying the best models is not new,
    the recent large increase in processing power make it more achievable
    than ever before. Especially if the type of models we want to optimise
    are computationally hungry (e.g. DNNs).
    The time has come! And it’s so important that even Google decided to
    include it in its annual Google I/O (~16:00 min) conference, and I’m
    sure many others in the industry are doing too. It is already a
    spotlighted project in our team @ AimBrain."""

    # Load prediction method
    predict_fun = make_prediction_fun()

    # Make a prediction
    pred = predict_fun(title, keywords, description, body_text)
    # Print a results
    print(pred)

