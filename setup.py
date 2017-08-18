from setuptools import setup

setup(name='cnn-adult-filter',
      version='0.1',
      description='Adult content filter by NN',
      url='https://github.com/it-svit/cnn-adult-filter',
      author='Illia Zinkevych',
      author_email='ilya.zinckevich@gmail.com',
      packages=['adultfilter'],
      # FIXME: tensorflow not found at remote repo. Need to install manually
      install_requires=['numpy', 'tensorflow', 'keras>=2.0.5', 'h5py'],
      python_requires='>=3',
      package_data={
          'adultfilter': ['data/embeddings.pickle',
                          'data/vocabulary.json',
                          'models/trained_model.h5py'],
      })