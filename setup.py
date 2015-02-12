import os
import setuptools

setuptools.setup(
    name='theanets',
    version='0.5.1',
    packages=setuptools.find_packages(),
    author='Leif Johnson',
    author_email='leif@lmjohns3.com',
    description='A library of neural nets in theano',
    long_description=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.rst')).read(),
    license='MIT',
    url='http://github.com/lmjohns3/theanets',
    keywords=('machine-learning '
              'neural-network '
              'deep-neural-network '
              'recurrent-neural-network '
              'autoencoder '
              'sparse-autoencoder '
              'classifier '
              'theano '
              ),
    install_requires=['theano', 'climate'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )
