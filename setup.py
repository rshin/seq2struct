from setuptools import setup, find_packages
from os import path

setup(
    name='seq2struct',
    version='0.0.1',
    description='seq2struct',
    packages=find_packages(exclude=["*_test.py", "test_*.py"]),
    install_requires=[
        'asdl~=0.1.5',
        'astor~=0.7.1',
        'attrs~=18.2.0',
        'cython~=0.29.1',
        'jsonnet~=0.11.2',
        'nltk~=3.4',
        'numpy~=1.15.4',
        'pyrsistent~=0.14.9',
        'torch~=0.4.0',
        'tqdm~=4.28.1',
    ],
)
