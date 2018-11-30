from setuptools import setup, find_packages
from os import path

setup(
    name='seq2struct',
    version='0.0.1',
    description='seq2struct',
    packages=find_packages(exclude=["*_test.py", "test_*.py"]),
    install_requires=[
        'asdl~=0.1.5',
        'torch~=0.4.0',
    ],
)
