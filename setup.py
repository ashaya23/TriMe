#coding=utf-8
from setuptools import setup

REQUIRED_PACKAGES = [
    'scikit-learn',
    'optuna',
]

OPTIONAL_PACKAGES = {
    'xgboost': ['xgboost'],
    'catboost': ['catboost'],
}

setup(
    name='TriMe',
    version='0.1.0',
    author='ashaya',
    author_email='',
    description='Tree based Multiple Imputation',
    url="https://github.com/ashaya23/redis-filemem-cache",
    python_requires='>=3.6',
    packages=['trime'],
    install_requires=REQUIRED_PACKAGES,
    extras_require=OPTIONAL_PACKAGES,
    install_requires=[line.strip() for line in openf("requirements.txt") if line.strip()],
    classifiers=[
        'Topic :: Utilities',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python'
    ],
)
