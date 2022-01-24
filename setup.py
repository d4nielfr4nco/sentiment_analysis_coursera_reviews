from setuptools import setup, find_packages

setup(
    name='sentiment_analysis_coursera_reviews',
    version='0.1.0',
    packages=find_packages(include=['sentiment_analysis_coursera_reviews']),
    install_requires=[
        'pandas>=1.3.5',
        'numpy>=1.22.1',
        'nltk>=3.6.7',
        'scikit-learn>=1.0.2'
    ],
    extras_require={
        'interactive': ['matplotlib>=3.5.1', 'jupyter>=1.0.0'],
    }
)