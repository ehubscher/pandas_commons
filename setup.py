from setuptools import setup, find_packages

setup(
    name='pandas_commons',
    version='0.0.1',
    packages=find_packages(),
    author='Jordan Hubscher',
    author_email='jordan.hubscher@gmail.com',
    description='Provides common utility functions and constants relevant to Pandas.',
    keywords='pandas commons utility library',
    project_urls={'Source Code': 'https://github.com/jhubscher/pandas_commons'},
    install_requires=[
        'openpyxl',
        'pandas',
        'regex',
        'xlrd'
    ]
)
