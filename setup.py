from setuptools import setup, find_packages

setup(
    name='pandas_commons',
    version='0.0.1',
    packages=find_packages(),
    author='Eli Hubscher',
    author_email='eliyahu.hubscher@icloud.com',
    description='Pandas helper library.',
    keywords='pandas helper library',
    project_urls={'Source Code': 'https://github.com/ehubscher/pandas_commons'},
    install_requires=[
        'openpyxl',
        'pandas',
        'regex',
        'xlrd'
    ]
)
