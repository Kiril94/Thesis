from setuptools import setup, find_packages

setup(
    name='cobra',
    version='0.1.0',
    packages=find_packages(include=['cobra','cobra.*'])
)