from setuptools import find_packages, setup

setup(
    name='better-classifiers',
    packages=find_packages(),
    version='0.1.0',
    description='ML classifiers with improvement',
    author='maxbolzy@gmail.com',
    license='MIT',
    install_requires=['scikit-learn']
)