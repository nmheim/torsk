from setuptools import setup, find_packages
import os

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as fp:
    install_requires = fp.read()

setup(
    name='torsk',
    description='ESN implementation in PyTorch',
    author='Niklas Heim',
    author_email='heim.niklas@gmail.com',
    packages=find_packages(),
    version=0.1,
    install_requires=install_requires,
)
