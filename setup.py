from setuptools import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='odap',
    version='0.1.0',
    author='Reece Humphreys',
    author_email='reecewh@icloud.com',
    packages=['odap'],
    license_files=['LICENSE'],
    description='An implementation of the NASA Standard Breakup Model in Python with associated tools and demonstrations.',
    long_description=read("README.md")
)
