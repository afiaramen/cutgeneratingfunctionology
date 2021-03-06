## -*- encoding: utf-8 -*-
import os
import sys
from setuptools import setup
from codecs import open # To open the README file with proper encoding
from setuptools.command.test import test as TestCommand # for tests


# Get information from separate files (README, VERSION)
def readfile(filename):
    with open(filename,  encoding='utf-8') as f:
        return f.read()

# For the tests
class SageTest(TestCommand):
    def run_tests(self):
        errno = os.system("make check")
        if errno != 0:
            sys.exit(1)

setup(
    name = "cutgeneratingfunctionology",
    version = readfile("VERSION").strip(), # the VERSION file is shared with the documentation
    description='An example of a basic sage package',
    long_description = readfile("README.rst"), # get the long description from the README
    url='https://github.com/mkoeppe/cutgeneratingfunctionology',
    author='Matthias Koeppe, Yuan Zhou, Chun Yu Hong, Jiawei Wang, with contributions by undergraduate programmers',
    author_email='mkoeppe@math.ucdavis.edu', # choose a main contact email
    license='GPLv2+', # This should be consistent with the LICENCE file
    classifiers=[
      # How mature is this project? Common values are
      #   3 - Alpha
      #   4 - Beta
      #   5 - Production/Stable
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering :: Mathematics',
      'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
      'Programming Language :: Python :: 2.7',
    ], # classifiers list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    keywords = "integer programming, cutting planes",
    packages = ['cutgeneratingfunctionology', 'cutgeneratingfunctionology.igp', 'cutgeneratingfunctionology.multirow', 'cutgeneratingfunctionology.dff'],
    include_package_data=True,     # to install the .sage files too
    cmdclass = {'test': SageTest}, # adding a special setup command for tests
    setup_requires   = ['sage-package'],
    install_requires = ['sage-package', 'sphinx', 'sphinxcontrib-bibtex'],
)
