
from setuptools import setup

try:
    with open("README.md", 'r') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ''

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='BatAnalysis',
    version='0.0.1',
    packages=['batanalysis'],
    url='https://sed-gitlab.gsfc.nasa.gov/swift-bat-codes/scientific-product-analysis/',
    license='BSD-3-Clause',
    author='Tyler Parsotan and Sibasish Laha',
    author_email='tyler.parsotan@nasa.gov',
    description='Routines for analyzing data from BAT on the Neil Gehrels Swift Observatory',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=['Development Status :: 3 - Alpha', 'Intended Audience :: Science/Research', 'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy', ],
    python_requires='>=3.8',
    install_requires=required,
    data_files=[
        ('data',['bat_analysis/data/survey6b_2.cat']),
        ],
    include_package_data=True,
)
