"""
pylibnxc
Machine learned density functionals
"""
#from setuptools import setup
from numpy.distutils.core import setup

#short_description = __doc__.split("\n")
short_description = "Enables the use of machine learned density functionals in electronic structure codes"

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:]),

setup(
    # Self-descriptive entries which should always be present
    name='pylibnxc',
    author='Sebastian Dick',
    author_email='sebastian.dick@stonybrook.edu',
    description=short_description[0],
    version=0.1,
    license='BSD-3-Clause',

    # Which Python importable modules should be included when your package is installed
    packages=['pylibnxc'],

    # Optional include package data to ship with your package
    # Comment out this line to prevent the files from being packaged with your software
    # Extend/modify the list to include/exclude other items as need be
    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # author_email='me@place.org',      # Author email
    # url='http://www.my_package.com',  # Website
    install_requires=["torch>=1.6.0",
                      "pyscf>=1.7",
                      "pytest>=1.6"],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor
    python_requires=">=3.6",          # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,
)
