#!/usr/bin/env python3

from distutils.core import setup

setup(
    name='Ephem',
    version='0.1',
    description='Ephemerides',
    author='Eivind Fonn',
    author_email='evfonn@gmail.com',
    license='GPL3',
    url='https://github.com/TheBB/butter',
    py_modules=['ephem'],
    entry_points={
        'console_scripts': ['ephem=ephem.__main__:main'],
    },
    install_requires=[
        'click',
        'requests',
        'vispy',
        'numpy',
    ],
)
