#!/usr/bin/python
try:
    import numpy as np
except ImportError:
    np = None
from setuptools import setup, find_packages
from clsmview import info

NAME = info.__name__
VERSION = info.__version__
AUTHOR = info.__author__
LICENSE = info.__license__
DESCRIPTION = info.__description__
LONG_DESCRIPTION = info.LONG_DESCRIPTION
URL = info.__url__
EMAIL = info.__email__
APP_ID = info.__app_id__


def dict_from_txt(fn):
    d = {}
    with open(fn) as f:
        for line in f:
            (key, val) = line.split()
            d[str(key)] = val
    return d


gui_scripts = dict_from_txt("./clsmview/entry_points/gui.txt")
console_scripts = dict_from_txt("./clsmview/entry_points/cmd.txt")


metadata = dict(
    name=NAME,
    version=VERSION,
    license=LICENSE,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    app_id=APP_ID,
    url=URL,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
    keywords='fluorescence microscopy CLSM',
    packages=find_packages(
        include=(NAME + "*",)
    ),
    package_dir={
        NAME: NAME
    },
    include_package_data=True,
    package_data={
        '': [
            '*.json', '*.yaml',
            '*.ui', '*.css', '*.qss',
            '*.png', '*.svg',
            '*.csv', '*.npy', '*.dat',
            '*.dll', '*.so', '*.pyd'
        ]
    },
    install_requires=[
        'PyQt5', 'qtpy', 'sip', 'pyqtgraph',
        'numpy',
        'PyYAML',
        'typing-extensions',
        'opencv-python'
    ],
    setup_requires=[
        'numpy',
        'PyYAML',
        'setuptools'
    ],
    entry_points={
        "console_scripts": [
            # There is currently no console interface
            # "%s=%s" % (key, console_scripts[key]) for key in console_scripts
        ],
        "gui_scripts": [
            "%s=%s" % (key, gui_scripts[key]) for key in gui_scripts
        ]
    }
)

setup(**metadata)
