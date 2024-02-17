![conda build](https://github.com/fluorescence-tools/clsmview/actions/workflows/conda-build.yml/badge.svg)
[![Anaconda-Server Version](https://anaconda.org/tpeulen/clsmview/badges/version.svg)](https://anaconda.org/tpeulen/clsmview)

# clsmview

``clsmview`` is an application to work with time-tagged time-resolved (TTTR) confocal laser scanning (CLSM) data. ``clsmview`` 
can be used to

 * visualize and convert TTTR CLSM images 
 * create fluorescence decays based on pixel selections and masks 
 * and as a tool integrated in other software.

![clsmview GUI][1]


## Building and installation
``clsmview`` can be installed from the source code in an python environment that
resolves all necessary dependencies. Alternatively, ``clsmview`` can be installed 
via conda.

### Source
To install ``clsmview`` from the source code clone the git repository and run the
setup script.
```commandline
git clone https://gitlab.peulen.xyz/tpeulen/clsmview
cd clsmviewer
python setup.py install
```
After installing ``clsmview`` you can open the GUI from the commandline.

```commandline
clsmview
```

### Conda
``clsmview`` can be installed using conda (best in a separate environment)

```bash
conda create clsmview
conda activate clsmview
mamba install clsmview -c tpeeulen
```

To run `clsmview` activate its environment and use the `clsmview` command

```bash
conda activate clsmview
clsmview
```


[1]: doc/gui.png "clsmview GUI"
