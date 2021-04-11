[![pipeline status](http://192.168.124.254:9080/tpeulen/clsmview/badges/master/pipeline.svg)](http://gitlab.peulen.xzy/tpeulen/clsmview/-/commits/master) 
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
``clsmview`` depends on common python packages such as ``numpy``, ``opencv``, and ``pandas``.
Additionally, ``clsmviewer`` depends on ``scikit-fluorescence``, ``tttrlib``, and ``imp``. Thus,
to install ``clsmviewr`` make sure that conda channels that provide packages for the necessary
dependencies are listed in the ``.condarc`` file 

```yaml
channels:
  - salilab
  - tpeulen
  - tpeulen/label/nightly
  - conda-forge
  - defaults
```

To avoid potential conflicts ``clsmview`` can be installed in a separate environment. 

```commandline
conda create -n test
conda install clsmview
```


[1]: doc/gui.png "ndxplorer GUI"
