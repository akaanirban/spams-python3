# SPAMS 2.6 and python3.x

This directory contains files to install and use (at the end) the python interfaces to the functions of SPAMS library already interfaced with matlab.

Manipulated objects are imported from numpy and scipy. Matrices should be stored by columns, and sparse matrices should be "column compressed".

The python SPAMS package consists of 4 files:
* `spams.py`
* `myscipy_rand.py`
* `spams_wrap.py`
* `_spams_wrap.so`

that should be in the path of the python interpreter (for instance in the current directory).

**NOTE:** myscipy_rand.py supplies a random generator for sparse matrix
      for some old scipy distributions

**WARNING:** the API of spams.OMP and spams.OMPMask has changed since version V2.2

Available functions in python are defined in `spams.py` and documented (the doc is extracted from matlab files).

This file describes how to directly install the interface from sources.

Porting to python3.x based on https://aur.archlinux.org/packages/python-spams-svn/

## INTERFACE INSTALLATION (python3.x) for LINUX and MacOS

### Installation

Packages required: python3-numpy, python3-scipy, blas + lapack (preferably from atlas).

```
tar zxf spams-python3-v2.6-2017-06-06.tar.gz
cd spams-python3
python3 setup.py build

inst=<your-python-install-dir>
python3 setup.py install --prefix=$inst
```

Two documentations are installed in `$inst/doc`:
* doc_spams.pdf and html/index.html : the detailed user documentation
* sphinx/index.html : the documentation of python function extracted by sphinx

### Testing the interface :

```
PYV=`python3 -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";` # get python current version
export PYTHONPATH=$inst/lib/python${PYV}/site-packages
cd $inst/test
python3 test_spams.py -h # to get help
python3 test_spams.py  # will run all the tests
python3 test_spams.py linalg # test of linalg functions
python3 test_spams.py name1 name2 ... # run named tests
```

### Comments
#### Linux:
Carefully install atlas. For example on ubuntu, necessary to `apt-get install libatlas-dev libatlas3gf-base libatlas-3gf.so`

If you don't have libblas.so and liblapack.so in /lib or /usr/lib, you need to edit `setup.py`

## INITIAL START
Run the `source ./runDev.sh` from inside `~/spams-python3` everytime you start the session to set the `PYTHONPATH` environment variable.
### 
On Ubuntu 16.04 make sure you have the packages

`liblapack-dev`
`liblapack3`
`libopenblas-base`
`libopenblas-dev`
installed. After that, "-L/usr/lib -llapack -lblas" should work.

#### MacOS:
TODO
<!-- The installation has been tested with MacOS 10 (Lion), it required that packages were installed with `port install`:
```
port install atlas;port install py26-numpy;install py26-scipy
```
Maybe necessary to add `/opt/local/bin` to `PATH` and specified the compiler by setting CC and CXX, for example:
```
export CC=/opt/local/bin/gcc-mp-4.3;export CXX=/opt/local/bin/g++-mp-4.3
``` -->

## INTERFACE INSTALLATION (python2.x) for Windows

TODO

### Installation of the binary windows packages

### Testing the interface (binary install)
