inst=/home/anirban/Softwares/spams-python3
python3 setup.py install --prefix=$inst
PYV=`python3 -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";` 
export PYTHONPATH=$inst/lib/python${PYV}/site-packages
