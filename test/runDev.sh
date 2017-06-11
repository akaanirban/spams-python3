inst=/home/anirban/Softwares/spams-python3
PYV=`python3 -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";` # get python current version
export PYTHONPATH=$inst/lib/python${PYV}/site-packages
