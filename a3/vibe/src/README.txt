Prerequisites:
	libboost-all-dev
	swig
	python-dev
	python3-dev

cd <open-swig-dir>

git clone https://github.com/renatoGarcia/opencv-swig.git

cd <vibe-src-dir>

swig -I<open-swig-dir>/lib $(pkg-config --cflags opencv) -python -c++ vibe.i

# The below commands did not work with Python 2.7 for me, so stuck with Python3
# Also, you can build the shared library in one step as well as show after OR

g++ -std=c++11 -Wall -Werror -c -fPIC vibe.cpp vibe_wrap.cxx $(pkg-config --cflags --libs python3) $(pkg-config --cflags --libs opencv)

g++ -shared vibe.o vibe_wrap.o -o _vibe.so

OR

g++ -std=c++11 -Wall -Werror -shared -fpic vibe.cpp vibe_wrap.cxx $(pkg-config --cflags --libs python3) $(pkg-config --cflags --libs opencv) -o _vibe.so

# Now go to terminal
python3
import vibe

ENJOY!
