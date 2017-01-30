#!/usr/bin/env bash

pushd `dirname $0` > /dev/null
# The absolute path to the current script directory
SCRIPTPATH=`pwd`
popd > /dev/null

# test1
cat "$SCRIPTPATH"/right_window.txt | "$SCRIPTPATH"/../a1.py "$SCRIPTPATH"/example1.mov "$SCRIPTPATH"/right_window.ogv

# produces the same video
cat "$SCRIPTPATH"/test_case2.txt | "$SCRIPTPATH"/../a1.py "$SCRIPTPATH"/example1.mov "$SCRIPTPATH"/test_case2.ogv

# tested negative resolution of output video
cat "$SCRIPTPATH"/test_case3.txt | "$SCRIPTPATH"/../a1.py "$SCRIPTPATH"/example1.mov "$SCRIPTPATH"/test_case3.ogv

