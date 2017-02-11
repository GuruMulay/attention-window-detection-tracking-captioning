#!/usr/bin/env bash

pushd `dirname $0` > /dev/null
# The absolute path to the current script directory
SCRIPTPATH=`pwd`
popd > /dev/null

cat "$SCRIPTPATH"/"$1".txt | "$SCRIPTPATH"/../a1.py "$SCRIPTPATH"/example1.ogv "$SCRIPTPATH"/"$1$2".ogv "$2"
