#!/bin/sh

UPDATE_CMD="${COPYRIGHT_SCRIPT_DIR}/update-copyright-header.py --copyright-header=${PWD}/Copyright.txt --file={} --script-mode"
find . -name "*.py" -exec ${UPDATE_CMD} \;
find . -name "*.pyx" -exec ${UPDATE_CMD} \;
