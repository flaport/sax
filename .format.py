#! /usr/bin/env python

import sys
from subprocess import check_call

args = sys.argv[1:]
autoimport_args = [arg for arg in args if not "__" in arg]  # __init__.py, __main__.py

check_call(["autoimport", *autoimport_args])
check_call(["isort", *args])
check_call(["black", *args])
