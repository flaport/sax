#! /usr/bin/env python

import sys
from subprocess import check_call

args = sys.argv[1:]

check_call(["isort", "--profile", "black", *args])
check_call(["black", *args])
