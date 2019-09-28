#! /usr/bin/python

from glue.lal import Cache
from glue.lal import CacheEntry
import sys

for line in sys.stdin:
    c = CacheEntry.from_T050017(line)
    print str(c),

