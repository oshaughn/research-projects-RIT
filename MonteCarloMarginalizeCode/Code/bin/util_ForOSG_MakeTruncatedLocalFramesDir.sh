#!/bin/bash
# The Python implementation validates channels/times and keeps the input cache
# intact if any frame fails to crop. Preserve this entry point for existing jobs.
exec util_ForOSG_MakeTruncatedLocalFramesDir.py "$@"
