#!/bin/bash
source $HOME/rift_ghlaplace_20260902/devnotes/env.sh
cd $HOME/rift_ghlaplace_20260902/devnotes
out=$1; shift
exec $PY "$@" > "$out" 2>&1
