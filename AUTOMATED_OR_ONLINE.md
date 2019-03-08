
# RIFT automated or online operation

This document describes the current online and/or automated RIFT PE setup for use in LDG work.
It re

 * Sinead [development](https://git.ligo.org/sinead.walsh/automated_rapidpe_submission/wikis/setup-a-new-lvalert)


## Draft configuration (lvem)
The draft configuration stores results at CIT in a user directory; runs the code via ``lvem_alert``

```
/home/oshaughn/RIFT_automated/   # subdirectories
    online
    offline
```
and corresponding web directory output in
```

The codes documented below will use the following environment variables to identify output locations
```
export RIFT_AUTOMATED_OUTPUT=/home/oshaughn/RIFT_automated
export RIFT_AUTOMATED_OUTPUT_WEB=/home/oshaughn/public_html/RIFT_automated
```


## Future configuration (gwcelery)
The future configuration will run under ``gwcelery``.  It will store information in the same place.
The draft GWCelery code will lie in [a branch of a fork of GWCelery](https://git.ligo.org/richard-oshaughnessy/gwcelery/tree/start-RIFT) for now.
