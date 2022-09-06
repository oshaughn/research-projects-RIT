


## TEOBResumS

TEOBResumS is available from [git source](https://bitbucket.org/eob_ihes/teobresums/src) and via pypi.  We recommend you install via
```
pip install teobresums
```
which provides the ``EOBRun_module``.  However, we have noticed some incompatibilties with numpy can arise if this is done.  If needed, please instead install from source as described above

## gwsurrogate

gwsurrogate is available from pypi [here](https://pypi.org/project/gwsurrogate/) via 
```
pip isntall gwsurrogate
```
Because of version changes, you need to be careful about precisely which version you install, as the default version may be incompatible with your RIFT installation.  You will need to set environment variables so RIFT knows about this installation  (e.g., the ``GW_SURROGATE`` environment variable to identify where surrogate downloads have been stored).


## NRWaveformCatalogManager

This package is available as source to people who collaborate using it.

If this package is installed, you must define the ``NR_BASE`` environment variable.  Provide  an empty string to avoid introducing possible errors.

## Concordance

Concordance provides extensions to RIFT to enable joint EM+GW PE.  This package is available at https://github.com/oshaughn/Concordance
This package installs a collection of related tools, documented at the source installation page.
