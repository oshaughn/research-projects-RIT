Data Formats
=============

This section describes the data formats used throughout the RIFT pipeline for parameter grids, samples, and result files.

Conventional ASCII Tabular Data
-------------------------------

The most common format for providing parameter grids and recording results is labeled ASCII tabular data.

**Characteristics:**
- **Labelled**: The first line (header) contains the names of the parameters/columns.
- **Tabular**: Data is organized in columns and rows.
- **Compatibility**: This format is designed to be natively compatible with standard Python data analysis tools:
    - ``numpy.genfromtxt``
    - ``pandas.read_csv``

**Usage Example:**
A typical parameter grid file might look like this:

.. code-block:: text

   mass1  mass2  chi_eff  dist
   30.5   28.2   0.05     400
   31.0   27.5   0.04     410
   30.8   28.0   0.06     390

This simplicity ensures that RIFT outputs can be easily processed by external scripts and that input grids can be generated using any text editor or spreadsheet software.

Composite Format
----------------

The composite format is used for high-efficiency data transfer between specific pipeline stages.

**Characteristics:**
- **Unlabelled**: Data contains no header.
- **Context-Dependent**: The number of columns depends on the context of the analysis.
- **Pipeline Role**: 
    - **Input**: Used as input data for ``util_ConstructIntrinsicPosterior.py`` (abbreviated as **CIP**).
    - **Output**: Used as output data from ``integrate_likelihood_extrinsic_*`` (abbreviated as **ILE**).

XML Format
-----------

RIFT utilizes standard ``ligolw`` XML files for structured data storage, event metadata, and coincidence information. This ensures compatibility with the broader LIGO/Virgo/KAGRA software ecosystem.

Hyperpipe-style Format
----------------------

The Hyperpipe-style format is a specialized version of the labeled ASCII format designed to maintain a consistent column order throughout a complex analysis.

**Characteristics:**
- **Labelled**: Follows the same labeled ASCII structure as the conventional format.
- **Strict Column Order**: Enforces a consistent column order to prevent randomization during data processing.
- **Required Fields**: The first two fields are always ``lnL`` (log-likelihood) and ``sigma_lnL`` (uncertainty in log-likelihood).

This format is critical for pipelines where downstream tools expect a fixed schema regardless of the specific parameters being analyzed.
