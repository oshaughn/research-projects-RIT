RIFT Architecture
==================

The RIFT (Rapid inference via Iterative FiTting) framework is designed for highly parallelized, efficient parameter inference for gravitational wave (GW) sources. Its core philosophy is to replace computationally expensive waveform evaluations with fast interpolants, iteratively refining the parameter space.

Core Components
---------------

1. **Simulation Manager** (``RIFT.simulation_manager``)
   The entry point for large-scale runs. It handles the submission and tracking of simulations across different HPC schedulers (Condor, Slurm, PBS). It manages the ``SimulationArchive``, which tracks which simulations have been generated and where their results are stored.

2. **Integrators** (``RIFT.integrators``)
   The "engines" of the inference process. RIFT provides multiple sampling strategies:
   - **Standard MC Samplers**: Basic Monte Carlo integration.
   - **Ensemble Samplers**: Using parallel chains to explore the posterior.
   - **GPU Samplers**: Leveraging CUDA for massive acceleration of likelihood evaluations.
   - **NFlow Samplers**: Using Normalizing Flows for more efficient sampling.
   - **Unreliable Oracle**: A set of optimization tools (like hill climbing) used to find peaks in the likelihood surface.

3. **Interpolators** (``RIFT.interpolators``)
   To avoid calling expensive waveform generators millions of times, RIFT uses interpolants. This module implements various techniques:
   - **Gaussian Processes (GP)**: For flexible, non-parametric interpolation.
   - **Bayesian Least Squares**: For structured fits.
   - **Senni / GPyTorch**: Integration with modern ML frameworks for high-dimensional interpolation.

4. **Likelihood Engine** (``RIFT.likelihood``)
   The core mathematical heart of the package. It computes the inner products between data and templates. It is heavily optimized, with GPU-accelerated kernels for the most frequent operations.

5. **Physics Layer** (``RIFT.physics``)
   Handles the astrophysical and numerical relativity (NR) inputs.
   - **EOSManager**: Manages Equation of State tables.
   - **Waveform Managers**: Handles the loading and rotation of NR waveforms.

6. **Miscellaneous Utilities** (``RIFT.misc``)
   The "glue" of the package, providing XML parsing, coordinate transformations, and parameter handling.

Data Flow
----------

The typical execution flow is as follows:

1. **Simulation Setup**: The ``SimulationManager`` identifies missing simulations and submits them to the cluster.
2. **Integration**: An ``Integrator`` (e.g., ``mcsamplerEnsemble``) is initialized.
3. **Evaluation**: For each sample, the ``Integrator`` asks the ``Likelihood`` engine for a value.
4. **Acceleration**: The ``Likelihood`` engine checks if an ``Interpolator`` can provide a fast approximation. If not, it triggers a new simulation or uses a base-level waveform.
5. **Refinement**: The process repeats, with the interpolators becoming more accurate as more simulation data is gathered.
