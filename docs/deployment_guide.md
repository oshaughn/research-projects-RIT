# ReadTheDocs Deployment Guide

This project uses Sphinx for documentation and is configured for deployment via ReadTheDocs (RTD).

## Current Configuration
- **Config File**: `.readthedocs.yaml` in the root directory.
- **Build Environment**: Ubuntu 22.04, Python 3.11.
- **Sphinx Configuration**: `docs/source/conf.py`.
- **Requirements**: Defined in `docs/requirements.txt`.

## Deployment Steps

### 1. Connect Repository to ReadTheDocs
1. Log in to [readthedocs.org](https://readthedocs.org/).
2. Click **"Import a Project"**.
3. Connect your GitHub/GitLab/Bitbucket account.
4. Select the repository containing this project.
5. Follow the prompts to create the project.

### 2. Configure Build Settings
ReadTheDocs will automatically detect the `.readthedocs.yaml` file and use the specified build environment. 

### 3. Trigger Build
Once connected, RTD will trigger a build on every push to the designated branch (usually `main` or `master`).

## Verification
After the build completes, you can verify the documentation at:
`https://<project-name>.readthedocs.io/en/latest/`

## Troubleshooting
- **Build Failures**: Check the "Builds" tab in the RTD dashboard for logs.
- **Missing Modules**: Ensure all required Python packages are listed in `docs/requirements.txt`.
- **Formatting Issues**: Review the Sphinx warnings in the build logs.
