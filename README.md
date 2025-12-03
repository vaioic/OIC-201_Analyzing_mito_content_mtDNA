# Mitochondria and mtDNA quantification

The goal of this project is to quantify the mitochondrial area and mtDNA density in _C. elegans_. The project is a collaboration with the Patel Lab at the Van Andel Institute.

## Getting started

### Prerequisites

- [Python](https://www.python.org/downloads/) version 3.14.0

### Installation

1. Download or clone the GitHub repository
   ```bash
   git clone git@github.com:vaioic/OIC-201_Analyzing_mito_content_mtDNA.git
   cd OIC-201_Analyzing_mito_content_mtDNA
   ```

2. Create a python virtual environment
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment
   ```bash
   .\venv\Scripts\activate
   ```

4. Install the dependencies using Pip
   ```bash
   python -m pip install -r .\requirements.txt
   ```

## Install sam2

```bash
mkdir inc
cd inc
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .

### Running the code

1. Start the virtual environment if not already loaded
   ```bash
   .\venv\Scripts\activate
   ```

2. Start Jupyter Lab
   ```bash
   python -m jupyterlab
   ```

3. Open the ``analyze_images.ipynb`` notebook

4. Modify the variable ``path`` as needed to point to the image file

## Issues

If you encounter any issues with running the code or have any questions, please create an [Issue](https://github.com/vaioic/OIC-201_Analyzing_mito_content_mtDNA/issues) or send an email to opticalimaging@vai.org. If you are reporting a programmatic bug, please include any error messages to aid with troubleshooting.

## Acknowledgements

### Dependencies

This project relies on the following packages:

- jupyterlab v4.4.10
- bioio v3.0.0
- bioio-nd2 v1.5.9
- scikit-image v0.25.2
- numpy v2.3.4
- scipy v1.16.3

**Note:** For full dependency list, see [requirements.txt](requirements.txt).

