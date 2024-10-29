# Up sampled PSF enables high accuracy 3D super-resolution imaging with sparse sampling rate
Single-molecule localization microscopy (SMLM) provides nanoscale imaging, but pixel integration of acquired SMLM images limited the choice of sample rate which restricts the information content conveyed within each image. We propose an up-sampled point spread function (PSF) inverse modeling method for large-pixel single molecule localization, enabling precise 3D super-resolution imaging with sparse sampling rate. Our approach could reduce data volume or expand the field of view by nearly an order of magnitude, while maintaining high localization accuracy, greatly improving the imaging throughput with limited pixels offered by the existing cameras. 

# Systems tested
- Windows 10 with RTX 3080, RTX 3090
- Rocky Linux 8.7 with A6000

# Installation
## Windows
1. Install miniconda for windows, [miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Open Anaconda Powershell Prompt, clone the up sampled PSF package     
```
git clone https://github.com/ries-lab/uiPSF.git
cd uiPSF
```
3. Create a new conda enviroment for the up sampled PSF package  
- for GPU: 
```
conda env create --name psfinv --file=environment.yml
```   
- for CPU: 
```
conda create --name psfinv python=3.7.10
```
4. Activate the installed enviroment and install the up sampled PSF package
```
conda activate psfinv
pip install -e .
```

## Mac
1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) for Mac.
2. Open Terminal and follow the [installation for Windows](#Windows) to install the up sampled PSF package. Only the CPU version is supported. 

## Linux
1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) for Linux.
2. Follow the [installation for Windows](#Windows) to install the uiPSF package.
3. If the GPU version is installed, add cudnn path
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

# Demo notebooks
- For bead data
  - [Up_sampled PSF modelling_for_simulated_data](demo/demo1_up_sampled_PSF_for_Sim.ipynb).
  - [Up_sampled PSF modelling_for_experimental_data](demo/demo2_up_sampled_PSF_for_Exp.ipynb).


# Example data 
- simulated data using a vector model with 330nm pixel size.
- experimental bead data from a microscopes with 321nm pixel size.

Download the [example data](https://zenodo.org/records/14000637)
# How to run demo notebook
1. Install up sampled PSF for your operating system.
2. Install [Visual Studio Code](https://code.visualstudio.com/Download).
3. Open Visual Studio Code (VScode), click *Extensions* from the sidebar menu and search for `Python` and install `Python extension for VScode`.
4. Go to File->Open Folder, select the uiPSF folder from git clone.
5. Open the file *demo/datapath.yaml*, change the `main_data_dir` to the path of the downloaded example data.
6. Navigate to a demo notebook, e.g. *demo1_up_sampled_PSF_for_Sim.ipynb*.
7. Click the run button of the first cell, if running for the first time, a window will popup asking to install the `ipykernel` package, click install. Then a drop down menu will show up asking to select the kernel, select the created conda enviroment `psfinv` during the installation.
8. Run subsequent cells sequentially.

For explanation of user defined parameters, please see list of [all user defined parameters](config/parameter%20description.md). 
## Tips
- If a GPU is not available, comment the last two lines in the first cell *Setup environment* of the demo notebook.
- Don't run two notebooks at the same time, click `Restart` at the top of the notebook to release the memory.
## Need help?
Open an issue here on github, or contact Jianwei Chen (12149038@mail.sustech.edu.cn) and Yiming Li (liym2019@sustech.edu.cn).
