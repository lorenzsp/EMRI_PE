#!/bin/bash

#echo Now going to build environments and set up dependencies. 
#
cd .. # enter root directory

echo building directory for data files
mkdir data_files  # Build data directory to store .h5 samples

echo Loading conda module
# module load conda  # CNES cluster, need to load conda prior to using it

echo Now creating environment      # Set up conda environment -- vanilla_few
conda create -y -n sbi_emri -c conda-forge gcc_linux-64 gxx_linux-64 wget gsl lapack=3.6.1 hdf5 numpy Cython scipy tqdm jupyter ipython h5py requests matplotlib corner python=3.9 
conda activate sbi_emri         

echo Installing cupy-cuda toolkit
pip install cupy-cuda12x           # Warning: this is SPECIFIC to the CNES cluster for the GPUs available

# module load gcc

# Important to load cuda when installing the repos below! 
# module load cuda
export PATH=$PATH:/usr/local/cuda-12.5/bin/

echo Installing eryn, sampler built off emcee. 
pip install eryn                   # Install Eryn 

# The code below git clones various repositories and installs them in one sitting
echo Now going to clone dependencies 
mkdir Github_Repos; cd Github_Repos

# Clone the repositories
git clone https://github.com/mikekatz04/LISAanalysistools.git
git clone https://github.com/mikekatz04/lisa-on-gpu.git
git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git

# install each one using python
echo Now installing LISAanalysistools 
pip install lisaanalysistools

cd ../lisa-on-gpu
git reset --hard f042d4f
echo Now installing lisa-on-gpu 
python setup.py install

echo Now installing FastEMRIWaveforms 
cd ../FastEMRIWaveforms
git reset --hard e4038da
python setup.py install

cd ../../ # Get back to root directory

echo Your installation is complete!




