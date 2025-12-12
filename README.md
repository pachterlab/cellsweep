# cellsweep

## Install
### Basic use
```
pip install cellsweep  #!!! not yet uploaded on pypi - see below
```

### To run notebooks:
```
pip install cellsweep[analysis]  # pip install cellsweep["analysis"] for Mac users  #!!! not yet uploaded on pypi - see below
```

### To remake figures from the paper:
```
git clone https://github.com/pachterlab/cellsweep.git
cd cellsweep
conda env create -f environment.yml  #!!! until on pypi, in environmeny.yml replace cellsweep[analysis]==0.1.0 with .[analysis]
```

To install [scAR](https://github.com/Novartis/scar):
```
git clone https://github.com/Novartis/scar.git
cd scar
If CPU: conda env create -f scar-cpu.yml
If GPU: conda env create -f scar-gpu.yml
```

To install [biowomp](https://github.com/pachterlab/biowomp):
```
git clone https://github.com/pachterlab/biowomp.git
cd biowomp
conda env create -f environment.yml
conda activate wompwomp_env
Rscript -e 'remotes::install_local(".")'
```
