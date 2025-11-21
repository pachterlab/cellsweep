# cellmender

## Install
### Basic use
```
pip install cellmender  #!!! not yet uploaded on pypi - see below
```

### To run notebooks:
```
pip install cellmender[analysis]  #!!! not yet uploaded on pypi - see below
```

### To remake figures from the paper:
```
git clone https://github.com/pachterlab/cellmender.git
cd cellmender
conda env create -f environment.yml
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
