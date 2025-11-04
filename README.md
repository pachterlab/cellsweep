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

### To remake local environments:
```
git clone https://github.com/pachterlab/cellmender.git
cd cellmender
conda env create -f environment.yml
```

To install [wompwomp](https://github.com/pachterlab/wompwomp):
```
git clone https://github.com/pachterlab/wompwomp.git
cd wompwomp
conda env create -f environment.yml
conda activate wompwomp_env
Rscript -e 'remotes::install_local(".")'
```
