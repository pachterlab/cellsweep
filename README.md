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

To install [CellBender](https://github.com/broadinstitute/CellBender):
```
git clone https://github.com/broadinstitute/CellBender.git
conda create -n cellbender python=3.7 -y
conda activate cellbender
pip install -e CellBender
```

To install [wompwomp](https://github.com/pachterlab/wompwomp):
```
git clone https://github.com/pachterlab/wompwomp.git
cd wompwomp
conda env create -f environment.yml
conda activate wompwomp_env
Rscript -e 'remotes::install_local(".")'
```

## Docker
docker run -it --rm -p 8888:8888 josephrich98/cellmender:0.1.0
*go to localhost:8888*
*copy-paste token on login - eg http://localhost:8888/lab?token=<your_token>*