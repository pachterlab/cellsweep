# cellmender

To install this cellmender:
- conda create -n cellmender python=3.10 -y
- conda activate cellmender
- git clone https://github.com/pachterlab/cellmender.git
- cd cellmender
- pip install -e .[dev,analysis]

To install [CellBender](https://github.com/broadinstitute/CellBender):
- conda create -n cellbender python=3.7 -y
- conda activate cellbender
- git clone https://github.com/broadinstitute/CellBender.git
- pip install -e CellBender

To install [wompwomp](https://github.com/pachterlab/wompwomp):
- git clone https://github.com/pachterlab/wompwomp.git
- cd wompwomp
- conda env create -f environment.yml
- conda activate wompwomp_env
- Rscript -e 'remotes::install_local(".")'