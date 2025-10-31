# cellstraightener

To install this notebook:
conda create -n cellstraightener python=3.10 -y
conda activate cellstraightener
git clone https://github.com/pachterlab/cellstraightener.git
cd cellstraightener
pip install -e .[dev,analysis]

To install CellBender:
conda create -n cellbender python=3.7 -y
conda activate cellbender
git clone https://github.com/broadinstitute/CellBender.git
pip install -e CellBender

To install wompwomp:
git clone https://github.com/pachterlab/wompwomp
cd wompwomp
conda env create -f environment.yml
conda activate wompwomp_env
Rscript -e 'remotes::install_local(".")'