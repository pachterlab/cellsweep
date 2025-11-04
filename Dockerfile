FROM jupyter/datascience-notebook:python-3.10.11

USER root
# Install system dependencies for R packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER $NB_UID
WORKDIR /home/jovyan/work

# ------------------------------------------------------------
# 🧬 R setup: wompwomp, SoupX, singleCellTK, decontX
# ------------------------------------------------------------
RUN R -e 'install.packages(c("remotes", "BiocManager", "SoupX"), repos="https://cloud.r-project.org")' && \
    R -e 'BiocManager::install(c("singleCellTK", "decontX"), ask=FALSE, update=FALSE)' && \
    R -e 'remotes::install_github("pachterlab/wompwomp")'

# ------------------------------------------------------------
# 🧫 Python setup: CellBender (Python 3.7 env)
# ------------------------------------------------------------
SHELL ["/bin/bash", "-c"]
RUN conda create -y -n cellbender python=3.7 && \
    source activate cellbender && \
    git clone https://github.com/broadinstitute/CellBender.git && \
    pip install -e CellBender && \
    conda deactivate

# ------------------------------------------------------------
# 🧩 CellMender install (analysis extras)
# ------------------------------------------------------------
RUN git clone https://github.com/pachterlab/cellmender.git && \
    cd cellmender && pip install '.[analysis]'

# ------------------------------------------------------------
# 🔧 Set working directory and environment defaults
# ------------------------------------------------------------
WORKDIR /home/jovyan/work
EXPOSE 8888

CMD ["start-notebook.sh"]
