# Dockerfile.soupx
FROM rocker/r-ver:4.5.1

# Become root to install system dependencies
USER root

# Optional: install system dependencies needed for building R packages
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libhdf5-dev \
    libpng-dev \
    libtiff-dev \
    libjpeg-dev \
    libcairo2-dev \
    libfontconfig1-dev \
    libmagick++-dev \
    libglpk-dev \
    libfftw3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install BiocManager and decontX from CRAN
R -e 'install.packages("BiocManager", repos="https://cloud.r-project.org")'
R -e 'BiocManager::install(c("singleCellTK", "decontX"), ask=FALSE, update=FALSE, version = "3.22")'

# Set working directory (same as your mounted path)
WORKDIR /home/ruser/work/cellmender
