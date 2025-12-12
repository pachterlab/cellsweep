# Dockerfile.soupx
FROM rocker/r-ver:4.5.1

# Become root to install system dependencies
USER root

# --- Fix Ubuntu GPG verification issue ---
RUN apt-get update || true && apt-get install -y --no-install-recommends gnupg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# --- Install system dependencies ---
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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install SoupX from CRAN
RUN R -e 'install.packages("SoupX", repos="https://cloud.r-project.org")'

# Set working directory (same as your mounted path)
WORKDIR /home/ruser/work/cellsweep
