# Benchmarking-sketching-for-spatial-transcriptomics
This repository contains the analysis code and Jupyter notebooks supporting the paper "Benchmarking sketching for spatial transcriptomics". The study evaluates how common sketching/subsampling strategies—uniform sampling, leverage-score sampling, Geosketch (minimax/Hausdorff), and scSampler (maximin)—perform on spatial transcriptomics (ST) data, and tests three input representations: PCA embeddings, raw spatial coordinates, and spatially smoothed embeddings. We show that expression-only sketching often over-samples high-variability regions and distorts tissue architecture, while a spatially smoothed leverage-score extension recovers rare cell states while preserving uniform spatial coverage.

![Analysis pipeline](./figure_1.pdf)

## Repository layout 

     
     notebooks/ — Jupyter notebooks used for analysis and figure generation.  
     HPC scripts/ — scripts run on HPC system to generate sketches and quality metrics.  

## Data access
Large ST datasets used in the paper are not stored in this repository. See the paper’s Methods and the notebooks for links and instructions to download publicly available datasets (MERFISH, public spatial datasets) or for how to access processed subsets used in the analysis. 

## Contact
For questions about code or results, open an issue on this repo or contact ian.gingerich.gr@dartmouth.edu

