import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import scanpy as sc,anndata as ad
import squidpy as sq
import os
import re
import sys
from scipy.sparse import vstack
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,scale
from scipy.spatial import distance_matrix, distance
from sklearn.neighbors import KernelDensity
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from sklearn.neighbors import BallTree
import time
import rdata
from scipy.sparse import csr_matrix,bsr_matrix,coo_matrix,issparse,lil_matrix,diags
from scipy.sparse.linalg import inv
import scipy as sp
from  scipy.ndimage import gaussian_filter
import igraph as ig
import glasbey
import warnings
import cairocffi as cairo
from sklearn.metrics import adjusted_rand_score,make_scorer
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn import preprocessing
import libpysal
from esda.losh import LOSH
from multiprocessing import Pool
from mclustpy import mclustpy
from sklearn.cluster import KMeans
import torch
#from GraphST import GraphST
import SEDR
#from GraphST.utils import clustering
from matplotlib.colors import ListedColormap
from esda import Moran
from libpysal.weights import KNN
from scsampler import scsampler 
from geosketch import gs
from scvalue import SCValue
from tqdm import tqdm
from fbpca import pca
from annoy import AnnoyIndex
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error
from numpy.linalg import norm



def geo_coord_index(adata,fraction =0.1,seed =0): 
    N =int(fraction * adata.X.shape[0])
    geo_coord_index = gs(adata.obsm['spatial'], N, seed=seed, replace=False)
    return geo_coord_index
def geo_transcriptomic_index(adata,fraction = 0.1,seed = 0):
    N =int(fraction * adata.X.shape[0])
    geo_transcriptomic_index = gs(adata.obsm['X_pca'], N, seed=seed, replace=False)
    return geo_transcriptomic_index

def geo_coord_shuffled_index(adata,fraction = 0.1, seed = 0):
    N =int(fraction * adata.X.shape[0])
    coords = adata.obsm['spatial']
    shuffled_coords = shuffle_coordinates(coords,seed = seed)
    geo_coord_shuffled_index = gs(shuffled_coords,N,seed = seed,replace = False)
    return geo_coord_shuffled_index
    
def scsampler_coord_index(adata,fraction = 0.1,seed = 0):
    res = scsampler(adata.obsm['spatial'], fraction=fraction, random_state=seed, copy=True,random_split = 16)
    scsampler_transcptomic_index = res[1]
    return sorted(list(scsampler_transcptomic_index))

def scsampler_transcriptomics_index(adata,fraction = 0.1,seed = 0):
    res = scsampler(adata.obsm['X_pca'], fraction=fraction, random_state=seed, copy=True,random_split = 16)
    return sorted(list(res[1]))


def scsampler_coord_shuffled_index(adata,fraction = 0.1,seed = 0):
    coords = adata.obsm['spatial']
    shuffled_coords = shuffle_coordinates(coords,seed = seed)
    res = scsampler(shuffled_coords,fraction = fraction,random_state = seed,copy = True,random_split = 16)
    return sorted(list(res[1]))

def uniform_index(adata,fraction = 0.1,seed = 0):
    np.random.seed(seed)
    num_cells = adata.n_obs
    sample_size = int(num_cells*fraction)
    sampled_indices = np.random.choice(adata.obs.index,size = sample_size,replace = False)
    return sorted(list(sampled_indices))



def shuffle_coordinates(coordinates,seed = 0):
    """Shuffles pairs of coordinates in a NumPy array.

    Args:
        coordinates (numpy.ndarray): An array of shape (N, 2) 
                                     where N is the number of coordinate pairs.

    Returns:
        numpy.ndarray: A new array with shuffled coordinate pairs.
    """
    rng = np.random.default_rng(seed = seed)
    shuffled_indices = rng.permutation(coordinates.shape[0])
    shuffled_coordinates = coordinates[shuffled_indices]
    return shuffled_coordinates



def leverage_index(adata,fraction = 0.1,score_column="gene_score",seed = 0):
    #print("running leverage sampling")
    np.random.seed(seed)

    if score_column not in adata.obs:
        raise ValueError(f"Column '{score_column}' not found in adata.obs")

    scores = adata.obs[score_column].values
    # scores = np.clip(scores, a_min=0, a_max=None)  # Ensure no negative values
    probabilities = scores / scores.sum()  # Normalize to get probabilities

    num_cells = adata.n_obs
    sample_size = int(num_cells * fraction)

    sampled_indices = np.random.choice(adata.n_obs, size=sample_size, replace=False, p=probabilities)
    return sorted(list(sampled_indices))

def create_leverage_index_dataframe(adata, fraction=0.1, num_seeds=10, start_seed=0):
    """
    Creates a DataFrame where each column contains the sorted indices from uniform_index
    for a different random seed.
    
    Args:
        adata: AnnData object to sample from
        fraction (float): Fraction of cells to sample
        num_seeds (int): Number of different seeds to use
        start_seed (int): Starting seed value
        
    Returns:
        pandas.DataFrame: DataFrame with columns named 'seed_{seed}' containing sorted indices
    """
    # Dictionary to store results
    results = {}
    
    # Run uniform_index for each seed
    for seed in range(start_seed, start_seed + num_seeds):
        column_name = f'seed_{seed}'
        indices = get_leverage_index(adata, fraction=fraction, seed=seed)
        results[column_name] = indices
    
    # Create DataFrame from results
    # Note: Columns might have different lengths, so we'll use a different approach
    df_dict = {}
    
    # Find the maximum length of any index list
    max_length = max(len(indices) for indices in results.values())
    
    # Pad shorter lists with NaN values
    for column_name, indices in results.items():
        # Pad with NaN if needed
        padded_indices = indices + [np.nan] * (max_length - len(indices))
        df_dict[column_name] = padded_indices
    
    # Create DataFrame
    result_df = pd.DataFrame(df_dict)
    
    return result_df



def neighborhood_sweep(adata, fraction = 0.1, base_directory = "test/dir", ground_truth = "celltype",knn_for_neighborhood_analysis = 10):

    method_index_dict = {
        #"geo_coords":"index/geo_coord_index",
        #"geo_coord_shuffled":"index/geo_coord_shuffled_index",
        "geo_transcriptomic":"index/geo_transcriptomic_index",
        "leverage":"index/leverage_index",
        #"scsampler_coord":"index/scsampler_coord_index",
        #"scsampler_coord_shuffled":"index/scsampler_coord_shuffled_index",
        "scsampler_transcriptomic":"index/scsampler_transcriptomic_index",
        "uniform":"index/uniform_index"}
    
    results_df = pd.DataFrame()
    results = {}
    cell_type_proportions_full = compute_cell_type_proportions_annoy(adata, adata,ground_truth = ground_truth,k = knn_for_neighborhood_analysis)

    for method, child_dir in method_index_dict.items():
        if child_dir != "index/uniform_index":
            index_df = load_csv_by_fraction(os.path.join(base_directory,child_dir),target_fraction = fraction)
            #sort the columns for easier access next time: 
            index_df = index_df.reindex(columns = sorted(index_df.columns))
        elif child_dir == "index/uniform_index": 
            index_df = generate_uniform_index_df(adata,fraction = fraction,random_seeds = range(10))
        results = {}
        if method not in results:
            results[method] = {}


        for column_name, index_values in index_df.items():
            seed = int(column_name.replace("seed_",""))
            sampled_adata = adata[index_values.values]
            
            cell_type_proportions_sub = compute_cell_type_proportions_annoy(sampled_adata, adata,ground_truth = ground_truth,k = knn_for_neighborhood_analysis)
            cell_type_proportions_full_partial = cell_type_proportions_full.loc[adata.obs.index[index_values.values]]
            metrics = compare_cell_type_proportions(cell_type_proportions_sub, cell_type_proportions_full_partial) 
            
            results[method] = metrics  
            results[method]['knn_for_neighborhood_analysis'] = knn_for_neighborhood_analysis
            results[method]['seed'] = seed
            results[method]['fraction'] = fraction
            partial_results_df = pd.DataFrame.from_dict(results,orient = 'index').reset_index()
            partial_results_df.rename(columns = {partial_results_df.columns[0]:'method'},inplace = True)
            results_df=pd.concat([results_df,partial_results_df],ignore_index = True)
    return results_df





def generate_uniform_index_df(adata, fraction=0.1, random_seeds = range(10)):
    """
    Generate a DataFrame containing indices of uniformly sampled subsets of the AnnData object.

    :param adata: The AnnData object to sample from.
    :param fraction: The fraction of data to sample in each iteration.
    :param random_seeds: A list of random seeds for reproducibility in each sampling iteration.
    :return: A DataFrame where each column represents the indices of a sampled subset.
    """
    # Validate inputs
    if not 0 < fraction <= 1:
        raise ValueError("The fraction must be between 0 and 1.")
    
    num_samples = len(adata)
    num_subsamples = len(random_seeds)  # Should be 10 based on your example
    
    # DataFrame to store index results
    uniform_index_df = pd.DataFrame()
    
    for i, seed in enumerate(random_seeds):
        # Set the random seed
        np.random.seed(seed)
        
        # Determine the number of samples for this fraction
        sample_size = int(num_samples * fraction)
        
        # Sample indices
        sampled_indices = sorted(np.random.choice(adata.n_obs, size=sample_size, replace=False))
        
        # Add sampled indices to DataFrame as a new column
        uniform_index_df[f'seed_{i}'] = pd.Index(sampled_indices)
    
    return uniform_index_df


def partial_hausdorff_distance_annoy(array1, array2, q=1e-4, metric='euclidean', n_trees=10):
    """
    Compute the partial Hausdorff distance from array1 to array2 using Annoy for approximation.
    
    Parameters:
    - array1: numpy.ndarray, shape (n, d)
        The first array of points.
    - array2: numpy.ndarray, shape (m, d)
        The second array of points.
    - q: float, optional
        The parameter for the partial Hausdorff distance, should be between 0 and 1.
    - metric: string, optional
        The metric to use for distance calculations. Annoy supports 'euclidean' and 'angular'.
    - n_trees: int, optional
        Number of trees to use for Annoy's index construction.
    
    Returns:
    - float
        The estimated partial Hausdorff distance between array1 and array2.
    """
    # Build Annoy index for array2
    num_features = array2.shape[1]
    annoy_index = AnnoyIndex(num_features, metric)
    for i, vector in enumerate(array2):
        annoy_index.add_item(i, vector)
    print("building Annoy index")
    start = time.time()
    annoy_index.build(n_trees)
    end = time.time()
    print(f"Annoy index took {end-start:.4f} seconds to build")

    # Query for nearest neighbors
    print("finding nearest neighbors")
    start = time.time()
    distances = []
    for point in array1:
        nearest_idx, dist = annoy_index.get_nns_by_vector(point, 1, include_distances=True)
        distances.append(dist[0])
    end = time.time()
    print(f"ANN distance calculation took: {end-start:.4f} seconds")

    # Sort all computed distances
    distances.sort()
    # Determine the index of the Kth largest value for partial Hausdorff
    K = int(np.floor((1 - q) * len(distances)))
    # Handle edge cases and return the Kth largest value
    return distances[K-1] if K > 0 else distances[0]




def load_csv_by_fraction(directory, target_fraction= 0):
    """
    Walks through a directory to find and load a CSV file with a matching fraction in its name.
    
    Args:
        directory (str): Path to the directory containing CSV files
        target_fraction (float): The fraction to match in the filename (between 0.0 and 1.0)
        
    Returns:
        pandas.DataFrame: DataFrame containing the CSV data, or None if no matching file is found
    """
    # Convert target_fraction to string for comparison
    target_str = str(target_fraction)
    
    # Pattern to match index_fraction.csv files
    pattern = re.compile(r'index_(0?\.\d+)\.csv$')
    
    for root, _, files in os.walk(directory):
        for file in files:
            match = pattern.match(file)
            if match:
                file_fraction = match.group(1)
                
                # Compare fractions (accounting for different string representations)
                if float(file_fraction) == float(target_fraction):
                    file_path = os.path.join(root, file)
                    print(f"Found matching file: {file_path}")
                    return pd.read_csv(file_path)
    
    print(f"No CSV file found with fraction {target_fraction}")
    return None




def compare_cell_type_proportions(matrix1, matrix2):
    """
    Computes cosine similarity, Jensen-Shannon Divergence (JSD), Mean Squared Error (MSE),
    and Frobenius norm between two cell type proportion matrices.

    Parameters:
    - matrix1: pandas DataFrame (n_cells x n_cell_types), first cell type proportion matrix
    - matrix2: pandas DataFrame (n_cells x n_cell_types), second cell type proportion matrix

    Returns:
    - results: dict with 'mean_cosine_similarity', 'mean_jsd', 'mse', and 'frobenius_norm'
    """
    if matrix1.shape != matrix2.shape:
        raise ValueError("Both matrices must have the same shape.")
    
    # Initialize results with NaN
    results = {
        "mean_cosine_similarity": np.nan,
        "mean_jsd": np.nan,
        "mse": np.nan,
        "frobenius_norm": np.nan
    }

    # Compute Cosine Similarity (mean across all cells) and handle memory issues
    try:
        cos_sim = cosine_similarity(matrix1, matrix2)
        results["mean_cosine_similarity"] = np.mean(np.diag(cos_sim))  # Take mean of diagonal elements (self-similarity)
    except MemoryError:
        print("MemoryError encountered during cosine similarity calculation. Calculating manually.")
        try:
            cos_sim_values = []
            for i in range(matrix1.shape[0]):
                vec1 = matrix1.iloc[i].values
                vec2 = matrix2.iloc[i].values
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 > 0 and norm2 > 0:
                    cos_sim_values.append(dot_product / (norm1 * norm2))
                else:
                    cos_sim_values.append(0)  # Handle zero vector case
            
            results["mean_cosine_similarity"] = np.mean(cos_sim_values)
        except Exception as e:
            print(f"An unexpected error occurred during manual cosine similarity calculation: {e}")

    # Compute Jensen-Shannon Divergence (JSD) and handle memory issues
    try:
        jsd_values = [jensenshannon(matrix1.iloc[i], matrix2.iloc[i]) for i in range(len(matrix1))]
        results["mean_jsd"] = np.mean(jsd_values)
    except MemoryError:
        print("MemoryError encountered during JSD calculation. Returning NaN.")
    except Exception as e:
        print(f"An unexpected error occurred during JSD calculation: {e}")

    # Compute Mean Squared Error (MSE) and handle memory issues
    try:
        results["mse"] = mean_squared_error(matrix1, matrix2)
    except MemoryError:
        print("MemoryError encountered during MSE calculation. Returning NaN.")
    except Exception as e:
        print(f"An unexpected error occurred during MSE calculation: {e}")

    # Compute Frobenius Norm and handle memory issues
    try:
        results["frobenius_norm"] = norm(matrix1.values - matrix2.values, 'fro')
    except MemoryError:
        print("MemoryError encountered during Frobenius Norm calculation. Returning NaN.")
    except Exception as e:
        print(f"An unexpected error occurred during Frobenius Norm calculation: {e}")

    return results
def compute_neighborhood_metrics(adata, fraction=0.1, seed=0, k=20, ground_truth="Cell_Type", n_trees=10):
    """
    Evaluates four sampling methods by computing cell type proportions and comparing them to the full dataset.

    Parameters:
    - adata: AnnData object
    - fraction: float, fraction of cells to sample
    - seed: int, random seed for reproducibility
    - k: int, number of nearest neighbors for Annoy
    - ground_truth: str, column in `adata.obs` containing cell type labels
    - n_trees: int, number of trees for Annoy index

    Returns:
    - A DataFrame containing similarity metrics for each sampling method
    """
    start_time = time.time()
    # Compute full dataset cell type proportions
    print("Computing cell type proportions for full dataset...")
    cell_type_proportions_full = compute_cell_type_proportions_annoy(adata, adata,ground_truth = ground_truth)

    # Generate sampled indices using different methods
    print("Generating sampled indices...")
    sampled_indices = sampling_methods_get_index(adata,fraction=fraction, seed=seed)
    results = {}

    for method, indices in sampled_indices.items():
        print(f"Processing {method}...")
        # Compute cell type proportions for subsampled data
        sampled_adata = adata[indices]
        cell_type_proportions_sub = compute_cell_type_proportions_annoy(sampled_adata, adata,ground_truth = ground_truth)

        # Extract corresponding subset from full dataset
        cell_type_proportions_full_partial = cell_type_proportions_full.loc[adata.obs.index[indices]]

        # Compute similarity metrics
        metrics = compare_cell_type_proportions(cell_type_proportions_sub, cell_type_proportions_full_partial)
        
        # Store metrics
        results[method] = metrics

    # Convert results dictionary to DataFrame
    results_df = pd.DataFrame.from_dict(results,orient = 'index').reset_index()
    results_df.rename(columns = {results_df.columns[0]:'method'},inplace = True)

    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")
    return results_df



def compute_cell_type_proportions_annoy(subset_adata, full_adata, ground_truth="Cell_Type", k=20, n_trees=10):
    """
    Computes the proportion of cell types in the neighborhood of each cell using Annoy.

    Parameters:
    - subset_adata: AnnData object with spatial coordinates for the subset dataset in `subset_adata.obsm`
    - full_adata: AnnData object representing the full dataset with all possible cell types
    - k: Number of nearest neighbors to consider
    - n_trees: Number of trees for Annoy (higher = better accuracy but slower)

    Returns:
    - A DataFrame of shape (n_cells, n_cell_types) with proportions of each cell type in its neighborhood, aligned with full dataset cell types.
    """
    # Extract spatial coordinates from subset adata
    coords = subset_adata.obsm["spatial"]
    n_cells, n_dims = coords.shape

    # Build Annoy index
    annoy_index = AnnoyIndex(n_dims, metric='euclidean')
    for i in range(n_cells):
        annoy_index.add_item(i, coords[i])
    annoy_index.build(n_trees)  # Build the index

    # Extract cell type labels from full adata
    full_cell_types = np.array(full_adata.obs[ground_truth])
    full_unique_types = np.unique(full_cell_types)  # Full unique cell types
    type_to_idx = {t: i for i, t in enumerate(full_unique_types)}  # Mapping cell type to column index

    # Initialize output matrix
    proportions = np.zeros((n_cells, len(full_unique_types)))

    # Extract cell type labels from subset adata
    subset_cell_types = np.array(subset_adata.obs[ground_truth])

    # Compute kNN for each cell in subset adata
    for i in range(n_cells):
        neighbors = annoy_index.get_nns_by_item(i, k+1)[1:]  # Exclude self
        neighbor_types = subset_cell_types[neighbors]  # Get cell types of kNN
        # Count occurrences of each cell type
        for t in neighbor_types:
            proportions[i, type_to_idx[t]] += 1
        proportions[i] /= k  # Normalize to proportions

    # Convert to DataFrame with full unique cell types as columns
    prop_df = pd.DataFrame(proportions, columns=full_unique_types, index=subset_adata.obs.index)
    return prop_df




#lets try to re-run but with a flightly faster implementation: 
from annoy import AnnoyIndex

def build_annoy_index(adata, ground_truth="Cell_Type", k=20, n_trees=10):
    coords = adata.obsm["spatial"]
    n_cells, n_dims = coords.shape
    annoy_index = AnnoyIndex(n_dims, metric='euclidean')
    for i in range(n_cells):
        annoy_index.add_item(i, coords[i])
    annoy_index.build(n_trees)
    return annoy_index

def compute_cell_type_proportions_annoy_subset(adata, full_adata, annoy_index, subset_indices, ground_truth="Cell_Type", k=20):
    """
    Compute neighborhood cell type proportions for a subset, using prebuilt Annoy index on full data.

    Parameters:
    - adata: subset AnnData (optional, only for index purposes)
    - full_adata: full AnnData, provides labels
    - annoy_index: Annoy index built on full data coordinates
    - subset_indices: indices of subset cells in the full dataset
    - ground_truth: cell type column name
    - k: neighbors to query
    
    Returns:
    - DataFrame of shape (len(subset_indices), n_cell_types) with neighborhood cell type proportions
    """
    full_cell_types = np.array(full_adata.obs[ground_truth])
    full_unique_types = np.unique(full_cell_types)
    type_to_idx = {t: i for i, t in enumerate(full_unique_types)}

    proportions = np.zeros((len(subset_indices), len(full_unique_types)))

    for i, full_idx in enumerate(subset_indices):
        neighbors = annoy_index.get_nns_by_item(full_idx, k+1)[1:]  # exclude self
        neighbor_types = full_cell_types[neighbors]
        for t in neighbor_types:
            proportions[i, type_to_idx[t]] += 1
        proportions[i] /= k

    prop_df = pd.DataFrame(proportions, columns=full_unique_types, index=full_adata.obs.index[subset_indices])
    return prop_df

def neighborhood_sweep_faster(adata, fraction=0.1, base_directory="test/dir", ground_truth="celltype",
                       knn_for_neighborhood_analysis=10,
                       cell_type_proportions_full=None,
                       annoy_index=None):
    method_index_dict = {
        #"geo_coords":"index/geo_coord_index",
        #"geo_coord_shuffled":"index/geo_coord_shuffled_index",
        "geo_transcriptomic":"index/geo_transcriptomic_index",
        "leverage":"index/leverage_index",
        #"scsampler_coord":"index/scsampler_coord_index",
        #"scsampler_coord_shuffled":"index/scsampler_coord_shuffled_index",
        "scsampler_transcriptomic":"index/scsampler_transcriptomic_index",
        "uniform":"index/uniform_index"}
    
    results_df = pd.DataFrame()
    
    for method, child_dir in method_index_dict.items():
        # Load or generate indices as before
        if child_dir != "index/uniform_index":
            index_df = load_csv_by_fraction(os.path.join(base_directory,child_dir),target_fraction=fraction)
            index_df = index_df.reindex(columns=sorted(index_df.columns))
        else:
            index_df = generate_uniform_index_df(adata,fraction=fraction,random_seeds=range(10))
            
        results = {}
        
        for column_name, index_values in index_df.items():
            seed = int(column_name.replace("seed_",""))
            subset_indices = index_values.values  # These are indices relative to full adata
            
            # Compute neighborhood proportions using the prebuilt annoy_index 
            cell_type_proportions_sub = compute_cell_type_proportions_annoy_subset(
                adata[subset_indices], adata, annoy_index, subset_indices,
                ground_truth=ground_truth,
                k=knn_for_neighborhood_analysis)
            
            cell_type_proportions_full_partial = cell_type_proportions_full.loc[adata.obs.index[subset_indices]]
            
            metrics = compare_cell_type_proportions(cell_type_proportions_sub, cell_type_proportions_full_partial)
            
            results[method] = metrics
            results[method]['knn_for_neighborhood_analysis'] = knn_for_neighborhood_analysis
            results[method]['seed'] = seed
            results[method]['fraction'] = fraction
            
            partial_results_df = pd.DataFrame.from_dict(results,orient='index').reset_index()
            partial_results_df.rename(columns={partial_results_df.columns[0]: 'method'}, inplace=True)
            
            results_df = pd.concat([results_df, partial_results_df], ignore_index=True)
    
    return results_df

# Function to get full cell type proportions for any k <= max_k by slicing neighbors
def get_full_cell_type_proportions(k):
    return compute_full_proportions_for_k(all_neighbors, k, full_cell_types, full_unique_types, type_to_idx)

def compute_full_proportions_for_k(all_neighbors, k, full_cell_types, full_unique_types, type_to_idx):
    n_cells = len(all_neighbors)
    n_types = len(full_unique_types)
    proportions = np.zeros((n_cells, n_types))
    
    for i in range(n_cells):
        neighbors = all_neighbors[i][:k]  # take first k
        neighbor_types = full_cell_types[neighbors]
        for t in neighbor_types:
            proportions[i, type_to_idx[t]] += 1
        proportions[i] /= k
    
    # Return DataFrame indexed over full dataset cells
    return pd.DataFrame(proportions, columns=full_unique_types, index=adata.obs.index)

def get_full_proportions_k_cached(all_neighbors, k, full_cell_types, full_unique_types, type_to_idx):
    # you can cache or recompute on the fly from all_neighbors
    # see compute_full_proportions_for_k above
    return compute_full_proportions_for_k(all_neighbors, k, full_cell_types, full_unique_types, type_to_idx)

