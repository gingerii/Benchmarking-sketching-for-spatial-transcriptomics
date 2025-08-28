import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import scanpy as sc,anndata as ad
import squidpy as sq
import os
import sys
import re
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



class RASP:
    @staticmethod
    def build_weights_matrix(adata, n_neighbors=6, beta=2, platform='visium'):
        """
        Build a sparse distance matrix including only the K nearest neighbors, and compute inverse weighting.

        Parameters:
        - adata: Annotated data object.
        - n_neighbors: int - number of nearest neighbors to include.
        - beta: weight exponent parameter.
        - platform: string - type of platform.

        Returns:
        - sparse_distance_matrix: csr_matrix - sparse distance matrix of shape (n_samples, n_samples).
        """
        coords = adata.obsm['spatial']
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        # Build the sparse matrix
        data = distances.flatten()
        row_indices = np.repeat(np.arange(coords.shape[0]), n_neighbors)
        col_indices = indices.flatten()
        sparse_distance_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(coords.shape[0], coords.shape[0])).tocsr()

        # Remove outliers
        temp_matrix = sparse_distance_matrix.tocoo()
        percentile_99 = np.percentile(temp_matrix.data, 99)
        temp_matrix.data[temp_matrix.data > percentile_99] = 0
        sparse_distance_matrix = temp_matrix.tocsr()

        # Invert and exponentiate non-zero values
        non_zero_values = sparse_distance_matrix.data[sparse_distance_matrix.data > 0]
        min_non_zero_value = np.min(non_zero_values) if non_zero_values.size > 0 else 1

        if platform == 'visium':
            sparse_distance_matrix.setdiag(min_non_zero_value / 2)
        else:
            sparse_distance_matrix.setdiag(min_non_zero_value)

        inverse_sq_data = np.zeros_like(sparse_distance_matrix.data)
        inverse_sq_data[sparse_distance_matrix.data > 0] = 1 / (sparse_distance_matrix.data[sparse_distance_matrix.data > 0] ** beta)

        inverse_sq_matrix = csr_matrix((inverse_sq_data, sparse_distance_matrix.indices, sparse_distance_matrix.indptr),
                                        shape=sparse_distance_matrix.shape)

        row_sums = inverse_sq_matrix.sum(axis=1).A1
        row_sums[row_sums == 0] = 1
        weights = inverse_sq_matrix.multiply(1 / row_sums[:, np.newaxis])

        return weights

    @staticmethod
    def clustering(adata, n_clusters=7, n_neighbors=10, key='X_pca_smoothed', method='mclust'):
        """
        Spatial clustering.

        Parameters:
        - adata: AnnData object of scanpy package.
        - n_clusters: int, optional - The number of clusters. Default is 7.
        - n_neighbors: int, optional - The number of neighbors considered during refinement. Default is 15.
        - key: string, optional - The key of the learned representation in adata.obsm. Default is 'X_pca_smoothed'.
        - method: string, optional - The tool for clustering. Supported tools: 'mclust', 'leiden', 'louvain'.

        Returns:
        - adata: Updated AnnData object with clustering results.
        """

        if method == 'mclust':
            np.random.seed(2020)
            import rpy2.robjects as robjects
            robjects.r.library("mclust")
            import rpy2.robjects.numpy2ri
            rpy2.robjects.numpy2ri.activate()
            r_random_seed = robjects.r['set.seed']
            r_random_seed(2020)
            rmclust = robjects.r['Mclust']
            res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[key]), n_clusters, 'EEE')
            mclust_res = np.array(res[-2])
            adata.obs[f'RASP_{method}_clusters'] = mclust_res
            adata.obs[f'RASP_{method}_clusters'] = adata.obs[f'RASP_{method}_clusters'].astype('int')
            adata.obs[f'RASP_{method}_clusters'] = adata.obs[f'RASP_{method}_clusters'].astype('category')

        elif method == 'louvain':
            adata = RASP.louvain(adata, n_clusters, n_neighbors=n_neighbors, key_added='RASP_louvain_clusters')

        elif method == 'leiden':
            adata = RASP.leiden(adata, n_clusters, n_neighbors=n_neighbors, key_added='RASP_leiden_clusters')

        elif method =="walktrap":
                neighbors_graph = adata.obsp['connectivities']
                sources, targets = neighbors_graph.nonzero()
                weights = neighbors_graph[sources, targets].A1
                g = ig.Graph(directed=False)
                g.add_vertices(adata.n_obs)
                g.add_edges(zip(sources, targets))
                g.es['weight'] = weights
            
                # Perform Walktrap community detection
                start_time = time.time()
                walktrap = g.community_walktrap(weights='weight')
                clusters = walktrap.as_clustering(n=n_clusters)
                end_time = time.time()
                cluster_time = end_time - start_time
                adata.obs[f'RASP_{method}_clusters'] = pd.Categorical(clusters.membership)

        elif method == "KMeans":
            kmeans = KMeans(n_clusters = n_clusters,random_state = 10)
            adata.obs[f'RASP_{method}_clusters'] = pd.Categorical(kmeans.fit_predict(adata.obsm['X_pca_smoothed']))

        num_clusters = len(set(adata.obs[f'RASP_{method}_clusters']))
        palette = glasbey.create_palette(palette_size=num_clusters)
        adata.uns[f'RASP_{method}_clusters_colors'] = palette

        return adata

    @staticmethod
    def louvain(adata,n_clusters,n_neighbors = 10,use_rep = 'X_pca_smoothed',key_added = 'RASP_louvain_clusters',random_seed = 2023):
        res = RASP.res_search_fixed_clus_louvain(adata, n_clusters, increment=0.1, start = 0.001,random_seed=random_seed)
        print(f'resolution is: {res}')
        sc.tl.louvain(adata, random_state=random_seed, resolution=res)
       
        adata.obs[key_added] = adata.obs['louvain']
        adata.obs[key_added] = adata.obs[key_added].astype('int')
        adata.obs[key_added] = adata.obs[key_added].astype('category')

        return adata

    @staticmethod
    def leiden(adata,n_clusters,n_neighbors = 10,use_rep = 'X_pca_smoothed',key_added = 'RASP_leiden_clusters',random_seed = 2023):
        res = RASP.res_search_fixed_clus_leiden(adata, n_clusters, increment=0.1, start = 0.001,random_seed=random_seed)
        print(f'resolution is: {res}')
        sc.tl.leiden(adata, random_state=random_seed, resolution=res)
       
        adata.obs[key_added] = adata.obs['leiden']
        adata.obs[key_added] = adata.obs[key_added].astype('int')
        adata.obs[key_added] = adata.obs[key_added].astype('category')

        return adata

    @staticmethod
    def res_search_fixed_clus_louvain(adata, n_clusters, increment=0.1, start=0.001, random_seed=2023):
        """
        Search for the correct resolution for the Louvain clustering algorithm.

        Parameters:
        - adata: AnnData object containing the data.
        - n_clusters: int - The target number of clusters.
        - increment: float, optional - The step size for resolution search (default is 0.1).
        - start: float, optional - The starting resolution for the search (default is 0.001).
        - random_seed: int, optional - Random seed for reproducibility (default is 2023).

        Returns:
        - float: The largest correct resolution found for the specified number of clusters.
        """
        if increment < 0.0001:
            print("Increment too small, returning starting value.")
            return start  # Return the initial start value
        #keep track of the currect resolution and the largest resolution that is not to large. 
        largest_correct_res = None
        current_res = start
        for res in np.arange(start,2,increment):
            sc.tl.louvain(adata,random_state = random_seed,resolution = res)
            
            #increase res tracker to current res
            current_res = res

            
            num_clusters = len(adata.obs['louvain'].unique())
            print(f'Resolution: {res} gives cluster number: {num_clusters}')

            if num_clusters == n_clusters:
                largest_correct_res = res  # Update the largest correct resolution found
            
            #Check to see if the res resulted in too many clusters! 
            #break out of loop if we exceed this point. 
            if num_clusters > n_clusters:
                break

        
        #return correct res if you have one! 
        if largest_correct_res is not None:
            return largest_correct_res

        #perform tail end recursion until correct res is found! 
        else:
            return RASP.res_search_fixed_clus_louvain(
                adata,
                n_clusters,
                increment = increment/10,
                start = current_res - increment,
                random_seed = random_seed)


    @staticmethod
    def res_search_fixed_clus_leiden(adata, n_clusters, increment=0.1, start=0.001, random_seed=2023):
        """
        Search for the correct resolution for the Leiden clustering algorithm.

        Parameters:
        - adata: AnnData object containing the data.
        - n_clusters: int - The target number of clusters.
        - increment: float, optional - The step size for resolution search (default is 0.1).
        - start: float, optional - The starting resolution for the search (default is 0.001).
        - random_seed: int, optional - Random seed for reproducibility (default is 2023).

        Returns:
        - float: The largest correct resolution found for the specified number of clusters.
        """
        if increment < 0.0001:
            print("Increment too small, returning starting value.")
            return start  # Return the initial start value
        #keep track of the currect resolution and the largest resolution that is not to large. 
        largest_correct_res = None
        current_res = start
        for res in np.arange(start,2,increment):
            sc.tl.leiden(adata,random_state = random_seed,resolution = res)
            
            #increase res tracker to current res
            current_res = res

            
            num_clusters = len(adata.obs['leiden'].unique())
            print(f'Resolution: {res} gives cluster number: {num_clusters}')

            if num_clusters == n_clusters:
                largest_correct_res = res  # Update the largest correct resolution found
            
            #now check to see if the res resulted in too many clusters! 
            #break out of loop if we exceed this point. 
            if num_clusters > n_clusters:
                break

        
        #return correct res if you have one! 
        if largest_correct_res is not None:
            return largest_correct_res

        #perform tail end recursion until correct res is found! 
        else:
            return RASP.res_search_fixed_clus_leiden(
                adata,
                n_clusters,
                increment = increment/10,
                start = current_res - increment,
                random_seed = random_seed)
    
    
   

    

    @staticmethod
    def fx_1NN(index, location_in):
        """
        Python equivalent of the fx_1NN function that is called in the loop.
        Computes the distance from the point at 'index' to its nearest neighbor.
        """
        distances = cdist([location_in[index]], location_in, 'euclidean')
        nearest_neighbor = np.partition(distances, 1)[0, 1]  # 1st closest distance
        return nearest_neighbor
    @staticmethod
    def CHAOS(clusterlabel, location):
        matched_location = np.array(location)
        clusterlabel = np.array(clusterlabel)
        
        # Remove NA (None) values
        NAs = np.where(pd.isna(clusterlabel))[0]
        if len(NAs) > 0:
            clusterlabel = np.delete(clusterlabel, NAs)
            matched_location = np.delete(matched_location, NAs, axis=0)
    
        # Standardize the location data
        matched_location = scale(matched_location)
    
        unique_labels = np.unique(clusterlabel)
        dist_val = np.zeros(len(unique_labels))
        
        for count, k in enumerate(unique_labels):
            location_cluster = matched_location[clusterlabel == k]
            if location_cluster.shape[0] == 1:  # Only one point in cluster
                continue
    
            with Pool(5) as pool:  # Parallel processing with 5 cores
                results = pool.starmap(RASP.fx_1NN, [(i, location_cluster) for i in range(location_cluster.shape[0])])
            
            dist_val[count] = sum(results)
        
        dist_val = dist_val[~np.isnan(dist_val)]  # Remove any NaN values
        return np.sum(dist_val) / len(clusterlabel)


    @staticmethod
    def reconstruct_gene(adata, 
                                smoothed_pca_matrix, 
                                weights,
                                gene_name='test', 
                                quantile_prob=0.001,
                                scale = False,
                                threshold_method = 'ALRA',
                                rank_k = 20):

 
        """
        Restore true biological zeros while considering excess zeros and apply scaling.
        
        Parameters:
        - adata: AnnData object containing the gene expression data.
        - smoothed_pca_one: PCA smoothed data after initial PCA.
        - smoothed_pca_two: PCA smoothed data after adding features.
        - pca_weights_initial: Weights from the initial PCA.
        - pca_weights_final: Weights from the final PCA (optional).
        - gene_name: The specific gene for which to restore zeros.
        - quantile_prob: The quantile threshold to use for determining biological zeros.
        - plot_hist: Bool indicator to output the histogram of the gene expression before and after reconstruction
        - scale: Bool indicator to scale values to match original expression
        - threshold_method: ALRA or Zero, how to deal with restoration of biological zeros to the imputed data. 
        
        Returns:
        - adata: Updated AnnData object with reconstructed zeros.
        """
        
        # Get the original gene expression data
        original_data = adata.X.toarray()
        indices = range(rank_k)
        gene_index = adata.var.index.get_loc(gene_name)
        original_expression = original_data[:, gene_index].toarray().flatten() if isinstance(original_data, csr_matrix) else original_data[:, gene_index]
    
        
    
            #subset to get rank k reconstruction: 
        indices = range(rank_k)
        smoothed_pca_matrix = smoothed_pca_matrix[:,indices]
    
        gene_weights = weights[indices, gene_index]
        reconstructed_gene_expression = np.dot(smoothed_pca_matrix, gene_weights)
    
        delta_mean = np.mean(original_expression)
        reconstructed_gene_expression += delta_mean
    
        # Calculate the quantile threshold using absolute value
        #note: the ALRA method uses the abs of the quantile and then restores the expression of some cell cells that are non-zero 
        # from the original expression matrix. This is different than what I am doing which is taking whatever is smaller: the threshold or 
        # zero. 
    
    
        if threshold_method == 'Zero':
            threshold_value = np.quantile(reconstructed_gene_expression, quantile_prob)
            threshold_value = max(0,threshold_value)
        
            print(f'Threshold read value: {np.quantile(reconstructed_gene_expression, quantile_prob)}')
            
            
            # Restore the biological zeros based on the excess zeros logic
            restored_expression = reconstructed_gene_expression.copy()
            print(f'Number of cells below the threshold: {np.sum(restored_expression < threshold_value)}')
            print(f'Number of cells below zero: {np.sum(restored_expression < 0)}')
        
            restored_expression[restored_expression < threshold_value] = 0
    
            
        
            #in case negative values remain, set those to zero as well! 
            #restored_expression[restored_expression < 0] = 0 
        
            print(f'Number of cells with zero before imputation:{np.sum(original_expression==0)}')
            print(f'Number of cells with zero AFTER imputation:{np.sum(restored_expression==0)}')
    
        if threshold_method == 'ALRA':
            threshold_value =  np.abs(np.quantile(reconstructed_gene_expression, quantile_prob))
            print(f'Threshold (absolute value for ALRA method): {threshold_value}')
            restored_expression = reconstructed_gene_expression.copy()
            print(f'Number of cells below the threshold: {np.sum(restored_expression < threshold_value)}')
            print(f'Number of cells below zero: {np.sum(restored_expression < 0)}')
            restored_expression[restored_expression < threshold_value] = 0
            
            # Restore original values for Non-Zero entries that were thresholded out
            mask_thresholded_to_zero = (reconstructed_gene_expression < threshold_value) & (original_expression > 0)
    
            #note: the ALRA method restors the original expression here. What I am doing is instead restoring the 
            #reconstructed expression, as long as it is not zero! 
            #restored_expression[mask_thresholded_to_zero] = original_expression[mask_thresholded_to_zero]
            restored_expression[mask_thresholded_to_zero] = reconstructed_gene_expression[mask_thresholded_to_zero]
            print(f'Number of cells restored to original values:{np.sum(mask_thresholded_to_zero != 0)}')
            print(f'Number of cells that where negative: {np.sum(reconstructed_gene_expression[mask_thresholded_to_zero]<0)}')
    
            #finally, set anything that is still negative to zero, should be a very small number of cells! 
            restored_expression[restored_expression < 0] = 0
            
        if scale:
            
    
            # Now, perform scaling based on the original and restored values
            sigma_1 = np.std(restored_expression[restored_expression > 0])
            sigma_2 = np.std(original_expression[original_expression > 0])
            mu_1 = np.mean(restored_expression[restored_expression > 0])
            mu_2 = np.mean(original_expression[original_expression > 0])
        
            # Avoid division by zero
            if sigma_1 == 0:
                sigma_1 = 1e-10  # Or choose to keep restored_expression intact
            
            # Determine scaling factors
            scaling_factor = sigma_2 / sigma_1
            offset = mu_2 - (mu_1 * scaling_factor)
        
            # Apply scaling
            restored_expression = restored_expression * scaling_factor + offset
        
            # If case scaling results in any negative values, turn those to zero as well! 
            #print(f'Number of cells turned negative after scaling: {np.sum(restored_expression_scaled < 0)}')
            restored_expression[restored_expression < 0] = 0
            
    
        # Store the final restored gene expression back into adata
        adata.obs['restored_' + gene_name] = restored_expression.flatten()
            
        return adata


#functions: 
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


def get_leverage_index(adata,fraction = 0.1,score_column="gene_score",seed = 0):
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

def uniform_sample_adata(adata, fraction=0.1, random_state=None):
    """
    Uniformly samples a fraction of cells from an AnnData object.
    
    Parameters:
    - adata: AnnData object
    - fraction: float, fraction of cells to sample (default is 10%)
    - random_state: int, random seed for reproducibility (default is None)
    
    Returns:
    - AnnData object with sampled cells
    """
    print("running uniform sampling")
    np.random.seed(random_state)
    num_cells = adata.n_obs  # Total number of cells
    sample_size = int(num_cells * fraction)  # Compute number of cells to sample

    sampled_indices = np.random.choice(adata.obs.index, size=sample_size, replace=False)
    
    return adata[sampled_indices].copy()  # Return a new AnnData object


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


def compute_full_metrics(adata,
	fraction = 0.1,
	seed = 0, 
	base_directory = "/test/dir", 
	ground_truth = 'celltype',
	knn_for_neighborhood_analysis = 10,
	n_neighbors_for_clustering = 10,
	cluster_algorithm = 'louvain',n_clusters = 15):


    method_index_dict = {
        "rasp_geo":"rasp_geo_index",
        "rasp_scsampler":"rasp_scsampler_index",
        "rasp_leverage":"rasp_leverage_index",
        
    }
    
    
    results = {}
    full_pca = PCA(n_components = 20, svd_solver = 'randomized',random_state = 2024)
    full_pca.fit(adata.X.toarray())

    #full_pca.fit(adata.X)
    
    #cell_type_proportions_full = compute_cell_type_proportions_annoy(adata, adata,ground_truth = ground_truth,k = knn_for_neighborhood_analysis)
    
    for method, child_dir in method_index_dict.items():
        if child_dir != "uniform_index":
            index_df = load_csv_by_fraction(os.path.join(base_directory,child_dir),target_fraction = fraction)
            #sort the columns for easier access next time: 
            index_df = index_df.reindex(columns = sorted(index_df.columns))
        elif child_dir == "uniform_index": 
            index_df = generate_uniform_index_df(adata,fraction = fraction,random_seeds = range(10))
        if method not in results:
            results[method] = {}
        
        
        column_name = f"seed_{seed}"
        if column_name in index_df.columns:
        	index_values = index_df[column_name].values
        	sampled_adata = adata[index_values]

        #next we will iterate over the rows of the dataframe to subsample. 
        #for column_name, index_values in index_df.items():
    
            #sampled_adata = adata[index_values.values]
    
            #neighborhood metrics: 
            # cell_type_proportions_sub = compute_cell_type_proportions_annoy(sampled_adata, adata,ground_truth = ground_truth,k = knn_for_neighborhood_analysis)
            # cell_type_proportions_full_partial = cell_type_proportions_full.loc[adata.obs.index[index_values.values]]
            # metrics = compare_cell_type_proportions(cell_type_proportions_sub, cell_type_proportions_full_partial)
            
            #now for the metrics: 
            #compute the cell type metrics 
            #comput the two distance metrics! 
        transcriptomic_distance=partial_hausdorff_distance_annoy(adata.X.toarray(), sampled_adata.X.toarray(), q=1e-4)
        #transcriptomic_distance=partial_hausdorff_distance_annoy(adata.X, sampled_adata.X, q=1e-4)

        coord_distance = partial_hausdorff_distance_annoy(adata.obsm['spatial'], sampled_adata.obsm['spatial'], q=1e-4)
        
        results[method]['transcriptomic_distance'] = transcriptomic_distance
        results[method]['coord_distance'] = coord_distance
                #now we need to calcualte 2 ARIs: The ARI on the full dataset PCA solution (subset)
        # and the ARI on the recomputed PCA solution. 
        #first lets do the original PCA clustering: 
        #note: only calculate the ARI values if seed==0, otherwise we won't need to recompute it. 
        #add nans to the ARi otherwise: 
        #seed = int(column_name.replace("seed_",""))
        if seed ==0:
            sc.pp.neighbors(sampled_adata, n_neighbors=n_neighbors_for_clustering, use_rep='X_pca')
            sampled_adata = RASP.clustering(sampled_adata, n_clusters=n_clusters, n_neighbors=n_neighbors_for_clustering, key='X_pca', method=cluster_algorithm)
    
            # Get true and predicted labels
            ground_truth_labels = sampled_adata.obs[ground_truth].astype(str)
            labels = sampled_adata.obs[f'RASP_{cluster_algorithm}_clusters'].astype(str)
    
            # Compute ARI
            ari_original_pca = adjusted_rand_score(ground_truth_labels, labels)
            results[method]['ari_original_pca'] = ari_original_pca
    
            #now we will re-compute the PCA and re-cluster. 
            pca_model_sketch = PCA(n_components=20, svd_solver='randomized', random_state=2024)
    
            pca_data = pca_model_sketch.fit_transform(sampled_adata.X.toarray())
            #pca_data = pca_model_sketch.fit_transform(sampled_adata.X)
            sampled_adata.obsm['X_pca_recomputed'] = pca_data
            sc.pp.neighbors(sampled_adata, n_neighbors=n_neighbors_for_clustering, use_rep='X_pca_recomputed')
            sampled_adata = RASP.clustering(sampled_adata, n_clusters=n_clusters, n_neighbors=n_neighbors_for_clustering, key='X_pca_recomputed', method=cluster_algorithm)
            labels = sampled_adata.obs[f'RASP_{cluster_algorithm}_clusters'].astype(str)
            ari_recomputed_pca = adjusted_rand_score(ground_truth_labels, labels)
            results[method]['ari_recomputed_pca'] = ari_recomputed_pca
        
        #Now for the PCA distances 
        S = sampled_adata.X.toarray()
        #S = sampled_adata.X
        # full_pca = PCA(n_components = 20, svd_solver = 'randomized',random_state = 2024)
        # full_pca.fit(adata.X.toarray())
        P_f = full_pca.transform(S)
        # Project S onto PCs of sampled data (sampled_adata)
        sketch_pca = PCA(n_components=20, svd_solver='randomized', random_state=2024)
        sketch_pca.fit(S)
        P_s = sketch_pca.transform(S)

        #Step 5: compute euclidean distance between rows of P_f and P_s
        distances = np.linalg.norm(P_f-P_s,axis = 1)
        mean_distance = np.mean(distances)
        median_distance = np.median(distances)

        results[method]['pca_mean_diff'] = mean_distance
        results[method]['pca_median_diff'] = median_distance


        #save the other function parameters as well: seed, k, and fraction 
        results[method]['knn_for_neighborhood_analysis'] = knn_for_neighborhood_analysis
        results[method]['seed'] = seed
        results[method]['fraction'] = fraction
        # Convert results dictionary to DataFrame
    results_df = pd.DataFrame.from_dict(results,orient = 'index').reset_index()
    results_df.rename(columns = {results_df.columns[0]:'method'},inplace = True)
    return results_df