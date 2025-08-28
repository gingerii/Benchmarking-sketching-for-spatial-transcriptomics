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
        original_data = adata.X
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


#lets try the subsampling on a 
# need implimentation of Robust hausdorff distance computation 
from scipy.spatial.distance import directed_hausdorff
def partial_hausdorff_distance(array1, array2, q=1e-4):
    """
    Compute the partial Hausdorff distance from array1 to array2.
    
    Parameters:
    - array1: numpy.ndarray, shape (n, d)
        The first array of points (e.g., representing the full dataset).
    - array2: numpy.ndarray, shape (m, d)
        The second array of points (e.g., representing the subset).
    - q: float, optional
        The parameter for the partial Hausdorff distance, should be between 0 and 1.
        q=0 corresponds to the classical Hausdorff distance, q=1 would correspond to
        considering up to the smallest distance, practically ineffective in most uses.
    
    Returns:
    - float
        The partial Hausdorff distance based on q between array1 and array2.
    """
    
    # Calculate distances from each point in array1 to the nearest point in array2
    # distances = [directed_hausdorff([point], array2)[0] for point in tqdm(array1)]
    
    # # Sort the distances
    # distances.sort()
    
    # # Calculate the index for the Kth largest value
    # K = int(np.floor((1 - q) * len(distances)))
    
    # # Return the Kth largest value from the sorted distances
    # return distances[K-1] if K > 0 else distances[0]

    # Build KD-Tree for array2
    print("building KDTree")
    start = time.time()
    tree = KDTree(array2)
    end = time.time()
    print(f"KDTree took {end-start:.4f} seconds to build")
    # Query the tree with all points in array1 to find nearest neighbors efficiently
    print("query tree")
    start = time.time()
    distances, _ = tree.query(array1, k=1)
    end = time.time()
    print(f"distance calculation took: {end-start:.4f} seconds")

    # Sort all computed distances
    distances.sort()

    # Determine the index of the Kth largest value for partial Hausdorff
    K = int(np.floor((1 - q) * len(distances)))

    # Handle edge cases and return the Kth largest value
    return distances[K-1] if K > 0 else distances[0]


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

def probability_sample_adata(adata, fraction=0.1, score_column="gene_score", random_state=None):
    """
    Samples a fraction of cells from an AnnData object using weighted probabilities.

    Parameters:
    - adata: AnnData object
    - fraction: float, fraction of cells to sample (default is 10%)
    - score_column: str, column in adata.obs containing sampling probabilities
    - random_state: int, random seed for reproducibility (default is None)

    Returns:
    - AnnData object with sampled cells
    """
    print("running leverage sampling")
    np.random.seed(random_state)

    if score_column not in adata.obs:
        raise ValueError(f"Column '{score_column}' not found in adata.obs")

    scores = adata.obs[score_column].values
    scores = np.clip(scores, a_min=0, a_max=None)  # Ensure no negative values
    probabilities = scores / scores.sum()  # Normalize to get probabilities

    num_cells = adata.n_obs
    sample_size = int(num_cells * fraction)

    sampled_indices = np.random.choice(adata.obs.index, size=sample_size, replace=False, p=probabilities)

    return adata[sampled_indices].copy()

def probability_sample_adata_efficient(adata, fraction=0.1, score_column="gene_score", random_state=None):
    """
    Efficiently samples a fraction of cells from an AnnData object using weighted probabilities.
    Parameters:
    - adata: AnnData object
    - fraction: float, fraction of cells to sample (default is 10%)
    - score_column: str, column in adata.obs containing sampling probabilities
    - random_state: int, random seed for reproducibility (default is None)
    Returns:
    - AnnData object with sampled cells
    """
    print("running efficient leverage sampling")
    np.random.seed(random_state)

    # Validate input and obtain scores
    if score_column not in adata.obs:
        raise ValueError(f"Column '{score_column}' not found in adata.obs")
    scores = adata.obs[score_column].values

    # Ensure non-negative scores and calculate probabilities
    scores = np.clip(scores, a_min=0, a_max=None)
    probabilities = scores / scores.sum()

    # Sample indices with weights using reservoir sampling
    num_cells = adata.n_obs
    sample_size = int(num_cells * fraction)
    sample_indices = []

    print("Start iterating over data")
    for i in range(num_cells):
        if len(sample_indices) < sample_size:
            sample_indices.append(i)
        else:
            j = np.random.randint(0, i + 1)
            if j < sample_size:
                sample_indices[j] = i
    print("get sample indices")
    sampled_indices = adata.obs.index[sample_indices]
    print("return object")
    return adata[sampled_indices].copy()

def evaluate_sampling_methods(adata, fraction=0.1, seed=0, k=20):
    """
    Evaluates different sampling methods for spatial transcriptomics data and computes partial Hausdorff distances.

    Parameters:
    - adata: AnnData object
    - fraction: float, fraction of cells to sample
    - seed: int, random seed for reproducibility
    - k: int, number of principal components for PCA-based geosketching

    Returns:
    - A dictionary containing sampled datasets
    - A pandas DataFrame with computed distances
    """
    start = time.time()
    print("starting eval")
    # Define sampling methods
    sampling_methods = {
        "uniform": lambda adata: uniform_sample_adata(adata, fraction=fraction, random_state=seed),
       "leverage": lambda adata: probability_sample_adata_efficient(adata, fraction=fraction, random_state=seed),
        "scsampler_transcriptomic": lambda adata: scsampler(adata, fraction=fraction, random_state=seed, copy=True,random_split = 16),
    }
    #print("running uniform, leverage and scsampling methods now ")
    # Run sampling methods
    #sampled_data = {name: method(adata) for name, method in sampling_methods.items()}
    sampled_data = {}
    for name, method in sampling_methods.items():
        print(f"Running method: {name}")
        sampled_data[name] = method(adata)
    print("running geosketch on transcripts now")
    # PCA-based geosketching
    U, s, _ = pca(adata.X, k=k)
    X_dimred = U[:, :k] * s[:k]
    N = int(fraction * adata.X.shape[0])
    sampled_data["geo_transcriptomic"] = adata[gs(X_dimred, N, seed=seed, replace=False)].copy()

    print("running coordinate based sampling now")
    # Coordinate-based sampling
    sampled_data["scsampler_coords"] = adata[scsampler(adata.obsm['spatial'], fraction=fraction, random_state=seed, copy=True,random_split = 16)[1]].copy()
    sampled_data["geo_coords"] = adata[gs(adata.obsm['spatial'], N, seed=seed, replace=False)].copy()

    # Compute partial Hausdorff distances
    distance_data = []
    print("starting hausdorff distance computations")
    for name, subset in sampled_data.items():
        distance_data.append({
            "method": name,
            "transcriptomic_distance": partial_hausdorff_distance_annoy(adata.X.toarray(), subset.X.toarray(), q=1e-4),
            "spatial_distance": partial_hausdorff_distance_annoy(adata.obsm['spatial'], subset.obsm['spatial'], q=1e-4),
            "fraction":fraction,
            "seed":seed,
            "k":k
        })

    # Convert to DataFrame
    distance_df = pd.DataFrame(distance_data)

    end = time.time()
    print(f"Sampling and distance computation completed in {end - start:.2f} seconds.")

    return sampled_data, distance_df

def evaluate_sampling_methods_with_ari(adata, fraction = 0.1,
                              seed = 0 , 
                              n_neighbors = 10, 
                              n_clusters = 8, 
                              ground_truth= 'ground_truth',
                              cluster_algorithm='louvain', 
                              k=20):
    # Define sampling methods
    sampling_methods = {
        "uniform": lambda adata: uniform_sample_adata(adata, fraction=fraction, random_state=seed),
        "leverage": lambda adata: probability_sample_adata_efficient(adata, fraction=fraction, random_state=seed),
        "scsampler_transcriptomic": lambda adata: scsampler(adata, fraction=fraction, random_state=seed, copy=True,random_split = 16),
    }

    # Run sampling methods and store results
    sampled_data = {name: method(adata) for name, method in sampling_methods.items()}

    # PCA-based geosketching
    U, s, _ = pca(adata.X, k=k)
    X_dimred = U[:, :k] * s[:k]
    N = int(fraction * adata.X.shape[0])
    sampled_data["geo_transcriptomic"] = adata[gs(X_dimred, N, seed=seed, replace=False)].copy()

    # Coordinate-based sampling
    sampled_data["scsampler_coords"] = adata[scsampler(adata.obsm['spatial'], fraction=fraction, random_state=seed, copy=True,random_split = 16)[1]].copy()
    sampled_data["geo_coords"] = adata[gs(adata.obsm['spatial'], N, seed=seed, replace=False)].copy()

    # Initialize DataFrame to store ARI results
    results = []

    # Evaluate ARI for each sampled dataset
    for method_name, sampled_adata in sampled_data.items():
        sc.pp.neighbors(sampled_adata, n_neighbors=n_neighbors, use_rep='X_pca')
        sampled_adata = RASP.clustering(sampled_adata, n_clusters=n_clusters, n_neighbors=10, key='X_pca', method=cluster_algorithm)

        # Get true and predicted labels
        ground_truth_labels = sampled_adata.obs[ground_truth].astype(str)
        labels = sampled_adata.obs[f'RASP_{cluster_algorithm}_clusters'].astype(str)

        # Compute ARI
        ari = adjusted_rand_score(ground_truth_labels, labels)

        # Append results to the DataFrame
        result = {
            'method': method_name, 
            'fraction':fraction,
            'seed':seed,
            'k':k,
            'ari': ari,
        }
        results.append(result)
    results_df = pd.DataFrame(results)
              

    return results_df

def leverage_score_only_redo(adata,fraction = 0.1,seed = None,k=20): 

    # sample based on full gene set leverage score as the probabilites: 
    adata_leverage_gene_sub = probability_sample_adata(adata, fraction=fraction,score_column='gene_score', random_state=seed)
    trans_distance=partial_hausdorff_distance(adata.X.toarray(), adata_leverage_gene_sub.X.toarray(), q=1e-4)
    coord_distance = partial_hausdorff_distance(adata.obsm['spatial'], adata_leverage_gene_sub.obsm['spatial'], q=1e-4)

    #coord leverage score is a bit difficult to impliment, wait for now. 
    result = pd.DataFrame({
        'method':['leverage'],
        'transcriptomic_distance':[trans_distance],
        'spatial_distance':[coord_distance],
        'fraction':[fraction],
        'seed':[seed],
        'k':[k]
        
    })
    return result


def compute_distances(adata, knn=5, beta=2, platform='merfish', seed=2024, num_pcs=20, fraction=0.1):
    # Handle sparse matrix conversion
    if issparse(adata.X):
        data = adata.X.toarray()
    else:
        data = adata.X

    # Build weights matrix
    weights = RASP.build_weights_matrix(adata, n_neighbors=knn, beta=beta, platform=platform)

    # Perform PCA using fbpca
    from fbpca import pca
    U, s, Vt = pca(adata.X, k=num_pcs)
    X_dimred = U[:, :num_pcs] * s[:num_pcs]

    # Apply weights to PCA results
    smoothed_pca = weights @ csr_matrix(X_dimred)
    adata.obsm['X_pca_smoothed'] = smoothed_pca.toarray()
    # Define number of samples for sketching
    N = int(fraction * adata.X.shape[0])

    # GeoSketch sampling
    sketch_index = gs(smoothed_pca.toarray(), N, seed=seed, replace=False)
    spatial_geo_sub = adata[sketch_index]

    # SCSampler sampling
    res = scsampler(smoothed_pca.toarray(), fraction=fraction, random_state=seed, copy=True,random_split = 16)
    sub_index = res[1]
    scsampler_sketch_coords = adata[sub_index].copy()

    # Compute distances using partial_hausdorff_distance function
    scsampler_transcriptomic_distance = partial_hausdorff_distance_annoy(
        adata.X.toarray(), scsampler_sketch_coords.X.toarray(), q=1e-4)
    scsampler_spatial_distance = partial_hausdorff_distance_annoy(
        adata.obsm['spatial'], scsampler_sketch_coords.obsm['spatial'], q=1e-4)

    geo_transcriptomic_distance = partial_hausdorff_distance_annoy(
        adata.X.toarray(), spatial_geo_sub.X.toarray(), q=1e-4)
    geo_spatial_distance = partial_hausdorff_distance_annoy(
        adata.obsm['spatial'], spatial_geo_sub.obsm['spatial'], q=1e-4)

    

    


    # Store results in a dataframe with separate rows
    results = pd.DataFrame({
        'method': [f'scsampler_RASP_knn_{knn}_beta_{beta}', f'geo_RASP_knn_{knn}_beta_{beta}'],
        'transcriptomic_distance': [scsampler_transcriptomic_distance, geo_transcriptomic_distance],
        'spatial_distance': [scsampler_spatial_distance, geo_spatial_distance],
        'fraction': [fraction, fraction],
        'seed': [seed, seed],
        'num_pcs': [num_pcs, num_pcs],
        'beta': [beta, beta],
        'knn':[knn,knn]
    })

    return results

def RASP_ari(adata, fraction = 0.1,
             seed = 0, 
             n_neighbors = 10,
             n_clusters = 8,
             ground_truth = 'Cell_Type',
             cluster_algorithm = 'louvain',
             knn=5, beta=2, platform='merfish', num_pcs=20,k=20):
    # Handle sparse matrix conversion
    if issparse(adata.X):
        data = adata.X.toarray()
    else:
        data = adata.X

    # Build weights matrix
    weights = RASP.build_weights_matrix(adata, n_neighbors=knn, beta=beta, platform=platform)

    # Perform PCA using fbpca
    from fbpca import pca
    U, s, Vt = pca(adata.X, k=num_pcs)
    X_dimred = U[:, :num_pcs] * s[:num_pcs]

    # Apply weights to PCA results
    smoothed_pca = weights @ csr_matrix(X_dimred)
    adata.obsm['X_pca_smoothed'] = smoothed_pca.toarray()
    # Define number of samples for sketching
    N = int(fraction * adata.X.shape[0])

    # GeoSketch sampling
    sketch_index = gs(smoothed_pca.toarray(), N, seed=seed, replace=False)
    spatial_geo_sub = adata[sketch_index]

    # SCSampler sampling
    res = scsampler(smoothed_pca.toarray(), fraction=fraction, random_state=seed, copy=True)
    sub_index = res[1]
    scsampler_sketch_coords = adata[sub_index].copy()

    sampled_data = {}
    sampled_data[f'geo_RASP_knn_{knn}_beta_{beta}'] = spatial_geo_sub
    sampled_data[f'scsampler_RASP_knn_{knn}_beta_{beta}'] = scsampler_sketch_coords

    
    results = []

    # Evaluate ARI for each sampled dataset
    for method_name, sampled_adata in sampled_data.items():
        sc.pp.neighbors(sampled_adata, n_neighbors=n_neighbors, use_rep='X_pca_smoothed')
        sampled_adata = RASP.clustering(sampled_adata, n_clusters=n_clusters, n_neighbors=10, key='X_pca_smoothed', method=cluster_algorithm)

        # Get true and predicted labels
        ground_truth_labels = sampled_adata.obs[ground_truth].astype(str)
        labels = sampled_adata.obs[f'RASP_{cluster_algorithm}_clusters'].astype(str)

        # Compute ARI
        ari = adjusted_rand_score(ground_truth_labels, labels)

        # Append results to the DataFrame
        result = {
            'method': method_name, 
            'fraction':fraction,
            'seed':seed,
            'k':k,
            'ari': ari,
        }
        results.append(result)
    results_df = pd.DataFrame(results)



    return results_df