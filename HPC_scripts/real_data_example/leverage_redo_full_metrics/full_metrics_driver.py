print("starting script")
from functions import *
import argparse
print("imported functions")


parser = argparse.ArgumentParser(description="Run driver with specified parameters.")
#parser.add_argument('--knn', type=float, required=True, help='knn parameter')
parser.add_argument('--seed', type=int, required=True, help='seed parameter')
parser.add_argument('--fraction', type=float, required=True, help='Fraction parameter')


args = parser.parse_args()
fraction = args.fraction
seed = args.seed

directory = "/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/processed_data"
adata = sc.read_h5ad(os.path.join(directory,"xenium_cancer_processed.h5ad"))



gene_scores_df = pd.read_csv('/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/processed_data/leverage_scores_pca.csv')
adata.obs['gene_score'] = gene_scores_df['leverage_score'].values

parent_dir = '/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/sketching_test/index'





base_directory = '/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/sketching_test/index'
ground_truth = 'region'
knn_for_neighborhood_analysis = 10
n_neighbors_for_clustering = 10
cluster_algorithm = 'leiden'
n_clusters = 8

pca_model = PCA(n_components=20, svd_solver='randomized', random_state=2024)
pca_data = pca_model.fit_transform(adata.X.toarray())
adata.obsm['X_pca'] = pca_data


out_df = compute_full_metrics(adata,
	fraction = fraction, 
	seed = seed,
	base_directory =base_directory, 
	ground_truth = ground_truth,
	knn_for_neighborhood_analysis = knn_for_neighborhood_analysis,
	n_neighbors_for_clustering = n_neighbors_for_clustering,
	cluster_algorithm = cluster_algorithm,
	n_clusters = n_clusters)

out_df.to_csv(f"/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/sketching_test/combined_metrics/leverage_metrics_seed_{seed}_{fraction}.csv")