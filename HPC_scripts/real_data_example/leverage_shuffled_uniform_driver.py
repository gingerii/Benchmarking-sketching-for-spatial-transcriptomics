print("starting script")
from functions import *
import argparse
print("imported functions")



parser = argparse.ArgumentParser(description="Run leverage driver with specified parameters.")
parser.add_argument('--fraction', type=float, required=True, help='Fraction parameter')
args = parser.parse_args()


fraction = args.fraction

print(f"Running with: fraction={fraction}")

directory = "/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/processed_data"
adata = sc.read_h5ad(os.path.join(directory,"xenium_cancer_processed.h5ad"))


#re-run PCA for now: 
pca_model = PCA(n_components=20, svd_solver='randomized', random_state=2024)
pca_data = pca_model.fit_transform(adata.X.toarray())
adata.obsm['X_pca'] = pca_data


gene_scores_df = pd.read_csv('/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/processed_data/leverage_scores.csv')
adata.obs['gene_score'] = gene_scores_df['leverage_score'].values


#get all the indices 
geo_shuffled_index = load_csv_by_fraction("/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/sketching_test/index/geo_shuffled",
                           target_fraction = fraction)
scsampler_shuffled_index = load_csv_by_fraction("/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/sketching_test/index/scsampler_shuffled",
                           target_fraction = fraction)


uniform_index = create_uniform_index_dataframe(adata,fraction = fraction,num_seeds = 10, start_seed = 0)
leverage_index=create_leverage_index_dataframe(adata, fraction=fraction,num_seeds = 10,start_seed = 0)



dataframe = evaluate_subsampling_methods(
	adata, 
	geo_shuffled_index, 
	scsampler_shuffled_index, 
	uniform_index, 
	leverage_index, 
	ground_truth = 'region',
	knn_for_neighborhood_analysis=100,
	n_neighbors_for_clustering=10,
	n_clusters=8,
	cluster_algorithm='leiden',
	fraction=fraction
	)
output_file = f"/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/sketching_test/combined_metrics/normal_metrics_{fraction}_part3.csv"
dataframe.to_csv(output_file,index = False)
print(f"Saved fraction {fraction}")