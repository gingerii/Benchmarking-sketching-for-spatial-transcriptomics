print("starting script")
from functions import *
import argparse
print("imported functions")


parser = argparse.ArgumentParser(description="Run driver with specified parameters.")
#parser.add_argument('--knn', type=float, required=True, help='knn parameter')
#parser.add_argument('--seed', type=int, required=True, help='seed parameter')
parser.add_argument('--fraction', type=float, required=True, help='Fraction parameter')


args = parser.parse_args()
fraction = args.fraction
#seed = args.seed

directory = "/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/simulations_for_sketching_analysis/visium_like_data/complex/data"
adata = sc.read_h5ad(os.path.join(directory,"processed_data.h5ad"))


#re-run PCA for now: 
# pca_model = PCA(n_components=20, svd_solver='randomized', random_state=2024)
# pca_data = pca_model.fit_transform(adata.X.toarray())
# adata.obsm['X_pca'] = pca_data


#run for the svd leverage score 
gene_scores_df = pd.read_csv('/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/simulations_for_sketching_analysis/visium_like_data/complex/data/leverage_score_svd.csv')
adata.obs['gene_score'] = gene_scores_df['leverage_score_svd'].values

parent_dir = '/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/simulations_for_sketching_analysis/visium_like_data/complex/sketching_test/index'

sub_dirs = {
    'leverage_svd_index': leverage_svd_index
   
 
}


 
for sub_dir, index_function in sub_dirs.items():
	result_lists = {}#{f'seed_{seed}': []}  # Create an empty dictionary for results
	for seed in range(10):

		# You need to define `adata` appropriately (not shown)
		result = index_function(adata, fraction=fraction, seed=seed)
		result_lists[f'seed_{seed}'] = result  # Store result in the dictionary under the appropriate seed key

		# Create a DataFrame from the result lists
	df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in result_lists.items()]))

		# Save DataFrame to CSV in corresponding subdirectory
	output_file_path = os.path.join(parent_dir, sub_dir, f'index_{fraction}.csv')
	df.to_csv(output_file_path, index=False)

	print(f'Results saved in {output_file_path}')



#run again for the smoothed svd 
gene_scores_df = pd.read_csv('/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/simulations_for_sketching_analysis/visium_like_data/complex/data/leverage_score_svd_smoothed.csv')
adata.obs['gene_score'] = gene_scores_df['leverage_score_svd_smoothed'].values

parent_dir = '/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/simulations_for_sketching_analysis/visium_like_data/complex/sketching_test/index'

sub_dirs = {
    'leverage_svd_smoothed_index': leverage_svd_index
   
 
}


 
for sub_dir, index_function in sub_dirs.items():
	result_lists = {}#{f'seed_{seed}': []}  # Create an empty dictionary for results
	for seed in range(10):

		# You need to define `adata` appropriately (not shown)
		result = index_function(adata, fraction=fraction, seed=seed)
		result_lists[f'seed_{seed}'] = result  # Store result in the dictionary under the appropriate seed key

		# Create a DataFrame from the result lists
	df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in result_lists.items()]))

		# Save DataFrame to CSV in corresponding subdirectory
	output_file_path = os.path.join(parent_dir, sub_dir, f'index_{fraction}.csv')
	df.to_csv(output_file_path, index=False)

	print(f'Results saved in {output_file_path}')