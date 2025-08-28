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


#re-run PCA for now: 
pca_model = PCA(n_components=20, svd_solver='randomized', random_state=2024)
pca_data = pca_model.fit_transform(adata.X.toarray())
adata.obsm['X_pca'] = pca_data


parent_dir = '/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/sketching_test/index'

sub_dirs = {
    'uniform': uniform_index,
    'geo_shuffled': geo_coord_shuffled_index,
    'scsampler_shuffled':scsampler_coord_shuffled_index,
    'geo_coords': geo_coord_index,
    'geo_transcriptomic': geo_transcriptomic_index,
    'scsampler_coords': scsampler_coord_index,
    'scsampler_transcriptomic': scsampler_transcriptomics_index
}

# Fraction and seeds
# fractions = [0.01, 0.02, 0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
# seeds = range(10)  


for sub_dir, index_function in sub_dirs.items():
	result_lists = {f'seed_{seed}': []}  # Create an empty dictionary for results


	# You need to define `adata` appropriately (not shown)
	result = index_function(adata, fraction=fraction, seed=seed)
	result_lists[f'seed_{seed}'] = result  # Store result in the dictionary under the appropriate seed key

	# Create a DataFrame from the result lists
	df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in result_lists.items()]))

	# Save DataFrame to CSV in corresponding subdirectory
	output_file_path = os.path.join(parent_dir, sub_dir, f'index_seed_{seed}_{fraction}.csv')
	df.to_csv(output_file_path, index=False)

	print(f'Results saved in {output_file_path}')




