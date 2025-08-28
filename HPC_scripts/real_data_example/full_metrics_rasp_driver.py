print("starting script")
from functions import *
import argparse
print("imported functions")



parser = argparse.ArgumentParser(description="Run leverage driver with specified parameters.")
#parser.add_argument('--knn', type=float, required=True, help='knn parameter')
parser.add_argument('--beta', type=float, required=True, help='beta parameter')
parser.add_argument('--fraction', type=float, required=True, help='Fraction parameter')



args = parser.parse_args()

knn = 60 #int(args.knn) #hard code the knn for now. 
beta = args.beta
fraction = args.fraction

print(f"Running with: fraction={fraction}")

directory = "/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/processed_data"
adata = sc.read_h5ad(os.path.join(directory,"xenium_cancer_processed.h5ad"))


#re-run PCA for now: 
pca_model = PCA(n_components=20, svd_solver='randomized', random_state=2024)
pca_data = pca_model.fit_transform(adata.X.toarray())
adata.obsm['X_pca'] = pca_data


# Define the output file path where results will be stored
output_file = f"/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/sketching_test/combined_metrics/rasp_metrics_knn_{knn}_beta_{beta}_{fraction}.csv"

# Initialize an empty DataFrame to gather all results
all_results = pd.DataFrame()

# Loop through 10 iterations, changing the seed each time
# for i in range(10):
#     print(f"Iteration {i+1}/10")
for i in range(3):
	single_df = compute_sketching_metrics(adata, 
		fraction=fraction, 
		seed=i, 
		knn_for_neighborhood_analysis=100, 
		ground_truth="region", 
		n_neighbors_for_clustering = 10,
		n_clusters = 8,
		cluster_algorithm = 'leiden',
		rasp = True,
		knn_for_rasp = knn,
		beta = beta,
		platform = 'xenium',
		n_trees=10)

	all_results = pd.concat([all_results, single_df], ignore_index=True)

	# Save the concatenated results to a CSV file
	all_results.to_csv(output_file, index=False)
	print(f"Saved iteration {i+1} results to {output_file}")

# Indicate that the loop has completed
print("Loop completed. All results saved.")

