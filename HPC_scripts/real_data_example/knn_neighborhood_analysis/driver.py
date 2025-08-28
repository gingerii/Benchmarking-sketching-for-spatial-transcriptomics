from functions import *
import argparse
print("imported functions")



parser = argparse.ArgumentParser(description="Run leverage driver with specified parameters.")
parser.add_argument('--fraction', type=float, required=True, help='Fraction parameter')
args = parser.parse_args()
fraction = args.fraction
print(f"Running with: fraction={fraction}")



#data 
directory = "/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/processed_data"
adata = sc.read_h5ad(os.path.join(directory,"xenium_cancer_processed.h5ad"))



#re-run PCA for now: 
pca_model = PCA(n_components=20, svd_solver='randomized', random_state=2024)
pca_data = pca_model.fit_transform(adata.X.toarray())
adata.obsm['X_pca'] = pca_data


gene_scores_df = pd.read_csv('/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/processed_data/leverage_scores.csv')
adata.obs['gene_score'] = gene_scores_df['leverage_score'].values
output_file = f"/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/sketching_test/knn_neighborhood_analysis/normal_methods_{fraction}.csv"
everything_df = pd.DataFrame()
for seed in range(10):
	for k in [2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:
		# Create leverage dataframe
		leverage_df = create_leverage_index_dataframe(adata, fraction=fraction, num_seeds=10, start_seed=0)

		base =  f"/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/sketching_test/index"
		# Construct file paths dynamically based on `fraction`
		scsampler_coord_path = os.path.join(base,f"scsampler_coords/index_{fraction}.csv")
		scsampler_trans_path = os.path.join(base,f"scsampler_transcriptomic/index_{fraction}.csv")
		scsampler_shuffled_path = os.path.join(base,f"scsampler_shuffled/index_{fraction}.csv")
		geo_coord_path = os.path.join(base,f"geo_coords/index_{fraction}.csv")
		geo_trans_path = os.path.join(base,f"geo_transcriptomic/index_{fraction}.csv")
		geo_shuffled_path = os.path.join(base,f"geo_shuffled/index_{fraction}.csv")



		# Read in the subset indices
		scsampler_coord_df = pd.read_csv(scsampler_coord_path)
		scsampler_trans_df = pd.read_csv(scsampler_trans_path)
		scsampler_shuffled_df = pd.read_csv(scsampler_shuffled_path)
		geo_coord_df = pd.read_csv(geo_coord_path)
		geo_trans_df = pd.read_csv(geo_trans_path)
		geo_shuffled_df = pd.read_csv(geo_shuffled_path)


		# Get uniform index
		uniform = uniform_sample_adata(adata, fraction=fraction, random_state=seed)

		# Create sub-data objects
		scsampler_coords = adata[scsampler_coord_df[f'seed_{seed}'].values]
		scsampler_trans = adata[scsampler_trans_df[f'seed_{seed}'].values]
		scsampler_shuffled = adata[scsampler_shuffled_df[f'seed_{seed}'].values]
		geo_coords = adata[geo_coord_df[f'seed_{seed}'].values]
		geo_trans = adata[geo_trans_df[f'seed_{seed}'].values]
		geo_shuffled = adata[geo_shuffled_df[f'seed_{seed}'].values]
		leverage = adata[leverage_df[f'seed_{seed}'].values]



		# Compute spatial neighbors for each subset
		subsets = [scsampler_coords, scsampler_trans,scsampler_shuffled ,uniform, geo_coords, geo_trans,geo_shuffled, leverage]
		subset_dict = {name: obj for name, obj in zip(['scsampler_coords', 'scsampler_trans', 'scsampler_shuffled', 
		                                               'uniform', 'geo_coords', 'geo_trans', 'geo_shuffled', 'leverage'], subsets)}



		print("Computing cell type proportions for full dataset...")
		start_time = time.time()
		cell_type_proportions_full = compute_cell_type_proportions_annoy(adata, adata,ground_truth = 'region', k = k)

		results = {}
		methods = []

		for method,subset in subset_dict.items(): 
		    cell_type_proportions_sub = compute_cell_type_proportions_annoy(subset, adata,ground_truth = 'region', k = k)
		    # Extract corresponding subset from full dataset
		    valid_indices = adata.obs.index.intersection(subset.obs.index)
		    cell_type_proportions_full_partial = cell_type_proportions_full.loc[valid_indices]
		    # Compute similarity metrics
		    metrics = compare_cell_type_proportions(cell_type_proportions_sub, cell_type_proportions_full_partial)
		    metrics['fraction'] = fraction
		    metrics['seed'] = seed
		    metrics['knn'] = k
		    results[method] = metrics
		    
		results_df = pd.DataFrame.from_dict(results,orient = 'index').reset_index()
		results_df.rename(columns = {results_df.columns[0]:'method'},inplace = True)
		everything_df= pd.concat([everything_df,results_df],ignore_index = True)
		everything_df.to_csv(output_file,index=False)
		print(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")