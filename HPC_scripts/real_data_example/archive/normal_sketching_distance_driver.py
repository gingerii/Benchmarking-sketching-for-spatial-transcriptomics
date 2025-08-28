print("starting script")
from functions import *
import argparse
print("imported functions")



parser = argparse.ArgumentParser(description="Run leverage driver with specified parameters.")
parser.add_argument('--fraction', type=float, required=True, help='Fraction parameter')
parser.add_argument('--seed', type=float, required=True, help='Seed parameter')
args = parser.parse_args()


fraction = args.fraction
seed = int(args.seed)

print(f"Running with: fraction={fraction}, seed={seed}")

directory = "/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/processed_data"
adata = sc.read_h5ad(os.path.join(directory,"xenium_cancer_processed.h5ad"))
sc.pp.highly_variable_genes(adata,n_top_genes = 540)
highly_variable_genes=adata.var_names[adata.var['highly_variable']==True].tolist()
sc.tl.score_genes(adata,gene_list = highly_variable_genes, score_name = 'gene_score')
#add the X_pca 
pca_model = PCA(n_components=20, svd_solver='randomized', random_state=2024)
pca_data = pca_model.fit_transform(adata.X.toarray())
adata.obsm['X_pca'] = pca_data


# knn = 1
# beta = 0
# fraction = 0.1


# Define the output file path where results will be stored
output_file = f"/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/sketching_test/hausdorff_distance/normal_sketch_seed_{seed}_fraction_{fraction}.csv"

_, distance_df = evaluate_sampling_methods(adata, fraction=fraction, seed=seed, k=20)

distance_df.to_csv(output_file,index = False)
print("Saved")