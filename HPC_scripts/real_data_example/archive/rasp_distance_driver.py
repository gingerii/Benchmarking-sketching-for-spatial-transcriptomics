print("starting script")
from functions import *
import argparse
print("imported functions")



parser = argparse.ArgumentParser(description="Run leverage driver with specified parameters.")
parser.add_argument('--knn', type=int, required=True, help='Number of nearest neighbors')
parser.add_argument('--beta', type=int, required=True, help='Beta parameter')
parser.add_argument('--fraction', type=float, required=True, help='Fraction parameter')
parser.add_argument('--seed', type=float, required=True, help='Seed parameter')
args = parser.parse_args()

knn = args.knn
beta = args.beta
fraction = args.fraction

seed = int(args.seed)

print(f"Running with: knn={knn}, beta={beta}, fraction={fraction}, seed={seed}")

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
output_file = f"/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/sketching_test/hausdorff_distance/RASP_sketch_beta_{beta}_knn_{knn}_seed_{seed}_{fraction}.csv"

# Initialize an empty DataFrame to gather all results


    # Call the compute_distances function with varying seeds
distance_df = compute_distances(adata, knn=knn, beta=beta, platform='merfish', seed=seed, num_pcs=20, fraction = fraction)

    #

    # Save the concatenated results to a CSV file
distance_df.to_csv(output_file, index=False)
    

# Indicate that the loop has completed
print("Loop completed. All results saved.")