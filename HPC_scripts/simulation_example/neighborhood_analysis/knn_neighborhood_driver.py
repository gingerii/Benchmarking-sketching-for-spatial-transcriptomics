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

gene_scores_df = pd.read_csv('/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/simulations_for_sketching_analysis/visium_like_data/complex/data/leverage_scores.csv')
adata.obs['gene_score'] = gene_scores_df['leverage_score'].values

#parent_dir = '/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/allen_institute_data/sketching_test/index'
gene_scores_df = pd.read_csv('/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/simulations_for_sketching_analysis/visium_like_data/complex/data/leverage_scores.csv')
adata.obs['gene_score'] = gene_scores_df['leverage_score'].values

ground_truth = 'group'
base_directory = "/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/simulations_for_sketching_analysis/visium_like_data/complex/sketching_test"
#knn_for_neighborhood_analysis = 10
#fraction_df = pd.DataFrame()

#for fraction in [0.01, 0.02, 0.03, 0.04,0.05,0.06,0.07,0.08,0.09,0.10, 0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
fraction_df = pd.DataFrame()
for knn_for_neighborhood_analysis in [2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:
    print(f"Processing fraction: {fraction}, knn: {knn_for_neighborhood_analysis}")

    partial_fraction_df = neighborhood_sweep(adata, 
        fraction = fraction, 
        base_directory = base_directory, 
        ground_truth = ground_truth,
        knn_for_neighborhood_analysis = knn_for_neighborhood_analysis)

    fraction_df = pd.concat([fraction_df,partial_fraction_df],ignore_index = True)

    fraction_df.to_csv(os.path.join(base_directory, f"knn_neighborhood_analysis/normal_methods_{fraction}.csv"))
    print(f"saved iteration for fraction: {fraction}, knn: {knn_for_neighborhood_analysis}")





#


