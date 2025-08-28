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



gene_scores_df = pd.read_csv('/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/simulations_for_sketching_analysis/visium_like_data/complex/data/leverage_scores_smoothed_pca.csv')
adata.obs['gene_score'] = gene_scores_df['leverage_score'].values





base_directory = '/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/simulations_for_sketching_analysis/visium_like_data/complex/sketching_test'
ground_truth = 'group'


fraction_df = pd.DataFrame()
for knn_for_neighborhood_analysis in [10]:#[2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:

	partial_fraction_df = neighborhood_sweep(adata, 
		fraction = fraction, 
		base_directory = base_directory, 
		ground_truth = ground_truth,
		knn_for_neighborhood_analysis = knn_for_neighborhood_analysis)

	fraction_df = pd.concat([fraction_df,partial_fraction_df],ignore_index = True)

	fraction_df.to_csv(os.path.join(base_directory, f"knn_neighborhood_analysis/rasp_methods_{fraction}.csv"))


