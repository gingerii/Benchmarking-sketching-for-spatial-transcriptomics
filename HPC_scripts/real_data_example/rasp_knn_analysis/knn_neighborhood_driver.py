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

directory = "/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/processed_data"
adata = sc.read_h5ad(os.path.join(directory,"xenium_cancer_processed.h5ad"))



gene_scores_df = pd.read_csv('/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/processed_data/leverage_scores_smoothed_pca.csv')
adata.obs['gene_score'] = gene_scores_df['leverage_score'].values

#parent_dir = '/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/mouse_MERFISH_data/sketching_test/index'





base_directory = '/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/xenium_cancer/sketching_test'
ground_truth = 'region'


fraction_df = pd.DataFrame()
for knn_for_neighborhood_analysis in [2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:

	partial_fraction_df = neighborhood_sweep(adata, 
		fraction = fraction, 
		base_directory = base_directory, 
		ground_truth = ground_truth,
		knn_for_neighborhood_analysis = knn_for_neighborhood_analysis)

	fraction_df = pd.concat([fraction_df,partial_fraction_df],ignore_index = True)

	fraction_df.to_csv(os.path.join(base_directory, f"knn_neighborhood_analysis/rasp_methods_{fraction}.csv"))


