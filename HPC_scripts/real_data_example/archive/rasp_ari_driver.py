print("starting script")
from functions import *
import argparse
print("imported functions")



parser = argparse.ArgumentParser(description="Run leverage driver with specified parameters.")
parser.add_argument('--knn', type=int, required=True, help='Number of nearest neighbors')
parser.add_argument('--beta', type=int, required=True, help='Beta parameter')
parser.add_argument('--fraction', type=float, required=True, help='Fraction parameter')
args = parser.parse_args()

knn = args.knn
beta = args.beta
fraction = args.fraction

print(f"Running with: knn={knn}, beta={beta}, fraction={fraction}")



adata = sc.read_h5ad("/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/simulation/processed_data/stripes.h5ad")
#add just to improve understanding 
adata.obsm['spatial']=adata.obsm['spatial']*100




output_file = f"/dartfs-hpc/rc/lab/F/FrostH/members/igingerich/public_data/simulation/sketching_tests/stripes/ari/RASP__beta_{beta}_knn_{knn}_{fraction}.csv"

df=RASP_ari(adata, fraction = fraction,
             seed = 0, 
             n_neighbors = 10,
             n_clusters = 8,
             ground_truth = 'ground_truth',
             cluster_algorithm = 'louvain',
             knn=knn, beta=beta, platform='SRTsim', num_pcs=20,k=20)

df.to_csv(output_file, index=False)

