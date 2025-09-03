import argparse
import torch
import os

from tqdm import tqdm
import pickle

from scipy.stats import mannwhitneyu

parser = argparse.ArgumentParser(description="TEST VERSION: A script that takes as input complementary matrices that contain protein-level summarized data and protein-level properties/metadata, and outputs the association between the activations and the properties. Optionally, may restrict to only a few latents.")

parser.add_argument("summarizedActs", default="data/",
                    help="The input .pt file with summarized activations")
parser.add_argument("propertyInfo", default="metadata/",
                    help="The auxiliary .pt file with property information")

parser.add_argument("--subsetListFile", default="metadata/list_of_desired_latents.pkl")
parser.add_argument("--outfile", default="test_latents_property",
                    help="File stub for outfile name")
parser.add_argument("--test_latents", type=int, default=10,
                    help="Number of latents to process in test mode")
parser.add_argument("--test_properties", type=int, default=5,
                    help="Number of properties to process in test mode")

args = parser.parse_args()

if __name__ == "__main__":
    print("="*50)
    print("RUNNING IN TEST MODE")
    print("="*50)
    
    all_acts = torch.load(args.summarizedActs)
    all_properties = torch.load(args.propertyInfo)

    if args.subsetListFile is not None:
        with open(args.subsetListFile, 'rb') as f:
            subset_list = pickle.load(f)

        all_acts = all_acts[:,subset_list]
    
    # TEST: Only process first N latents and M properties for quick testing
    test_latents = min(args.test_latents, all_acts.shape[1])
    test_properties = min(args.test_properties, all_properties.shape[1])
    
    print(f"Full data shape: acts={all_acts.shape}, properties={all_properties.shape}")
    print(f"Processing {test_latents} latents and {test_properties} properties")
    print("-"*50)
    
    latents_property_pvals = torch.zeros(test_latents, test_properties)
    latents_property_effect_size = torch.zeros(size=
        (test_latents, test_properties, 4))

    # TODO This works for binary properties but doesn't for more categories or for continuous variables
    for i in tqdm(range(test_latents), desc="Processing latents"):
        for j in range(test_properties):
            tmp_x = all_acts[all_properties[:,j] != 0,i]
            tmp_y = all_acts[all_properties[:,j] == 0,i]
            latents_property_pvals[i,j] = mannwhitneyu(tmp_x, tmp_y).pvalue
            latents_property_effect_size[i,j,0] = tmp_x.mean()
            latents_property_effect_size[i,j,1] = tmp_y.mean()
            latents_property_effect_size[i,j,2] = all_acts[:,0].std()
            latents_property_effect_size[i,j,3] = (tmp_x.mean() - tmp_y.mean())/(all_acts[:,0].std())

    print("\nTest completed successfully!")
    print(f"Output shape: pvals={latents_property_pvals.shape}, effect_size={latents_property_effect_size.shape}")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    torch.save(latents_property_pvals, f"data/{args.outfile}_pvals.pt")
    torch.save(latents_property_effect_size, f"data/{args.outfile}_effect_size.pt")
    
    print(f"\nResults saved to:")
    print(f"  - data/{args.outfile}_pvals.pt")
    print(f"  - data/{args.outfile}_effect_size.pt")