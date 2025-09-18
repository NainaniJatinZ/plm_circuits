import argparse
import torch

from tqdm import tqdm
import pickle

from scipy.stats import mannwhitneyu

parser = argparse.ArgumentParser(description="A script that takes as input complementary matrices that contain protein-level summarized data and protein-level properties/metadata, and outputs the association between the activations and the properties. Optionally, may restrict to only a few latents.")

parser.add_argument("summarizedActs", default="data/",
                    help="The input .pt file with summarized activations")
parser.add_argument("propertyInfo", default="metadata/",
                    help="The auxiliary .pt file with property information")

parser.add_argument("--subsetListFile", default="metadata/list_of_desired_latents.pkl")
parser.add_argument("--outfile", default="latents_property",
                    help="File stub for outfile name")
parser.add_argument("--alternative", choices=["two-sided", "greater", "less"], default="greater",
                    help="Alternative hypothesis for Mann-Whitney U test")

args = parser.parse_args()

if __name__ == "__main__":
    all_acts = torch.load(args.summarizedActs)
    all_properties = torch.load(args.propertyInfo)

    if args.subsetListFile is not None:
        with open(args.subsetListFile, 'rb') as f:
            subset_list = list(set(pickle.load(f)))
            subset_list.sort()
        all_acts = all_acts[:,subset_list]
    
    # TODO ensure that the matrix sizes match up
    # TODO documentation on matrix sizes
    latents_property_pvals = torch.full((all_acts.shape[1], all_properties.shape[1]), float("nan"))
    latents_property_statistic = torch.full((all_acts.shape[1], all_properties.shape[1]), float("nan"))
    latents_property_effect_size = torch.zeros(size=
        (all_acts.shape[1], all_properties.shape[1],4))
    latents_property_auc = torch.full((all_acts.shape[1], all_properties.shape[1]), float("nan"))
    latents_property_n_pos = torch.zeros(all_acts.shape[1], all_properties.shape[1])
    latents_property_n_neg = torch.zeros(all_acts.shape[1], all_properties.shape[1])

    # TODO This works for binary properties but doesn't for more categories or for continuous variables
    for i in tqdm(range(latents_property_pvals.shape[0])):
        for j in range(latents_property_pvals.shape[1]):
            tmp_x = all_acts[all_properties[:,j] != 0,i]
            tmp_y = all_acts[all_properties[:,j] == 0,i]
            latents_property_n_pos[i,j] = tmp_x.numel()
            latents_property_n_neg[i,j] = tmp_y.numel()
            if tmp_x.numel() == 0 or tmp_y.numel() == 0:
                # Leave NaNs for pval/stat/auc, still record means/std/effect size as NaN where needed
                latents_property_effect_size[i,j,0] = torch.nan if tmp_x.numel() == 0 else tmp_x.mean()
                latents_property_effect_size[i,j,1] = torch.nan if tmp_y.numel() == 0 else tmp_y.mean()
                std_all = all_acts[:,0].std()
                latents_property_effect_size[i,j,2] = std_all
                latents_property_effect_size[i,j,3] = torch.nan
                continue
            mw_res = mannwhitneyu(tmp_x, tmp_y, alternative=args.alternative)
            latents_property_pvals[i,j] = mw_res.pvalue
            latents_property_statistic[i,j] = mw_res.statistic
            # AUC from U statistic: U/(n_pos*n_neg). For two-sided, U corresponds to the given order
            n_pos = tmp_x.numel()
            n_neg = tmp_y.numel()
            latents_property_auc[i,j] = mw_res.statistic / (n_pos * n_neg)
            latents_property_effect_size[i,j,0] = tmp_x.mean()
            latents_property_effect_size[i,j,1] = tmp_y.mean()
            latents_property_effect_size[i,j,2] = all_acts[:,0].std()
            latents_property_effect_size[i,j,3] = (tmp_x.mean() - tmp_y.mean())/(all_acts[:,0].std())

    # TODO automatically adjust p-values using either BH or Bonferroni?
    torch.save(latents_property_pvals, f"data/{args.outfile}_pvals.pt")
    torch.save(latents_property_effect_size, f"data/{args.outfile}_effect_size.pt")
    torch.save(latents_property_statistic, f"data/{args.outfile}_statistic.pt")
    torch.save(latents_property_auc, f"data/{args.outfile}_auc.pt")
    torch.save(latents_property_n_pos, f"data/{args.outfile}_n_pos.pt")
    torch.save(latents_property_n_neg, f"data/{args.outfile}_n_neg.pt")