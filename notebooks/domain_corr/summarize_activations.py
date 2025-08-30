import argparse
import torch

from tqdm import tqdm

from utils import load_cached_activations

parser = argparse.ArgumentParser(description="A script that summarizes activations on the level of the whole protein.")
parser.add_argument("actsDir", default="../../jatin/plm_circuits/acts",
                    help="The input directory where the files are stored")
parser.add_argument("latentsPerLayer", type=int, default=4096,
                    help="Number of latents per layer")
parser.add_argument("--summarize", default="length_normalized",
                    help="Summarization procedure (Options: 'length_normalized', 'max')")

args = parser.parse_args()

if __name__ == "__main__":
    tmp = load_cached_activations(args.actsDir)
    summarized_activations = torch.zeros(len(tmp.metadata['proteins']), len(tmp.layers)*args.latentsPerLayer)

    for protein_ix in tqdm(range(summarized_activations.shape[0])):
        cur_ptn = tmp.load_protein_activations_original_format(protein_ix)
        if args.summarize == "length_normalized":
            tmp_store = torch.cat( 
                tuple([torch.sum(cur_ptn['activations'][l], axis=0) for l in tmp.layers]))/cur_ptn['sequence_length']
        elif args.summarize == "max":
            tmp_store = torch.cat( 
                tuple([torch.max(cur_ptn['activations'][l], axis=0).values for l in tmp.layers]))
        summarized_activations[protein_ix,:] = tmp_store

    torch.save(summarized_activations, f"data/summarized_acts_{args.summarize}.pt")
