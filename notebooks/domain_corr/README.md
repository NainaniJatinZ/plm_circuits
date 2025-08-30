# Automated latent labelling

Input:

- For each latent, a set of proteins and the activation on those proteins (default: 10k random uniprot proteins)
- A set of properties defined on those proteins (for now, binary only)

Processing:

- We summarize the latent activation per protein either with max or mean
- We do a MWU test between the activations on proteins with property A and without property A
- We also report effect sizes

Output:

- A measure of assocation between the latents and the protein properties