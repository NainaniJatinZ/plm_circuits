# %%

# random_sampler.py
from Bio import SeqIO
import random, pathlib

FASTA = pathlib.Path("../../plm_interp/uniprot_sprot.fasta")   # change if you renamed



def sample_random(fasta=FASTA, n=100, max_len=1500):
    """Return a list of (id, seq) tuples."""
    records = [
        (rec.id, str(rec.seq))
        for rec in SeqIO.parse(fasta, "fasta")
        if len(rec.seq) <= max_len
    ]
    random.shuffle(records)
    return records[:n]

# quick test
if __name__ == "__main__":
    for pid, seq in sample_random(n=5):
        print(pid, len(seq))
# %%

FASTA = pathlib.Path("/project/pi_jensen_umass_edu/jnainani_umass_edu/plm_data/uniref50_sp.fasta")
records = [
        (rec.id, str(rec.seq))
        for rec in SeqIO.parse(FASTA, "fasta")
        if len(rec.seq) <= 5000]
print(len(records))

# %%
