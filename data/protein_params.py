"""
protein_params
~~~~~~~~~~~~~~
Centralized protein-specific parameters used by notebooks.

This keeps small data dictionaries in one place for reuse across
notebooks and scripts.
"""

# Secondary-structure segment positions (inclusive start, exclusive end)
sse_dict = {
    "2B61A": [[182, 316]],
    "1PVGA": [[101, 202]],
}

# Flank lengths: [clean_flank_len, corrupted_flank_len]
fl_dict = {
    "2B61A": [44, 43],
    "1PVGA": [65, 63],
}

# Pretty names for figure titles
protein_name = {
    "2B61A": ["METXA_HAEIN", "MetXA"],
    "1PVGA": ["TOP2_YEAST", "Top2"],
}

# protein name to pdb id
protein2pdb = {
    "MetXA":"2B61A",
    "Top2":"1PVGA"
}

