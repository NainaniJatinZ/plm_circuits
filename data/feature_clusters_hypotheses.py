"""
This contains the classes of features that we hypothesized from manual analysis of case studies. 
"""

feature_clusters_MetXA = {
    # Layer 4 clusters
    4: {
        "direct_motif_detectors": {340: "FX'", 237: "FX'", 3788: "X'XI", 798: "D'XXGN", 1690: "X'XXF"},  # Example: specific (token, latent) pairs
        "indirect_motif_detectors": {2277: "G", 2311: "X'XM", 3634: "X'XXXG", 1682: "PXXXXXX'", 3326: "H"},
        "motif_detectors": {340: "FX'", 237: "FX'", 3788: "X'XI", 798: "D'XXGN", 1690: "X'XXF", 2277: "G", 2311: "X'XM", 3634: "X'XXXG", 1682: "PXXXXXX'", 3326: "H"}, 
    },
    
    # # Layer 8 clusters  
    8: {
        "annotated_domain_detector": {488:"AB_Hydrolase_fold"}, #, 2693: "AB_Hydrolase_fold", 1244:"AB_Hydrolase_fold"},
        "misc_domain_detector": {2677:"FAD/NAD", 2775:"Transketolase", 2166:"DHFR"},
        "motif_detectors": {488:"AB_Hydrolase_fold", 2677:"FAD/NAD", 2775:"Transketolase", 2166:"DHFR"}, #, 2693: "AB_Hydrolase_fold", 1244:"AB_Hydrolase_fold"},
    },
    
    # # Layer 12 clusters
    12: {
        "annotated_domain_detector": {2112: "AB_Hydrolase_fold"},
        "misc_domain_detector": {3536:"SAM_mtases", 1256: "FAM", 2797: "Aldolase", 3794: "SAM_mtases", 3035: "WD40"},
        "motif_detectors": {2112: "AB_Hydrolase_fold", 3536:"SAM_mtases", 1256: "FAM, Acyl-CoA N-acyltransferase", 2797: "Aldolase", 3794: "SAM_mtases", 3035: "WD40"},
    },
    # Add more layers and clusters as needed...
}

feature_clusters_Top2 = {
    4: {
        "direct_motif_detectors": {1509:"E", 2511:"X'XQ", 2112:"YXX'", 3069: "GX'", 3544: "C", 2929: "N"},
        "indirect_motif_detectors": {3170: "X'N", 3717:"V", 527: "DX'", 3229: "IXX'", 1297: "I", 1468: "X'XXN", 1196: "D"},
        "motif_detectors": {1509:"E", 2511:"X'XQ", 2112:"YXX'", 3069: "GX'", 3544: "C", 2929: "N", 3170: "X'N", 3717:"V", 527: "DX'", 3229: "IXX'", 1297: "I", 1468: "X'XXN", 1196: "D"}
    },
    8: {
        # "direct_motif_detectors": {},
        "indirect_motif_detectors": {1916: "NX'XXNA"},
        # "motif_detectors": {}, 
        "annotated_domain_detector": {2529:"Hatpase_C", 3159: "Hatpase_C", 3903: "Hatpase_C", 1055: "Hatpase_C", 2066: "Hatpase_C"},
    },
    12: {
        "annotated_domain_detector": {3943: "Hatpase_C", 1796: "Hatpase_C", 1204: "Hatpase_C", 1145:  "Hatpase_C"},
        "misc_domain_detector": {1082: "XPG-I", 2472: "Kinesin"},
        "domain_detectors": {3943: "Hatpase_C", 1796: "Hatpase_C", 1204: "Hatpase_C", 1145:  "Hatpase_C", 1082: "XPG-I", 2472: "Kinesin"},
    },
    16: {
        "annotated_domain_detector": {3077: "Hatpase_C", 1353: "Hatpase_C", 1597: "Hatpase_C", 1814: "Hatpase_C", 3994: "Ribosomal", 1166: "Hatpase_C"},
        # "misc_domain_detector": {},
        # "domain_detectors": {3077: "Hatpase_C", 1353: "Hatpase_C", 1597: "Hatpase_C", 1814: "Hatpase_C", 3994: "Ribosomal", 1166: "Hatpase_C"},
    }
}