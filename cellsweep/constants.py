"""cellmender constant values."""

CellBender_Fig2_to_Immune_All_High_celltype_mapping = {
    "Monocytes/neutrophils": [
        "Monocytes", "Mono-mac", "Monocyte precursor", "Macrophages", "Granulocytes"
    ],

    "Monocytes/pDCs": [
        "DC", "DC precursor", "pDC", "pDC precursor", "MNP"
    ],

    "T": [
        "T cells", "Double-negative thymocytes", "Double-positive thymocytes", "ETP"
    ],

    "B": [
        "B cells", "B-cell lineage", "Plasma cells"
    ],

    "NK": [
        "ILC", "ILC precursor"  # ILCs include NK-like subsets
    ],

    "Progenitor": [
        "HSC/MPP", "Early MK", "Megakaryocyte precursor"
    ],

    "Baso./neutro./progenitor": [
        "Promyelocytes", "Myelocytes"
    ],

}


# Broad-to-fine mapping
CellBender_Fig2_to_Immune_All_Low_celltype_mapping = {
    "Monocytes/neutrophils": [
        "Classical monocytes", "Non-classical monocytes", "Monocytes",
        "Intermediate macrophages", "Intestinal macrophages", "Macrophages",
        "Kupffer cells", "Kidney-resident macrophages", "Erythrophagocytic macrophages",
        "Neutrophils", "Granulocytes", "Mono-mac", "Monocyte precursor"
    ],

    "Monocytes/pDCs": [
        "pDC", "pDC precursor", "DC", "DC1", "DC2", "DC3",
        "Transitional DC", "Migratory DCs", "Cycling DCs", "DC precursor"
    ],

    "TrueT CD4+ naive/Treg": [
        "Tcm/Naive helper T cells", "Type 1 helper T cells", "Type 17 helper T cells",
        "Regulatory T cells", "Treg(diff)", "Follicular helper T cells"
    ],

    "B": [
        "B cells", "Cycling B cells", "Transitional B cells", "Age-associated B cells"
    ],

    "B naive": [
        "Naive B cells", "Pre-pro-B cells", "Pro-B cells", "Small pre-B cells", "Large pre-B cells"
    ],

    "B memory": [
        "Memory B cells", "Germinal center B cells", "Proliferative germinal center B cells"
    ],

    "T CD8+": [
        "CD8a/a", "CD8a/b(entry)"
    ],

    "T cytotoxic": [
        "Tem/Temra cytotoxic T cells", "Tem/Trm cytotoxic T cells",
        "Trm cytotoxic T cells", "Tcm/Naive cytotoxic T cells",
        "Memory CD4+ cytotoxic T cells"
    ],

    "T gd": [
        "gamma-delta T cells", "CRTAM+ gamma-delta T cells", "Cycling gamma-delta T cells"
    ],

    "MAIT": [
        "MAIT cells"
    ],

    "NK": [
        "NK cells", "CD16+ NK cells", "CD16- NK cells",
        "Cycling NK cells", "Transitional NK"
    ],

    "Monocyte NC/I": [
        "Non-classical monocytes", "Intermediate macrophages"
    ],

    "Progenitor": [
        "HSC/MPP", "CMP", "GMP", "MEMP", "ELP", "ETP",
        "Early lymphoid/T lymphoid", "Early MK", "Megakaryocyte precursor",
        "Megakaryocyte-erythroid-mast cell progenitor"
    ],

    "Baso./neutro./progenitor": [
        "Promyelocytes", "Myelocytes", "Neutrophil-myeloid progenitor"
    ],

    "pDCs": [
        "pDC", "pDC precursor"
    ]
}


CellTypistHigh_to_ImmuneMajor = {
    "Monocytes": "Monocytes",
    "Mono-mac": "Monocytes",
    "Monocyte precursor": "Macrophages",
    "Macrophages": "Macrophages",
    "Granulocytes": "Neutrophils",
    "DC": "DC",
    "DC precursor": "DC",
    "pDC": "DC",
    "pDC precursor": "DC",
    "MNP": "Neutrophils",
    "B cells": "B cells",
    "B-cell lineage": "B cells",
    "Plasma cells": "B cells",
    "T cells": "CD4 T cells",
    "Double-negative thymocytes": "CD4 T cells",
    "Double-positive thymocytes": "CD4 T cells",
    "ETP": "CD4 T cells",
}

CellTypistLow_to_ImmuneMajor = {

    # ---- Monocytes/neutrophils ----
    "Classical monocytes": "Monocytes",
    "Non-classical monocytes": "Monocytes",
    "Monocytes": "Monocytes",
    "Monocyte precursor": "Monocytes",
    "Mono-mac": "Monocytes",
    
    "Intermediate macrophages": "Macrophages",
    "Intestinal macrophages": "Macrophages",
    "Macrophages": "Macrophages",
    "Kupffer cells": "Macrophages",
    "Kidney-resident macrophages": "Macrophages",
    "Erythrophagocytic macrophages": "Macrophages",

    "Neutrophils": "Neutrophils",
    "Granulocytes": "Neutrophils",

    # ---- Monocytes/pDCs ----
    "pDC": "DC",
    "pDC precursor": "DC",
    "DC": "DC",
    "DC1": "DC",
    "DC2": "DC",
    "DC3": "DC",
    "Transitional DC": "DC",
    "Migratory DCs": "DC",
    "Cycling DCs": "DC",
    "DC precursor": "DC",

    # ---- CD4 T ----
    "Tcm/Naive helper T cells": "CD4 T cells",
    "Type 1 helper T cells": "CD4 T cells",
    "Type 17 helper T cells": "CD4 T cells",
    "Regulatory T cells": "CD4 T cells",
    "Treg(diff)": "CD4 T cells",
    "Follicular helper T cells": "CD4 T cells",

    # ---- B ----
    "B cells": "B cells",
    "Cycling B cells": "B cells",
    "Transitional B cells": "B cells",
    "Age-associated B cells": "B cells",
    "Naive B cells": "B cells",
    "Pre-pro-B cells": "B cells",
    "Pro-B cells": "B cells",
    "Small pre-B cells": "B cells",
    "Large pre-B cells": "B cells",
    "Memory B cells": "B cells",
    "Germinal center B cells": "B cells",
    "Proliferative germinal center B cells": "B cells",

    # ---- CD8 ----
    "CD8a/a": "CD8 T cells",
    "CD8a/b(entry)": "CD8 T cells",
    "Tem/Temra cytotoxic T cells": "CD8 T cells",
    "Tem/Trm cytotoxic T cells": "CD8 T cells",
    "Trm cytotoxic T cells": "CD8 T cells",
    "Tcm/Naive cytotoxic T cells": "CD8 T cells",
    "Memory CD4+ cytotoxic T cells": "CD8 T cells",
    "gamma-delta T cells": "CD8 T cells",
    "CRTAM+ gamma-delta T cells": "CD8 T cells",
    "Cycling gamma-delta T cells": "CD8 T cells",
    "MAIT cells": "CD8 T cells",

    # ---- NK ----
    "NK cells": "NK cells",
    "CD16+ NK cells": "NK cells",
    "CD16- NK cells": "NK cells",
    "Cycling NK cells": "NK cells",
    "Transitional NK": "NK cells",

    # ---- Progenitor ----
    "HSC/MPP": "Monocytes",   # default major category for HSC/MPP if forced into one bucket
    "CMP": "Monocytes",
    "GMP": "Neutrophils",
    "MEMP": "Eosinophils",
    "ELP": "B cells",
    "ETP": "CD4 T cells",
    "Early lymphoid/T lymphoid": "CD4 T cells",
    "Early MK": "Monocytes",
    "Megakaryocyte precursor": "Monocytes",
    "Megakaryocyte-erythroid-mast cell progenitor": "Eosinophils",

    # ---- Baso/neutro/progenitor ----
    "Promyelocytes": "Neutrophils",
    "Myelocytes": "Neutrophils",
    "Neutrophil-myeloid progenitor": "Neutrophils",
}

immune_markers = {
    "Monocytes": ["CD14", "LYZ", "FCGR3A", "MS4A7"],
    "Macrophages": ["CD68", "CD163", "C1QA", "C1QB", "C1QC"],
    "DC": ["CLEC9A", "XCR1", "CD1C", "FCER1A", "IL3RA", "TCF4", "ITGAX", "CST3"],
    "Neutrophils": ["S100A8", "S100A9", "MPO", "FCGR3B", "ELANE"],
    "Eosinophils": ["CLC", "RNASE2", "RNASE3", "PRG2"],
    "CD8 T cells": ["CD8A", "CD8B", "GZMB", "CD3E"],
    "CD4 T cells": ["CD4", "CCR7", "IL7R", "TCF7", "CD3E"],
    "NK cells": ["NKG7", "GNLY", "PRF1", "KLRD1", "GZMB"],
    "B cells": ["MS4A1", "CD79A", "CD79B", "HLA-DRA", "CD19"]
}
