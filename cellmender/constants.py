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
