# Whose Streets Are Greener?

Data and analysis code for the manuscript:

**Whose Streets Are Greener? Assessing the Effects of Housing Provision and Property Management on Green View Index in Urban Neighborhoods**

## Repository structure

```text
Whose_Streets_Are_Greener/
├── README.md
├── code/
│   ├── Whose_Streets_Are_Greener_Main_Analysis.R
│   └── street_view_segmentation/
│       ├── Street_View_Image_Segmentation_ADE20K.py
│       ├── color_coding_semantic_segmentation_classes - Sheet1.pdf
│       └── data/
│           ├── color150.mat
│           └── object150_info.csv
└── data/
    ├── Whose_Streets_Are_Greener_Dataset.xlsx
    └── street_view_segmentation_outputs.zip
```

## Data

`data/Whose_Streets_Are_Greener_Dataset.xlsx` is the analysis-ready community-level dataset used in the main statistical analysis.

`data/street_view_segmentation_outputs.zip` contains the processed street-view semantic-segmentation outputs for the four study districts:

- `Haicang_images_result_150.csv`
- `Huli_images_result_150.csv`
- `Jimei_images_result_150.csv`
- `Siming_images_result_150.csv`

## Main statistical analysis

The main analysis is implemented in:

`code/Whose_Streets_Are_Greener_Main_Analysis.R`

The script reproduces the community-level OLS model with district fixed effects and heteroskedasticity-robust standard errors reported in the manuscript. It also exports model coefficients, model-fit statistics, multicollinearity diagnostics, and R session information to a `results/` directory.

Required R packages:

- `readxl`
- `fixest`
- `car`

To run the analysis, download or clone the repository and execute the R script from the repository root.

## Street-view semantic segmentation

The street-view segmentation procedure is implemented in:

`code/street_view_segmentation/Street_View_Image_Segmentation_ADE20K.py`

The script uses an ADE20K semantic-segmentation model based on the CSAILVision semantic-segmentation implementation. Supporting class and color files used by the script are included in `code/street_view_segmentation/`.

Python dependencies include:

- `numpy`
- `Pillow`
- `scipy`
- `torch`
- `torchvision`
- `mit_semseg`

The pretrained ADE20K model checkpoints and the original street-view images are not included in this repository. A complete rerun of the image-segmentation stage therefore requires the corresponding model checkpoints and source images. The processed segmentation outputs used in the study are provided in `data/street_view_segmentation_outputs.zip`.

## Data availability

The processed data and analysis code supporting the findings of the study are available in this repository.

## Citation

Citation information will be added after publication.
