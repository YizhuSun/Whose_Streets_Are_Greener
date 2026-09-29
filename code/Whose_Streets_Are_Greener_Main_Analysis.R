# ==============================================================================
# Whose Streets Are Greener?
# Reproducible R script for the main community-level multivariable regression
#
# Purpose:
#   Reproduce the main OLS model with district fixed effects reported in the
#   manuscript (Table 2 and the coefficient estimates underlying Figure 5).
#
# Input:
#   data/Whose_Streets_Are_Greener_Dataset.xlsx
#   or, if the data file is stored in the repository root:
#   Whose_Streets_Are_Greener_Dataset.xlsx
#
# Output:
#   results/model_coefficients.csv
#   results/model_fit_statistics.csv
#   results/multicollinearity_diagnostics.csv
#   results/sessionInfo.txt
#
# Notes:
#   1. GVI is expressed in percentage points in the source dataset.
#   2. Housing-provision and property-management shares are proportions [0, 1].
#   3. Collective-land housing is the omitted housing-provision category.
#   4. Grassroots-coordinated management (owner management + community management)
#      is the omitted property-management category.
#   5. District fixed effects are included in the model.
#   6. Heteroskedasticity-robust standard errors are used.
#   7. No random procedure is used; therefore, no random seed is required.
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Required packages
# ------------------------------------------------------------------------------

required_packages <- c("readxl", "fixest", "car")

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))
]

if (length(missing_packages) > 0) {
  stop(
    "Missing required R package(s): ",
    paste(missing_packages, collapse = ", "),
    ". Install them before running this script."
  )
}

library(readxl)
library(fixest)
library(car)


# ------------------------------------------------------------------------------
# 2. Locate and read the analysis-ready dataset
# ------------------------------------------------------------------------------

data_candidates <- c(
  file.path("data", "Whose_Streets_Are_Greener_Dataset.xlsx"),
  "Whose_Streets_Are_Greener_Dataset.xlsx"
)

available_data <- data_candidates[file.exists(data_candidates)]

if (length(available_data) == 0) {
  stop(
    "Data file not found. Place 'Whose_Streets_Are_Greener_Dataset.xlsx' ",
    "either in the repository root or in the 'data' folder."
  )
}

data_path <- available_data[1]

data_raw <- read_excel(
  path = data_path,
  sheet = "厦门市中心城区实验数据"
)

cat("Data file:", data_path, "\n")
cat("Rows:", nrow(data_raw), " Columns:", ncol(data_raw), "\n")


# ------------------------------------------------------------------------------
# 3. Verify required source variables
# ------------------------------------------------------------------------------

required_columns <- c(
  "district",
  "community_id",
  "community_name",
  "number_compounds",
  "n_households",
  "V1_GVI",
  "V2_share_commodity_housing",
  "V2_share_work-unit _housing",
  "V2_share_collective-land_housing",
  "V2_share_affordable_housing",
  "V3_share_market_man",
  "V3_share_owner_man",
  "V3_share_unit_man",
  "V3_share_community_man",
  "V4_housing_price",
  "V5_building_age",
  "V6_road density",
  "V6_ln_road density"
)

missing_columns <- setdiff(required_columns, names(data_raw))

if (length(missing_columns) > 0) {
  stop(
    "Required source variable(s) missing from the dataset: ",
    paste(missing_columns, collapse = ", ")
  )
}

if (anyDuplicated(data_raw$community_id) > 0) {
  stop("Duplicate community_id values were detected.")
}

if (nrow(data_raw) != 155) {
  warning(
    "The dataset contains ", nrow(data_raw),
    " communities; the manuscript main analysis reports N = 155."
  )
}


# ------------------------------------------------------------------------------
# 4. Construct analysis variables
# ------------------------------------------------------------------------------

data <- data.frame(
  community_id = as.character(data_raw$community_id),
  community_name = as.character(data_raw$community_name),
  district = factor(data_raw$district),

  # Dependent variable
  GVI = as.numeric(data_raw$V1_GVI),

  # Housing provision
  H_commodity = as.numeric(data_raw$V2_share_commodity_housing),
  H_work_unit = as.numeric(data_raw[["V2_share_work-unit _housing"]]),
  H_collective_land = as.numeric(data_raw[["V2_share_collective-land_housing"]]),
  H_affordable = as.numeric(data_raw$V2_share_affordable_housing),

  # Property management
  M_market = as.numeric(data_raw$V3_share_market_man),
  M_owner = as.numeric(data_raw$V3_share_owner_man),
  M_work_unit = as.numeric(data_raw$V3_share_unit_man),
  M_community = as.numeric(data_raw$V3_share_community_man),

  # Controls
  housing_price = as.numeric(data_raw$V4_housing_price),
  building_age = as.numeric(data_raw$V5_building_age),
  households = as.numeric(data_raw$n_households),
  compounds = as.numeric(data_raw$number_compounds),

  # Road-network exposure proxy
  road_exposure_proxy = as.numeric(data_raw[["V6_road density"]]),
  ln_road_exposure_proxy_stored = as.numeric(data_raw[["V6_ln_road density"]]),

  stringsAsFactors = FALSE
)

# Log-transformed controls used in the model.
# Price and household count are transformed here from their unlogged source values.
data$ln_price <- log(data$housing_price)
data$ln_households <- log(data$households)

# Recalculate the logged road variable from the publicly available road proxy.
# The stored logged column is retained only for an explicit consistency check.
data$ln_road_exposure_proxy <- log1p(data$road_exposure_proxy)

# Grassroots-coordinated management is the reference category.
data$M_grassroots <- data$M_owner + data$M_community


# ------------------------------------------------------------------------------
# 5. Data-integrity and model-input checks
# ------------------------------------------------------------------------------

tol <- 1e-8

numeric_check_vars <- c(
  "GVI",
  "H_commodity",
  "H_work_unit",
  "H_collective_land",
  "H_affordable",
  "M_market",
  "M_owner",
  "M_work_unit",
  "M_community",
  "housing_price",
  "building_age",
  "households",
  "compounds",
  "road_exposure_proxy",
  "ln_road_exposure_proxy_stored",
  "ln_price",
  "ln_households",
  "ln_road_exposure_proxy"
)

if (any(!is.finite(as.matrix(data[numeric_check_vars])))) {
  stop("Missing or non-finite values were detected in variables required for analysis.")
}

if (any(data$housing_price <= 0)) {
  stop("Housing price must be strictly positive before log transformation.")
}

if (any(data$households <= 0)) {
  stop("Household count must be strictly positive before log transformation.")
}

if (any(data$road_exposure_proxy < 0)) {
  stop("Road-network exposure proxy cannot be negative.")
}

share_vars <- c(
  "H_commodity", "H_work_unit", "H_collective_land", "H_affordable",
  "M_market", "M_owner", "M_work_unit", "M_community"
)

if (any(as.matrix(data[share_vars]) < -tol | as.matrix(data[share_vars]) > 1 + tol)) {
  stop("At least one housing-provision or property-management share lies outside [0, 1].")
}

housing_share_sum <- with(
  data,
  H_commodity + H_work_unit + H_collective_land + H_affordable
)

management_share_sum <- with(
  data,
  M_market + M_owner + M_work_unit + M_community
)

if (max(abs(housing_share_sum - 1)) > 1e-6) {
  stop("Housing-provision shares do not sum to 1 within numerical tolerance.")
}

if (max(abs(management_share_sum - 1)) > 1e-6) {
  stop("Property-management shares do not sum to 1 within numerical tolerance.")
}

road_log_difference <- max(
  abs(data$ln_road_exposure_proxy - data$ln_road_exposure_proxy_stored)
)

if (road_log_difference > 1e-8) {
  stop(
    "The stored logged road variable is inconsistent with log1p(V6_road density). ",
    "Maximum absolute difference = ", signif(road_log_difference, 6)
  )
}

model_vars <- c(
  "GVI",
  "H_commodity",
  "H_work_unit",
  "H_affordable",
  "M_market",
  "M_work_unit",
  "ln_price",
  "building_age",
  "ln_households",
  "compounds",
  "ln_road_exposure_proxy",
  "district"
)

if (!all(complete.cases(data[model_vars]))) {
  stop("Incomplete cases were detected among variables used in the main model.")
}

cat("Data-integrity checks passed.\n")
cat("Communities used in the main model:", nrow(data), "\n")
cat(
  "Maximum housing-share sum error:",
  format(max(abs(housing_share_sum - 1)), scientific = TRUE),
  "\n"
)
cat(
  "Maximum management-share sum error:",
  format(max(abs(management_share_sum - 1)), scientific = TRUE),
  "\n"
)
cat(
  "Maximum stored/recalculated road-log difference:",
  format(road_log_difference, scientific = TRUE),
  "\n"
)


# ------------------------------------------------------------------------------
# 6. Main OLS model with district fixed effects
# ------------------------------------------------------------------------------

# Reference categories:
#   Housing provision: collective-land housing
#   Property management: grassroots-coordinated management
#                        (owner management + community management)

m_main <- feols(
  GVI ~
    H_commodity +
    H_work_unit +
    H_affordable +
    M_market +
    M_work_unit +
    ln_price +
    building_age +
    ln_households +
    compounds +
    ln_road_exposure_proxy |
    district,
  data = data,
  vcov = "hetero"
)

print(summary(m_main))


# ------------------------------------------------------------------------------
# 7. Model-fit statistics
# ------------------------------------------------------------------------------

fit_statistics <- fitstat(
  m_main,
  c("n", "r2", "ar2", "wr2", "aic", "bic"),
  simplify = TRUE
)

print(fit_statistics)

# Internal consistency checks against the frozen analysis dataset.
# These values reproduce the main-model fit reported in the manuscript.
expected_fit <- c(
  n = 155,
  r2 = 0.21869175355694004,
  ar2 = 0.14665624147353729,
  wr2 = 0.2118413725720354,
  aic = 965.9375294658769,
  bic = 1008.5454811027463
)

fit_names_common <- intersect(names(expected_fit), names(fit_statistics))

fit_difference <- abs(
  as.numeric(fit_statistics[fit_names_common]) -
    as.numeric(expected_fit[fit_names_common])
)

if (any(fit_difference > 1e-6)) {
  warning(
    "At least one model-fit statistic differs from the frozen reference result. ",
    "Check the input data, package versions, or model specification."
  )
}


# ------------------------------------------------------------------------------
# 8. Coefficient estimates and 95% confidence intervals
# ------------------------------------------------------------------------------

coef_matrix <- coeftable(m_main)
ci_matrix <- confint(m_main, level = 0.95)

coefficient_results <- data.frame(
  term = rownames(coef_matrix),
  estimate = coef_matrix[, "Estimate"],
  robust_se = coef_matrix[, "Std. Error"],
  statistic = coef_matrix[, "t value"],
  p_value = coef_matrix[, "Pr(>|t|)"],
  conf_low_95 = ci_matrix[, 1],
  conf_high_95 = ci_matrix[, 2],
  row.names = NULL,
  check.names = FALSE
)

print(coefficient_results)


# ------------------------------------------------------------------------------
# 9. Multicollinearity diagnostics
# ------------------------------------------------------------------------------

# car::vif() is evaluated on the algebraically equivalent OLS specification
# with district entered as a factor. Because district has more than one degree
# of freedom, car::vif() returns generalized VIF (GVIF) diagnostics.
#
# For terms with one degree of freedom:
#   GVIF^(1/(2*Df)) = sqrt(VIF)
#
# Therefore, the 1.09-2.75 range previously reported in the manuscript
# corresponds to the adjusted GVIF column, not to the conventional raw VIF.
# The output below preserves both quantities so they can be labelled correctly.

vif_model <- lm(
  GVI ~
    H_commodity +
    H_work_unit +
    H_affordable +
    M_market +
    M_work_unit +
    ln_price +
    building_age +
    ln_households +
    compounds +
    ln_road_exposure_proxy +
    factor(district),
  data = data
)

vif_raw <- car::vif(vif_model)

if (is.matrix(vif_raw)) {
  vif_results <- data.frame(
    term = rownames(vif_raw),
    GVIF = vif_raw[, "GVIF"],
    Df = vif_raw[, "Df"],
    adjusted_GVIF = vif_raw[, "GVIF^(1/(2*Df))"],
    row.names = NULL,
    check.names = FALSE
  )
} else {
  vif_results <- data.frame(
    term = names(vif_raw),
    VIF = as.numeric(vif_raw),
    sqrt_VIF = sqrt(as.numeric(vif_raw)),
    row.names = NULL,
    check.names = FALSE
  )
}

print(vif_results)

if ("adjusted_GVIF" %in% names(vif_results)) {
  cat(
    "Adjusted GVIF range:",
    sprintf("%.2f", min(vif_results$adjusted_GVIF)),
    "-",
    sprintf("%.2f", max(vif_results$adjusted_GVIF)),
    "\n"
  )
}


# ------------------------------------------------------------------------------
# 10. Save reproducibility outputs
# ------------------------------------------------------------------------------

if (!dir.exists("results")) {
  dir.create("results", recursive = TRUE)
}

write.csv(
  coefficient_results,
  file = file.path("results", "model_coefficients.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

model_fit_results <- data.frame(
  statistic = names(fit_statistics),
  value = as.numeric(fit_statistics),
  row.names = NULL
)

write.csv(
  model_fit_results,
  file = file.path("results", "model_fit_statistics.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

write.csv(
  vif_results,
  file = file.path("results", "multicollinearity_diagnostics.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

capture.output(
  sessionInfo(),
  file = file.path("results", "sessionInfo.txt")
)

cat("\nAnalysis completed successfully.\n")
cat("Outputs saved in the 'results' directory.\n")
