# EyeTracking_DimRed_SmartFlat

Dimensionality reduction of eye tracking data recollected from patients presenting executive disfunctions at Smart Flato of the hospital of Percy of Clamart

Welcome to the EyeTracking_DimRed_SmartFlat wiki!

Inputs:
1. To start, open "Main.py" and change the "path" value to the directory of the eye tracking data in format ".txt".

The following is an example of the format of the tsv table containing the features that the script receives (one table per patient):

Patient	C1_dur	C1_fix_max_dur	C1_fix_min_dur	C1_fix_mean_dur	C1_fix_med_dur	C1_fix_std_dur	C1_fix_var_dur	C1_fix_sum_dur	C1_fix_Skew	C1_fix_kurt	C1_fix_abs_n	C1_fix_rel_n	C1_fr	C1_sac_max_dur	C1_sac_min_dur	C1_sac_mean_dur	C1_sac_med_dur	C1_sac_std_dur	C1_sac_var_dur	C1_sac_sum_dur	C1_sac_Skew	C1_sac_kurt	C1_sac_abs_n	C1_sac_rel_n	C1_sr	max_alpha_2L	max_alpha_2R	max_alpha_2L	max_alpha_2R	mean_alpha_2L	mean_alpha_2R	median_alpha_2L	median_alpha_2R	std_alpha_2L	std_alpha_2R	var_alpha_2L	var_alpha_2R	sum_alpha_2L	sum_alpha_2R	skew_alpha_2L	skew_alpha_2R	kurt_alpha_2L	kurt_alpha_2R
06_C3_08032017	9.58	1420.00	70.00	447.33	290.00	438.56	192335.24	6710.00	1.25	3.35	15.00	0.42	1.57	190.00	10.00	49.05	30.00	49.89	2489.05	1030.00	1.77	4.91	21.00	0.58	2.19	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN

2. Change the value of the "clf" variable to "lda" or "svm" to choose the classifier (read the documentation of the script)

Output:
The script generates 3 excel files (1 for each dimensionality reduction method). Each file contains 3 tabs:
- The first one contains the training time of the selected classifier.
- The second contains the training accuracy of the selected classifier
- The third contains the testing accuracy of the selected classifier
