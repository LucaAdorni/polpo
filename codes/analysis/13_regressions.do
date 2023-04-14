/*
STATA program to analyze changes to political polarization
Author: Luca Adorni
Date: January 2023

 NOTE: You need to set the Stata working directory to the path
 where the data file is located.

 Index
 0. SETUP
 1. REGRESSIONS - Simple DiD: Simple DiD specification to get the change in the distributions pre/post treatment
 2. REGRESSIONS - Event Study: Event study for missing and excess jobs pre/post treatment
	
*/
 
**# Bookmark #0
* 0. SETUP ---------------------------------------------------------------------
clear all

capture log close polpo_reg

set more off

*setup folder directory
if "`c(username)'" == "ADORNI"{
	global main_dir "/Users/ADORNI/Dropbox (BFI)"
}
else{
	global main_dir `c(pwd)'
}

global code_repo "`c(pwd)'/polpo/codes/"
global repo "${main_dir}/LUCA/polpo/"
global data "${repo}data/"
global raw "${data}raw/"
global processed "${data}processed/"
global logs "${repo}logs/"
global figures "${repo}figures/"
global tables "${repo}tables/"


log using "${logs}12_reg.log", text replace name(polpo_reg)

 
**# Bookmark #1
* 1. Import Data ---------------------------------------------------------------------

* Load our main data
gzuse "${processed}final_df_analysis.dta.gz", clear

* Encode our covariates
encode dist, gen(dist_enc)
encode gender, gen(gender_enc)
encode age, gen(age_enc)
replace region = "other" if region == ""
encode region, gen(region_enc)


foreach var in fear anger joy sadnes{
	replace `var' = `var'/n_tweets
}

* Define a global macro to store all our variables
global store ""

foreach var in center_left far_right center_right far_left center orient_change_toleft orient_change_toright extremism_toright extremism_toleft anger fear joy sadness{
	* Summarize the value at week = t - 1 (i.e. our reference category)
	sum `var' if dist_enc == 7
	local mean_var = r(mean)

	* Perform our regression
	reghdfe `var' ib7.dist_enc, vce(cluster scree_name) absorb(scree_name)


	* Store all our results
	cap drop b`var'
	cap drop se`var'
	gen b`var' = .
	gen se`var' = .
	levelsof dist_enc, local(iter_enc)
	foreach i in `iter_enc'{
		if `i' == 7{
			continue
		}
		* Get the increase/decrease in % relative to the omitted variable
	// 	lincom (_b[`i'.dist_enc]/`mean_var')
	// 	replace b`var' = r(estimate) if dist_enc == `i'
	// 	replace se`var' = r(se)  if dist_enc == `i'
		qui replace b`var' = _b[`i'.dist_enc] if dist_enc == `i'
		qui replace se`var' = _se[`i'.dist_enc]  if dist_enc == `i'
	}


	gen ul`var' = b`var' + 1.96*se`var'
	gen ll`var' = b`var' - 1.96*se`var'
	global store "${store} b`var' se`var' ul`var' ll`var'"

	preserve
	keep ${store} week_start dist dist_enc
	duplicates drop

	sort week_start

	gen ul = b`var' + 1.96*se`var'
	gen ll = b`var' - 1.96*se`var'

	twoway (line b`var' week_start) (line ul week_start) (line ll week_start), name(`var', replace)
	restore
	local i = 7
	if `i' == 7{
		replace b`var' = _b[_cons] if dist_enc == `i'
		replace se`var' = _se[_cons] if dist_enc == `i'
	}
}

keep ${store} week_start dist dist_enc
duplicates drop
save "${processed}reg_results.dta", replace