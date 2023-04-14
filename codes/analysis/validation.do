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


binscatter anger prediction

foreach var in anger pro_lock anti_gov immuni lombardia{
	replace `var' = `var'/n_tweets
}	

foreach var in anger pro_lock anti_gov immuni lombardia{
	binsreg `var' prediction, polyreg(3) polyregplotopt(lcolor(maroon)) nbins(40)  name(`var', replace) dots(3 0) ///
	graphregion(color(white))
	graph export "${figures}/final/`var'_validation.pdf", as(pdf) replace
}
 
hist pred, graphregion(color(white)) color(ltblue) frac lcolor(navy)
graph export "${figures}/final/prediction_histogram.pdf", as(pdf) replace

/*


TO BE CONSIDERED
- Controls? Individual FE? Anything?
- Print some statistics/fit stuff
*/
