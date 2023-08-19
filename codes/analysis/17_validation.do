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

* For our sentiments, make them shares of total tweets in a user/week pair
* For Hashtags, we are using the average per-tweet usage
foreach var in fear anger joy sadnes anger pro_lock anti_gov immuni lombardia{
	replace `var' = `var'/n_tweets
}

* Generate the polynomial for our polarization prediction
gen pred2 = prediction^2
gen pred3 = prediction^3

* Now iteratively produce our poly(3) graphs
foreach var in anger pro_lock anti_gov immuni lombardia{
	if "`var'" == "anger"{
		local ytitle = "Anger (%)"
		local round_factor = 0.001
	}
	else if "`var'" == "pro_lock"{
		local ytitle = "Pro-Lockdown - avg. hashtags per tweet"
		local round_factor = 0.0001
	}
	else if "`var'" == "anti_gov"{
		local ytitle = "Anti-Government - avg. hashtags per tweet"
		local round_factor = 0.0001
	}
	else if "`var'" == "immuni"{
		local ytitle = "Immuni - avg. hashtags per tweet"
		local round_factor = 0.00001
	}
	else if "`var'" == "lombardia"{
		local ytitle = "Lombardia - avg. hashtags per tweet"
		local round_factor = 0.0001
	}
//	
// 	binsreg `var' prediction, ///
// 	polyreg(3) polyregplotopt(lcolor(maroon)) nbins(20)  name(`var', replace) dots(3 0) ///
// 	graphregion(color(white)) ///
// 	ytitle("`ytitle'", margin(medsmall)) ///
// 	xtitle("Predicted polarization", margin(medsmall))
// 	graph export "${figures}/final/`var'_validation.pdf", as(pdf) replace
	
	
	
	* Generate binscatter bins
	binscatter `var'  prediction, genxq(bins)

	* Generate means by binscatter beans
	egen anger_m = mean(`var' ), by(bins)
	egen pred_m = mean(prediction), by(bins)

	* Estimate the polynomial regression
	reg `var'  prediction pred2 pred3
	* Predict the anger values
	predict yhat

	* Restrict the graph to the 99th percentiles
	xtile binlim = prediction, nquantiles(100)
	
	
	preserve
	* Remove unnecessary observations (needed to reduce graph size)
	keep anger_m pred_m yhat prediction binlim
	replace yhat = round(yhat, `round_factor')
	replace prediction = round(prediction, .001)
	
	
	duplicates drop
	* We restrict our graph to the 99th percentiles of our support
	twoway (scatter anger_m pred_m, mcolor("76 114 176")) ///
		   (line yhat prediction if binlim <= 99 & binlim >= 1, sort), ///
		   graphregion(color(white)) ///
		   leg(off) ///
		   ylab(, nogrid) ///
		   ytitle("`ytitle'", margin(medsmall)) ///
		   name(`var', replace) ///
		   xlab(-1(0.5)1) ///
		   xtitle("Predicted Political Alignment", margin(medsmall))
	graph export "${figures}/final/`var'_validation.pdf", as(pdf) replace
	restore
	
	drop yhat binlim anger_m pred_m bins
}

