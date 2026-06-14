* ----------------------------------------------------------------------------
* Marketing Analytics Project 2: What drives AI (Gemini vs ChatGPT) to mention brands?
* ----------------------------------------------------------------------------

clear all
set more off
set scheme s1color  

* ----------------------------------------------------------------------------
* 1. LOAD DATA & CLEANING
* ----------------------------------------------------------------------------
import delimited "30420_Project2_data_Group.csv", clear 

* Fix European decimals
capture replace avgrating_b_g2 = subinstr(avgrating_b_g2, ",", ".", .)
destring avgrating_b_g2, replace

* Encode string variables
encode brand, gen(brand_id)
encode topic, gen(topic_id)
encode source, gen(source_id)

* ----------------------------------------------------------------------------
* 2. DIAGNOSTIC PHASE: CHECKING INDIVIDUAL REGRESSORS
* ----------------------------------------------------------------------------
* Before aggregation, we inspect the individual granular variables.
* We look for Multicollinearity (Correlation > 0.7) to group or summarize those variables.

* Create Log Transformations for granular variables
gen ln_reddit   = ln(1 + reddit_topic_hits_bt)
gen ln_youtube  = ln(1 + youtube_topic_hits_bt)
gen ln_linkedin = ln(1 + linkedin_topic_hits_bt)
gen ln_listicle = ln(1 + listicle_topic_hits_bt)
gen ln_domain   = ln(1 + domain_topic_hits_bt)
gen ln_reviews  = ln(1 + reviewcount_b_g2)
gen raw_rating  = avgrating_b_g2

* A. PAIRWISE CORRELATIONS 
pwcorr ln_reddit ln_youtube ln_linkedin ln_listicle ln_domain raw_rating ln_reviews, star(0.05)

preserve
graph matrix ln_reddit ln_youtube ln_linkedin ln_listicle ln_domain raw_rating ln_reviews, ///
    title("Correlation Matrix") 
graph export "granular_correlation.png", replace
restore

* ----------------------------------------------------------------------------
* 3. FEATURE ENGINEERING (THE AGGREGATION STRATEGY)
* ----------------------------------------------------------------------------
* DECISION: 
* 1. Reddit & LinkedIn are highly correlated (>0.90) -> Group into "Social Buzz".
* 2. Listicles & Domain are correlated -> Group into "Articles".
* 3. Rating & Reviews are correlated -> Interaction term "Review Index".

* A. Social Buzz (Reddit + YouTube + LinkedIn)
gen raw_social_buzz = reddit_topic_hits_bt + youtube_topic_hits_bt + linkedin_topic_hits_bt
gen ln_social_buzz  = ln(1 + raw_social_buzz)

* B. Articles (Listicles + Domain)
gen raw_articles    = listicle_topic_hits_bt + domain_topic_hits_bt
gen ln_articles     = ln(1 + raw_articles)

* C. Review Index (Rating * Volume Interaction)
gen raw_review_score = avgrating_b_g2 * ln(1 + reviewcount_b_g2)

* D. Standardization (Z-Scores for Model Comparability)
egen z_social_buzz  = std(ln_social_buzz)
egen z_articles     = std(ln_articles)
egen z_review_index = std(raw_review_score)
egen z_seo          = std(lighthouse_seo_b)

* ----------------------------------------------------------------------------
* 4. VISUALIZATION OF FINAL LEVERS
* ----------------------------------------------------------------------------

* Signal Strength 
graph box z_social_buzz z_articles z_review_index, ///
    over(mention, label(labsize(small))) ///
    by(source, title("Signal Strength: Gemini vs ChatGPT") note("")) ///
    subtitle("Standardized Scores by Mention Status") ///
    ytitle("Z-Score") legend(off)
graph export "signal_strength.png", replace


* Mention Rate by Model
graph hbar (mean) mention, ///
    over(source, label(labsize(small))) ///
    over(brand, label(labsize(small))) ///
    title("Mention Rate by Model") ///
    ytitle("Probability of Mention") ///
    bar(1, color(navy)) ///
    blabel(bar, format(%9.2f))
graph export "mention_rate.png", replace

* Final Correlation Matrix
preserve
graph matrix ln_social_buzz ln_articles raw_review_score lighthouse_seo_b, ///
    half msymbol(point) mcolor(black%20) ///
    title("Correlation of Final Levers")
graph export "final_correlation.png", replace
restore

preserve
collapse (mean) mention_prob=mention, by(brand topic source)

* GEMINI PLOT
graph hbar (asis) mention_prob if source=="gemini", ///
    over(brand, sort(1) descending label(labsize(tiny))) ///
    over(topic, label(labsize(small))) ///
    title("Gemini Win Rates") ///
    ytitle("Probability") bar(1, color(maroon)) ///
    name(viz_topic_gemini, replace)

* CHATGPT PLOT
graph hbar (asis) mention_prob if source=="chatgpt", ///
    over(brand, sort(1) descending label(labsize(tiny))) ///
    over(topic, label(labsize(small))) ///
    title("ChatGPT Win Rates") ///
    ytitle("Probability") bar(1, color(forest_green)) ///
    name(viz_topic_chatgpt, replace)

* COMBINE
graph combine viz_topic_gemini viz_topic_chatgpt, ///
    title("Comparison: Who wins where?") xsize(20) ysize(10)
graph export "viz_F_topic_breakdown.png", replace
restore

* ----------------------------------------------------------------------------
* 5. UNIFIED REGRESSION MODEL (Pooled Clogit)
* ----------------------------------------------------------------------------

* We use the POOLED model because the Likelihood Ratio Test confirmed 
* that the coefficients (Drivers) are statistically identical between AIs.
* We control for Source (i.source_id) to capture the 'query complexity' difference.

clogit mention z_social_buzz z_articles z_review_index z_seo i.source_id, ///
    group(query_id) vce(cluster query_id) or

* ----------------------------------------------------------------------------
* 6. MARGINAL EFFECTS (AME) & VISUALIZATION
* ----------------------------------------------------------------------------

* A. Calculate Average Marginal Effects
margins, dydx(*) post

* B. Visualize the Effects
* This plots the effect size and confidence intervals.
* If the horizontal line crosses 0, the driver is NOT significant.
marginsplot, horizontal xline(0) yscale(reverse) ///
    recast(scatter) ///
    title("What actually drives a mention?", size(medium)) ///
    subtitle("Average Marginal Effects (95% CI)") ///
    xtitle("Change in Probability (0.1 = +10 pp)") ///
    ylabel(, labsize(small) notick nogrid) ///
    scheme(s1color) ///
    name(viz_AME_forest, replace)

graph export "AME.png", replace