# Surrender analysis in life insurance
Research project 2019-2021. <br/>
Accompanying code for the paper "Modelling surrender risk in life insurance: theoretical and experimental insight" by M. Kiermayer. <br/>
Paper available on https://www.tandfonline.com/doi/full/10.1080/03461238.2021.2013308 (publication) or https://arxiv.org/abs/2101.11590 (preprint). <br/>

# Goal
1) Perform extensive experiments to analyze the capabilities of several models (including logistic regression, random forest, XGBoost, neural networks bagged or boosted) to estimate surrender probabilities.  <br/>
2) Check the effect of resampling on model performance and predicted surrender probabilities <br/>
3) Investigate a time-dependent evaluation of surrender rates, including confidence bands <br/>

# Structure of the project

- The simulated data can be found in the directory "./Data", including the portfolio "Portfolio.csv" at time t=0. <br/>
        * Sub-directories "./Data/profile_{i}" include time-series data for years t>=0 wich is unique to surrender assumptions in the ith-profile, i = 0,1,2,3. <br/>
        * All data in these (sub-)directories is generated (and analyzed) by the scripts "\__data{i}\__{..}.py" <br/>
 
- The files "HPSearch_{..}.py" implement an automated hyperparameter-tuning (based on the python package 'hyperopt') <br/>

- The directories "./profile_{i}" contain the results for trials of HPTuning for all models and the resulting best model-parametrizations <br/>

- The main files analyzes all models (given the parametrization after HPTuning)

- Visual and statistical analyses of our experiments are saved in either "./Plots" or "./Tables"

- All helper-functions can be found in "./functions"
