# US Census Data Exploration




Census Dataset: [Link To Data](http://thomasdata.s3.amazonaws.com/ds/us_census_full.zip)  
This US Census dataset contains detailed but anonymized information for approximately 300,000 people.

The archive contains 3 files:  
1) A large training file (csv)  
2) Another test file (csv)  
3) A metadata file (txt) describing the columns of the two csv files (identical for both)

### GOAL   
1) Load the train and test files.  
2) Exploratory analysis with visualizations.  
3) Clean, preprocess, and feature engineering.  
4) Create a model to predict whether a person earns more or less than $50,000 per year.  
5) Model tuning.  
6) Final Model Selection and Application  


### CONTENT
* **US_income_census_analysis.ipynb**: Main Notebook with code, visulations, and explanations.  

* **US_income_census_analysis.html**: HTML version of the above for viewing in a browser.  

* **US_income_census_analysis_no_code.html**: Version of the above with no code.  

* **grid_search.ipynb**: Separate notebook to find better model parameters.  

* **hyperparams/**: Folder with json of best hyper-parameters from grid search.  

* **helpers.py**: Python module with a few functions to keep from cluttering notebook.  

* **custom.css**: To help format notebook markdown cells.  

* **metadata.txt**: Data description

* **requirements.txt**: Versions of packages used for exercise.
