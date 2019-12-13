# Silver Slugger Data Science Project

Edward Yuan, Daniel Quinn, Kenny McAvoy, Rayyan Karim

Regression
- The initial statistics used for creating the regression models are found in the all-np.csv, all-np14-16.csv, all-np15-17.csv, and all-np17-19.csv files.
- From there, masterRegressions.py was run using Python 3 to run a correlation analysis and create regression predictions with various models (Linear Regression, Linear SVR, SVR, Bayes Ridge, Huber, Ridge, and ARD).
- This is outputted to allPredictions2019.csv (number depends on the year). From there, Excel functions like VLOOKUP were used to match predicted output with actual statistics from Fangraphs in files like 2019PredictionsAndActual.csv. Note that recreating the project by running files will not work because additional work is required in Excel that is not reflected in the Python files.
- Accuracy (RMSE, MSE) is calculated with the Accuracy.py file.

Classification
- This is done with files SilverSluggers_NaiveBayes.py and knn.py, which output results for award winners. 

To recreate the process we used:
python masterRegressions.py\
Use Excel to match up predicted with actual stats\
python Accuracy.py\
python SilverSluggers_NaiveBayes.py\
python knn.py\

