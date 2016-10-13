
Course : CS583
Name : Saibabu Chandramouli
Cwid : A20345775

Readme file for project phase 2:

Language used : python 2.7.11
packages used : numpy, math, csv, heapq

functions: 
	
	openCSVFile(filename) : function takes file name as input parameter and opens csv file and return the data in csv file as array

	writeCSVFile(filename,array) : function takes files name and array as inputs and creates a new csv file or writes csv file if exists with given array of data.

	var_replace(var_window,budjet, var_array, col) this method takes the budget as reference and replace the highest variance values in a give col and returns the result var_window array.

	replace(window, budjet,col): this method takes budget value as reference and replace the window array values of a given col with testdata values from the actual intel readings.


	model1_train_predict: this method acts as driver function for the day stationary model.

		inthis approach I have taken same b1,b0 for predicting the 2 days data. I have only one b1,bo for the 2days prediction of data.

		first it calculates the values of mean
		then it calculates the b0,b1 and sigma values
		based on the budget it calls the inference by window
		similarly for based on the budget values it calls the inference by variance and finally caluculates the mean absolute error

	model2_train_predict: this method acts as driver function for the day stationary model.

		inthis approach I have taken different b1,b0 for each and every prediction. in total i have got 96*50 sets of b1,b0 values

		first it calculates the values of mean
		then it calculates the b0,b1 and sigma values
		based on the budget it calls the inference by window
		similarly for based on the budget values it calls the inference by variance and finally caluculates the mean absolute error
