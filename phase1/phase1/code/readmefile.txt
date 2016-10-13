
Course : CS583
Name : Saibabu Chandramouli
Cwid : A20345775

Readme file for project phase 1:

Language used : python 2.7.11
packages used : numpy, math, csv, heapq

functions: 
	
	openCSVFile(filename) : function takes file name as input parameter and opens csv file and return the data in csv file as array

	writeCSVFile(filename,array) : function takes files name and array as inputs and creates a new csv file or writes csv file if exists with given array of data.

	model_train_predict(): function will do the following tasks
		1) reads the train data csv file
		2) calculates the mean of each sensor at each time stamp (there are 3 datasets with 50 sensors and 48 readings per day) and save mean data to a csv file for the future usage
		3) calculate variance of each sensor at each time stamp similar to mean and save data to csv file for future usage
		4)by using above calculate mean and variance predict the data using 
			numpy.random.norma(mean,varaince) 
		for each sensor and each timestamp and save data in a csv file.

	cal_totalMean_using_window() : function will do the following tasks
		1) reads previously saved prediction arry from above method.
		2) reads IntelTestdata(temperature and humidity).
		3) calculate the difference between each pair of instances and save the data into new array
		4) based on budget given make 0s in new array created in above step.
		5) calculate absolute mean eror for the entire dataset generated in above step

	cal_totalMean_using_Variance():  function will do following tasks.
		1) reads the previously saved prediction array from above method.
		2) reads IntelTestdata(temparature and humidity)
		3) calculate difference beween each pair of values for all sensors and all timestamps
		4) for each column findout indexes with maximum variance by using variance arry created in method train_predict() and make them zeros based on budget values.
		5) calculate absolute mean error.
	Main(): fuction calls the corresponding methods with appropriate file names.







