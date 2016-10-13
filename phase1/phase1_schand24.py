import numpy as np
import csv
import math
import heapq


def openCSVFile(filename):
	with open(filename,'r') as test1_file:
		test1_data=np.loadtxt(test1_file, dtype=str, delimiter=',')
		data_array = np.asarray(test1_data, dtype = None)
		return data_array
	
def writeCSVfile(filename,M_mean):
	with open(filename, 'wb') as fp:
		a = csv.writer(fp, delimiter=',')
		a.writerows(M_mean)

def cal_totalMean_using_window(str1,str2):
	budget = [0,5,10,15,20,25]
	data_array = openCSVFile(str1)
	pred_array = openCSVFile(str2)
	temp = [[0 for x in range(97)] for x in range(50)] 
	for j in xrange(0,49):
		for k in xrange(1,49):
			err = abs(float(data_array[j][k])-float(pred_array[j][k]))
			err1 = abs(float(data_array[j][k+48])-float(pred_array[j][k]))
			temp[j][k] = err
			temp[j][k+48] = err1
	for x in budget:
		print 'Budget is:: '+str(x)
		j= 1;
		p = 0
		c = 0
		while (j<97):
			while (c < x):
				temp[p][j] = 0
				c = c + 1
				p = p + 1
				if(p > 49):
					p =0
			j = j+1
			c = 0
		abs_mean = 0
#		writeCSVfile('w'+str(x)+'.csv',temp)
		for i in xrange(0,49):
			for j in xrange(1,97):
				abs_mean = abs_mean +temp[i][j]

		print 'absolute mean error is :: '+str(abs_mean/(50*48*2))


def cal_totalMean_using_variance(str1,str2,str3):
	budget = [0,5,10,15,20,25]
	
	data_array = openCSVFile(str1)
	pred_array = openCSVFile(str2)
	var_array = openCSVFile(str3)
	
	temp = [[0 for x in range(97)] for x in range(50)] 
	for j in xrange(0,49):
		for k in xrange(1,49):
			err = abs(float(data_array[j][k])-float(pred_array[j][k]))
			err1 = abs(float(data_array[j][k+48])-float(pred_array[j][k]))
			temp[j][k] = err
			temp[j][k+48] = err1
	
	for x in budget:
		print 'Budget is :: '+str(x)
		for i in xrange(1,49):
			a = np.array(var_array[:,i])
			indecies = heapq.nlargest(x, range(len(a)), a.take)
			for index in indecies:
				temp[index][i] = 0
				temp[index][i+48] = 0
		abs_mean = 0
#		writeCSVfile('v'+str(x)+'.csv',temp)
		for i in xrange(0,49):
			for j in xrange(1,97):
				abs_mean = abs_mean+temp[i][j]

		print 'absolute mean error is:: '+str(abs_mean/(50*48*2))

def model_train_predict(str1,str2,str3,str4):
	data_array = openCSVFile(str1)
	#print data_array
	s = 0
	M_mean = [[0 for x in range(49)] for x in range(50)]
	M_var = [[0 for x in range(49)] for x in range(50)] 
	pred = [[0 for x in range(49)] for x in range(50)] 
	for i in xrange(0,50):
		for k in xrange(1,49):
			j = k
			s = 0
			p = 0
			m = 0
			while (p<3 and j<=144):
	 	 		s = s + float(data_array[i][j])
	 	 		#print data_array[i][j]
	 	 		j = j + 48
	 	 		p = p+1
		 	s = float("{0:.2f}".format(s/3))
		 	M_mean[i][k] = s
		 	#print M_mean
		 	v = 0
		 	y = 0
		 	j = k		 		
		 	while (m<3 and j<=144):
	 			y = float(data_array[i][j])
	 			v = v + (y - s)*(y - s)
	 			j = j + 48
	 			m = m + 1
	 		M_var[i][k] = v/3
			t = np.random.normal(M_mean[i][k],math.sqrt(M_var[i][k]))
			pred[i][k] = t				
	writeCSVfile(str2,M_mean)		
	# with open(str2, 'wb') as fp:
	# 	a = csv.writer(fp, delimiter=',')
	# 	a.writerows(M_mean)
 	writeCSVfile(str3,M_var)		
	writeCSVfile(str4,pred)		
	
 	
 	

if __name__ == '__main__':  
	
	print 'Temprature::\n'
	model_train_predict('intelTemperatureTrain.csv','temp_mean.csv','temp_variance.csv','temp_predction.csv')
	print 'Active Inference using Variance method::\n'
 	cal_totalMean_using_variance('intelTemperatureTest.csv','temp_predction.csv','temp_variance.csv')
	print 'Active Inference using Window method::\n'
	cal_totalMean_using_window('intelTemperatureTest.csv','temp_predction.csv')	

	print 'Humidity::\n'
	model_train_predict('intelHumidityTrain.csv','hum_mean.csv','hum_variance.csv','hum_predction.csv')
	print 'Active Inference using Variance method::\n'
 	cal_totalMean_using_variance('intelHumidityTest.csv','hum_predction.csv','hum_variance.csv')
 	print 'Active Inference using Window method::\n'
	cal_totalMean_using_window('intelHumidityTest.csv','hum_predction.csv')	
