import numpy as np
import csv
import math
import heapq


window_pos = 0

def openCSVFile(filename):
	with open(filename,'r') as test1_file:
		test1_data=np.loadtxt(test1_file, dtype=str, delimiter=',', skiprows=1)
		data_array = np.asarray(test1_data, dtype = None)
		return data_array
	
def writeCSVfile(filename,M_mean):
	with open(filename, 'wb') as fp:
		a = csv.writer(fp, delimiter=',')
		a.writerows(M_mean)

def find_mean(m_array,data_array):
	i = 0
	j = 0
	k = 0
	count = 0
	sum = 0
	while(count<3 and j<50):
		sum = sum + float(data_array[j][i+1])
		i = i + 48;
		count = count + 1
		if(count == 3):
			sum = sum/3
			m_array[j][k] = float("{0:.2f}".format(sum))
			count = 0
			sum = 0
			j = j + 1
			i = 0
	return m_array

def find_variance(var_array,data_array,m_array):
	count = 0
	i = 0
	j = 0
	k = 0
	v = 0
	while (count < 3 and j< 50):
		s = m_array[j][k]
		p = float(data_array[j][i+1])
		v = v + (p-s)*(p-s)
		i = i + 48
		count = count + 1
		if (count == 3):
			var_array[j][k] = float("{0:.2f}".format(v));
			count = 0
			v = 0
			j = j + 1
			i = 0
	return var_array

def find_beta_sigma(m_array):
		#find b1 and b0 and sigma
	p = [[1 for x in range(1)] for x in range(49)]
	q = [[1 for x in range(1)] for x in range(49)]
	q1 = [[1 for x in range(1)] for x in range(49)]
	for i in xrange(0,49):
		p[i] = m_array[i][0]
	for i in xrange(1,50):
		q[i-1] = m_array[i][0]
	A = np.vstack([p, np.ones(len(p))]).T
	b1, b0 = np.linalg.lstsq(A, q)[0]
	#print 'm:: '+str(b1)
	#print 'c:: '+str(b0)
	for i in xrange(0,49):
		p11 = b0 + b1*float(p[i])
		q1[i] = float("{0:.2f}".format(p11))
	sig = 0
	for i in xrange(0,49):
		q11 = q1[i]-q[i]
		sig = sig + q11 * q11
	sig = sig / 50
	#print 'sigma is :: '+ str(sig)
	return (b1,b0,sig) 


def inference_by_variance(m_array,var_array,sig,b0,b1):
	var_window = [[0 for x in range(95)] for x in range(50)]	
	sig_array = [[0 for x in range(95)] for x in range(50)]	
	budget = [0,5,10,20,25]
		
	vc = 0
	while (vc<len(budget)):
		for i in xrange(0,50):
			var_window[i][0] = m_array[i][0]
			sig_array[i][0] = var_array[i][0]
		var_window = var_replace(var_window,budget[vc],sig_array,0)
		
		for i in xrange(1,95):
			for j in xrange(0,49):
				v1 = var_window[j][i-1]*b1 + b0
				var_window[j][i] = float("{0:.2f}".format(v1))
				sig_array[j][i] = sig + b1 * b1 * (sig_array[j][i-1]) 	
			var_window = var_replace(var_window,budget[vc],sig_array,i)	

		a_mean = calculate_abs_mean(var_window)
		print 'phase 2-d-variance '+str(budget[vc])+' is::'+str(a_mean)
		writeCSVfile('d-v'+str(budget[vc])+'.csv',var_window)
		vc = vc + 1

def inference_by_window(m_array,var_array,b0,b1):
	global window_pos
	window = [[0 for x in range(95)] for x in range(50)]	
	budget = [0,5,10,20,25]
	vc = 0
	while (vc<len(budget)):
		for i in xrange(0,50):
			window[i][0] = m_array[i][0]
		window = replace(window,budget[vc],0)
		for i in xrange(1,95):
			for j in xrange(0,50):
				p1 = window[j][i-1]*b1 + b0
				#print j
				window[j][i] = float("{0:.2f}".format(p1))
			#print window[49][0],window[49][1]
			window = replace(window,budget[vc],i)
		
		a_mean = calculate_abs_mean(window)
		print 'phase 2-d-window '+str(budget[vc])+' is::'+str(a_mean)
		
		writeCSVfile('d-w'+str(budget[vc])+'.csv',window)
		vc = vc + 1
		window_pos = 0

def model2_train_predict():
	global window_pos
	data_array = openCSVFile('intelTemperatureTrain.csv')  #intelHumidityTrain.csv
	budget = [0,5,10,20,25]
	
	# declaring arrays
	m_array = [[1 for x in range(1)] for x in range(50)]
	var_array = [[1 for x in range(1)] for x in range(50)]
	
	#calculating mean and variance for the 0.5th hr of training data
	m_array = find_mean(m_array,data_array)
	var_array = find_variance(var_array,data_array,m_array)
	
	#print data_array[0][0], data_array[1][1], data_array[49][1]

	window = [[0 for x in range(96)] for x in range(50)]	
	var_window = [[0 for x in range(96)] for x in range(50)]	
	var_values = [[0 for x in range(96)] for x in range(50)]	

	for i in xrange(0,50):
		window[i][0] = m_array[i][0]

	#find beta_0, beta_1 and sigma_square
	b = 0
	#window_pos = 0
	while (b < len(budget)):
		k = 1
		x = 1	
		while(k<96) :
			window = replace(window,budget[b],k-1)
			j = 1
			while(j<50) :
				p = [1 for x in range(3)]
				q = [1 for x in range(3)]
				i = x
				#p[0] = data_array[j][0] + data_array[j][0] + data_array[j][0]
				count = 0
				while (count < 3 ):
					if ((i+1)==144):
						break
					p[count] = data_array[j][i]
					#print j
					q[count] = data_array[j][i+1]
					i = i + 48
					count = count + 1
				A = np.vstack([p, np.ones(len(p))]).T
				b1, b0 = np.linalg.lstsq(A, q)[0]
				window[j-1][k] = window[j-1][k-1] * b1 + b0
				j = j + 1
			x = x + 1
			k = k + 1
		a_mean = calculate_abs_mean(window)
		print 'phase 2-h-window '+str(budget[b])+' is::'+str(a_mean)
		writeCSVfile('h-w'+str(budget[b])+'.csv',window)
		b = b + 1
		window_pos = 0

	for i in xrange(0,50):
		var_window[i][0] = m_array[i][0]
		var_values[i][0] = var_array[i][0]


	b = 0
	#window_pos = 0
	while (b < len(budget)):
		k = 1
		x = 1	
		while(k<96) :
			var_window = var_replace(var_window,budget[b],var_values,k-1)
			j = 1
			while(j<50) :
				p1 = [1 for x in range(3)]
				q1 = [1 for x in range(3)]
				q2 = [1 for x in range(3)]
				i = x
				count = 0
				while (count < 3 ):
					if ((i+1)==144):
						break
					p1[count] = float(data_array[j][i])
					#print j
					q1[count] = float(data_array[j][i+1])
					i = i + 48
					count = count + 1
				A = np.vstack([p1, np.ones(len(p1))]).T
				b1, b0 = np.linalg.lstsq(A, q1)[0]
				for i in xrange(1,3):
					q1[i] = b0 + b1 * p1[i]
				sig = 0
				for i in xrange(1,3):
					sig = sig + (q1[i]-q2[i]) * (q1[i]-q2[i])
				sig = sig/3
				var_window[j-1][k] = var_window[j-1][k-1] * b1 + b0
				var_values[j-1][k] = var_values[j-1][k-1] * b1 * b1 + sig
				j = j + 1
			x = x + 1
			k = k + 1
		a_mean = calculate_abs_mean(var_window)
		print 'phase 2-h-variance '+str(budget[b])+' is::'+str(a_mean)
		writeCSVfile('h-v'+str(budget[b])+'.csv',var_window)
		b = b + 1


def calculate_abs_mean(pred_array):
	data_array = openCSVFile('intelTemperatureTest.csv') #intelTemperatureTest.csv
	temp = [[0 for x in range(97)] for x in range(50)] 
	for j in xrange(0,49):
		for k in xrange(1,49):
			err = abs(float(data_array[j][k])-float(pred_array[j][k]))
			err1 = abs(float(data_array[j][k+48])-float(pred_array[j][k]))
			temp[j][k] = err
			temp[j][k+48] = err1
	abs_mean = 0
	for i in xrange(0,49):
		for j in xrange(1,97):
			abs_mean = abs_mean +temp[i][j]
	abs_mean = abs_mean/(50*48*2)
	return abs_mean


def model1_train_predict():
	
	data_array = openCSVFile('intelTemperatureTrain.csv') #intelHumidityTrain.csv
	
	# declaring arrays
	m_array = [[1 for x in range(1)] for x in range(50)]
	var_array = [[1 for x in range(1)] for x in range(50)]
	
	#calculating mean and variance for the 0.5th hr of training data
	m_array = find_mean(m_array,data_array)
	var_array = find_variance(var_array,data_array,m_array)
	
	#find beta_0, beta_1 and sigma_square
	b1,b0,sig = find_beta_sigma(m_array)

	#Active inference by variance approach
	inference_by_variance(m_array,var_array,sig,b0,b1)
		
	#Active inference by window approach
	inference_by_window(m_array,var_array,b0,b1)	


def var_replace(var_window,budjet, var_array, col):
	test_data = openCSVFile('intelTemperatureTest.csv') #.intelTemperatureTestcsv
	a = np.array(var_array[:col])
	indecies = heapq.nlargest(budjet, range(len(a)), a.take)
	for index in indecies:
		var_window[index][col] = float(test_data[index][col+1])
	return var_window

def replace(window, budjet,col):
	test_data = openCSVFile('intelTemperatureTest.csv')  #intelTemperatureTest
	global window_pos
	i = window_pos
	count = 0
	#print window[49][0], window[49][1]
	while (count < budjet):
		#print test_data[i][col+1]
		window[i][col] = float(test_data[i][col+1])
		i = i + 1
		window_pos = window_pos + 1
		count = count + 1
		if( i == 49 ):
			window_pos = 0
			i = 0
	return window


if __name__ == '__main__': 
	model1_train_predict() #daily
	model2_train_predict()#hourly
