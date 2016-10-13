import numpy as np
import csv
import math
import heapq
from sklearn import linear_model

window_pos = 0

def openCSVFile(filename):
	with open(filename,'r') as test1_file:
		test1_data=np.loadtxt(test1_file, dtype=str, delimiter=',', skiprows = 1)
		data_array = np.asarray(test1_data, dtype = None)
		return data_array
	
def writeCSVfile(filename,M_mean):
	with open(filename, 'wb') as fp:
		a = csv.writer(fp, delimiter=',')
		a.writerows(M_mean)

def train():
	global window_pos
	data_array = openCSVFile('intelHumidityTrain.csv') #intelTemperatureTest.csv
	xm = [[0 for x in range(51)] for x in range(143)]
	ym = [[0 for x in range(51)] for x in range(143)]
	beta_array = [[0 for x in range(51)] for x in range(50)]
	res = [[1 for x in range(1)] for x in range(51)]
	variance = [[1 for x in range(1)] for x in range(50)]
	m_array = [[1 for x in range(1)] for x in range(50)]
	var_array = [[1 for x in range(1)] for x in range(50)]
	pred_array = [[0 for x in range(96)] for x in range(50)]
	v_pred_array = [[0 for x in range(96)] for x in range(50)]
	sig_array = [[0 for x in range(96)] for x in range(50)]
	pred_data = [[0 for x in range(96)] for x in range(50)]
	v_pred_data = [[0 for x in range(96)] for x in range(50)]

	var_window = [[0 for x in range(96)] for x in range(50)]

	for i in xrange(0,143):
		xm[i][0] = 1
	for i in xrange(0,143):
		for j in xrange(1,51):
			xm[i][j] =  float(data_array[j-1][i+1])
	
	for j in xrange(0,50):	
		for i in xrange(1,144):
			ym[i-1][j] = float(data_array[j][i+1])	

		q, r = np.linalg.qr(xm)
		p = np.dot(q.T, ym)
		#print len(np.dot(np.linalg.inv(r), p))
		for i in xrange(0,51):
			beta_array[j][i] = np.dot(np.linalg.inv(r), p)[i][j]
		#print len(beta_array)
	#print beta_array

	for j in xrange(0,49):
		col1 = [inner[j] for inner in ym]
		res = np.matrix(xm) * np.matrix(beta_array[j]).T - np.matrix(col1).T 
		vari = 0
		for i in xrange(0,143):
			#print res.item(i,0)
			vari = vari + float(res.item(i,0)) * float(res.item(i,0))
		vari = vari/143
		variance[j][0] = vari
	
	pred_data = openCSVFile('hwp.csv')
	v_pred_data = openCSVFile('hwp.csv')
	#print pred_data

	for i in xrange(1,50):
		for j in xrange(1,96):
			pred_data[i][j] = float(pred_data[i][j])		
			v_pred_data[i][j] = float(v_pred_data[i][j])	
	#print variance
	m_array = find_mean(m_array,data_array)
	var_array = find_variance(var_array,data_array,m_array)
	#print var_array
	for i in range(0,49):
		pred_array[i][0] = m_array[i][0]
	budget = [0,5,10,20,25]
	b = 0
	while (b<len(budget)):
		pred_array = replace(pred_array,budget[b],0)
		pred_data = replace(pred_data,budget[b],0)
		for k in xrange(1,96):
			for j in xrange(0,49):
				pr = beta_array[j][0]
				for i in xrange(1,50):
					pr = pr + pred_array[i-1][k-1] * beta_array[j][i]
				pred_array[j][k] = pr
			pred_array = replace(pred_array,budget[b],k)
			pred_data = replace(pred_data,budget[b],k)
		a_mean = calculate_abs_mean(pred_data)
		print 'phase 3-h-window '+str(budget[b])+' is::'+str(a_mean)
		

		writeCSVfile('w'+str(budget[b])+'.csv',pred_data)
		b = b + 1
	
	
	for i in range(0,49):
		v_pred_array[i][0] = m_array[i][0]
		sig_array[i][0] = var_array[i][0]
	
	
	budget = [0,5,10,20,25]
	b = 0
	
	#variance by window
	while (b<len(budget)):
		v_pred_array = var_replace(v_pred_array,budget[b],sig_array,0)
		v_pred_data = var_replace(v_pred_data,budget[b],sig_array,0)
		for k in xrange(1,96):
			for j in xrange(0,49):
				pr = beta_array[j][0]
				pv = variance[j][0]
				for i in xrange(1,50):
					pr = pr + v_pred_array[i-1][k-1] * beta_array[j][i]
				v_pred_array[j][k] = pr
				for i in xrange(1,50):
					pv = pv + beta_array[j][i] * beta_array[j][i] * (sig_array[i-1][k-1]) 	
				sig_array[j][k] = pv
			v_pred_array = var_replace(v_pred_array,budget[b],sig_array,k)	
			v_pred_data = var_replace(v_pred_data,budget[b],sig_array,k)
		a_mean = calculate_abs_mean(v_pred_data)
		print 'phase 3-h-variance '+str(budget[b])+' is::'+str(a_mean)
		
		writeCSVfile('v'+str(budget[b])+'.csv',v_pred_array)
		b = b + 1
		
def var_replace(var_window,budget, var_array, col):
	test_data = openCSVFile('intelHumidityTest.csv') #.intelTemperatureTestcsv
	writeCSVfile('var_data.csv',var_array)
	var_data = openCSVFile('var_data.csv') #.intelTemperatureTestcsv
	#print col
	a = np.array(var_data[:,col])
	indecies = heapq.nlargest(budget, range(len(a)), a.take)
	#print indecies
	for index in indecies:
		var_window[index][col] = float(test_data[index][col+1])
	return var_window

def replace(window, budjet,col):
	test_data = openCSVFile('intelHumidityTest.csv')  #intelTemperatureTest
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

def find_mean(m_array,data_array):
	i = 1
	j = 0
	k = 0
	count = 0
	sum = 0
	while(count<3 and j<50):
		sum = sum + float(data_array[j][i])
		i = i + 48;
		count = count + 1
		if(count == 3):
			sum = sum/3
			m_array[j][0] = float("{0:.2f}".format(sum))
			count = 0
			sum = 0
			j = j + 1
			i = 1
	return m_array

def calculate_abs_mean(pred_array):
	data_array = openCSVFile('intelHumidityTest.csv') #intelTemperatureTest.csv
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
			var_array[j][k] = float("{0:.3f}".format(v));
			count = 0
			v = 0
			j = j + 1
			i = 0
	return var_array

if __name__ == '__main__': 
	train()