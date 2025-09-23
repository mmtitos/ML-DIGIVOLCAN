import sys
import os
#import subprocess
import collections
#import matplotlib.pyplot as plt
import numpy as np
#from matplotlib import cm, ticker
import time
import datetime
from datetime import datetime
from datetime import timedelta
#import matplotlib as mpl
#from decimal import getcontext, Decimal
#from matplotlib.animation import FuncAnimation
#import io
#import getpass
import csv
#from itertools import zip_longest
#import logging as logger
import pandas as pd
import glob
import matplotlib.pyplot as plt
from calendar import monthrange

#plt.switch_backend('agg')
global rnn_equ_station1
global rnn_equ_station2
global rnn_equ_station3
global rnn_equ_station4
global rnn_equ_station5
global tcn_equ_station1
global tcn_equ_station2
global tcn_equ_station3
global tcn_equ_station4
global tcn_equ_station5
global rnn_stations
global tcn_stations
global data_ordered


rnn_equ_station1 = {}
rnn_equ_station2 = {}
rnn_equ_station3 = {}
rnn_equ_station4 = {}
rnn_equ_station5 = {}
tcn_equ_station1 = {}
tcn_equ_station2 = {}
tcn_equ_station3 = {}
tcn_equ_station4 = {}
tcn_equ_station5 = {}

rnn_stations=[]
tcn_stations=[]
data_ordered=[]
global events
events=['SIL','TRE','HYB','EQ','LPE']
global events_file
events_file=['Sil', 'Tre', 'Hyb', 'Equ', 'Lpe']

def convert_second_to_hour_min_sec(second):

	data=str(timedelta(seconds=int(second))).split(':')
	return data

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def Reading_data(filename,station, year, month, day, model, event):

	file_name=filename
	count=0
	year=year
	month=month
	day=day
	hour=0
	minu=0
	sec=0
	win_len=[2, 40, 14, 4, 4]
	#events=['SIL','TRE','HYB','EQ','LPE']
	#events_file=[' Sil', 'Tre', 'Hyb', 'Equ', 'Lpe']
	keys_inserted=[]
	'''
	dtime = datetime(year, month, day, hour,minu,sec)
	print("Datetime: ", dtime)

	dtimestamp = dtime.timestamp()
	print("Integer timestamp in seconds: ",
	int(round(dtimestamp)))

	print (timedelta(seconds=666))

	print (convert_second_to_hour_min_sec(666))
	#milliseconds = int(round(dtimestamp * 1000))
	#print("Integer timestamp in milliseconds: ",
	#milliseconds)
	'''
	with open(file_name, 'rt') as file_h:
		for line in file_h:
			if (line.startswith(' predi-> ')):#+ events_file[events.index(event)])):
				# introduce data in dictionary
				# go 2 lines later and get the propability
				splitted=line.split('|')
				event_detected=splitted[0].split('->')
				event_detected=event_detected[1].replace(' ','')
				event_detected=event_detected.replace('\t','')
				splitted=splitted[1].split('---')
				if (splitted[1]=='until end\n'):
					splitted[1]='86400'
				data=convert_second_to_hour_min_sec(float(splitted[0]))
				#avoid data up to 24 hours
				if (not data[0].startswith('1 day')):
					dtime = datetime(year, month, day, int(data[0]),int(data[1]),int(data[2]))
					dtimestamp = dtime.timestamp()
					key=''.join(str(dtimestamp))
					if (key in (keys_inserted)):
						key=float(key)+0.5
						key=''.join(str(key))
					tuple_data=[]
					duration=float(splitted[1])-float(splitted[0])
					tuple_data.append(duration)
					#compute probability
					line=next(file_h)
					line1=next(file_h)
					if (model=='tcn'):
						line2=next(file_h)
						probabilities=(line1+line2).replace('\n','')
						#probabilities=probabilities.replace('   ', ',')
						#probabilities=probabilities.replace('  ', '')

					else:
						if(line1.find(']') <0):
							line2=next(file_h)
						probabilities=(line1+line2).replace('\n','')	
						#probabilities=(line1).replace('\n','')
					probabilities=probabilities.replace(' ', '')
					index=find(probabilities,'.')

					for i in range (1,len(index),1):
						probabilities = ''.join((probabilities[:index[i]+i-2],',',probabilities[index[i]+i-2:])) #Character at specific pos

					probabilities=probabilities.replace('[', '')
					probabilities=probabilities.replace(']', '')
					probabilities=probabilities.split(',')

					probabilities= list(map(float, probabilities))
					tuple_data.append(100*np.array(probabilities))
					#tuple_data.append(float(probabilities[events.index(event)])*100)
					tuple_data.append(event_detected)
					if (model=='rnn'):
						if (station=='1'):
							rnn_equ_station1[key]= tuple_data
						elif (station=='2'):
							rnn_equ_station2[key]= tuple_data
						elif (station=='3'):
							rnn_equ_station3[key]= tuple_data
						elif (station=='4'):
							rnn_equ_station4[key]= tuple_data
						else:
							#event=rnn_equ_station3.get(key)
							rnn_equ_station5[key]= tuple_data
					else:
						if (station=='1'):
							#event=tcn_equ_station1.get(key)
							tcn_equ_station1[key]= tuple_data
						elif (station=='2'):
							#event=tcn_equ_station2.get(key)
							tcn_equ_station2[key]= tuple_data
						elif (station=='3'):
							#event=tcn_equ_station2.get(key)
							tcn_equ_station3[key]= tuple_data
						elif (station=='4'):
							#event=tcn_equ_station2.get(key)
							tcn_equ_station4[key]= tuple_data
						else:
							#event=tcn_equ_station3.get(key)
							tcn_equ_station5[key]= tuple_data
					keys_inserted.append(key)
					#count+=1

	#return count

def Reading_data_real_time(filename,station, year, month, day, model, event):

	file_name=filename
	count=0
	year=year
	month=month
	day=day
	hour=0
	minu=0
	sec=0
	win_len=[2, 40, 14, 4, 4]
	#events=['SIL','TRE','HYB','EQ','LPE']
	#events_file=[' Sil', 'Tre', 'Hyb', 'Equ', 'Lpe']
	keys_inserted=[]
	'''
	dtime = datetime(year, month, day, hour,minu,sec)
	print("Datetime: ", dtime)

	dtimestamp = dtime.timestamp()
	print("Integer timestamp in seconds: ",
	int(round(dtimestamp)))

	print (timedelta(seconds=666))

	print (convert_second_to_hour_min_sec(666))
	#milliseconds = int(round(dtimestamp * 1000))
	#print("Integer timestamp in milliseconds: ",
	#milliseconds)
	'''
	with open(file_name, 'rt') as file_h:
		for line in file_h:
			if (line.startswith(' predi-> ')):#+ events_file[events.index(event)])):
				# introduce data in dictionary
				# go 2 lines later and get the propability
				splitted=line.split('|')
				event_detected=splitted[0].split('->')
				event_detected=event_detected[1].replace(' ','')
				event_detected=event_detected.replace('\t','')
				splitted=splitted[1].split('---')
				if (splitted[1]=='until end\n'):
					splitted[1]='86400'
				data=convert_second_to_hour_min_sec(float(splitted[0]))
				#avoid data up to 24 hours
				if (not data[0].startswith('1 day')):
					dtime = datetime(year, month, day, int(data[0]),int(data[1]),int(data[2]))
					dtimestamp = dtime.timestamp()
					key=''.join(str(dtimestamp))
					if (key in (keys_inserted)):
						key=float(key)+0.5
						key=''.join(str(key))
					tuple_data=[]
					duration=float(splitted[1])-float(splitted[0])
					tuple_data.append(duration)
					#compute probability
					line=next(file_h)
					line1=next(file_h)
					if (model=='tcn'):
						line2=next(file_h)
						probabilities=(line1+line2).replace('\n','')
						#probabilities=probabilities.replace('   ', ',')
						#probabilities=probabilities.replace('  ', '')

					else:
						if(line1.find(']') <0):
							line2=next(file_h)
							probabilities=(line1+line2).replace('\n','')
						else:	
							probabilities=(line1).replace('\n','')
					probabilities=probabilities.replace(' ', '')
					index=find(probabilities,'.')

					for i in range (1,len(index),1):
						probabilities = ''.join((probabilities[:index[i]+i-2],',',probabilities[index[i]+i-2:])) #Character at specific pos

					probabilities=probabilities.replace('[', '')
					probabilities=probabilities.replace(']', '')
					probabilities=probabilities.split(',')

					probabilities= list(map(float, probabilities))
					tuple_data.append(100*np.array(probabilities))
					#tuple_data.append(float(probabilities[events.index(event)])*100)
					tuple_data.append(event_detected)
					if (model=='rnn'):
						rnn_stations[station][key]= tuple_data
					else:
						tcn_stations[station][key]= tuple_data
					keys_inserted.append(key)
					#count+=1

	#return count
def Histogram_plotting_day(station, data_ordered, variable_type, event, model):


	#events=['SIL','TRE','HYB','EQ','LPE']
	#events_file=[' Sil', 'Tre', 'Hyb', 'Equ', 'Lpe']
	whole_keys= list(data_ordered)
	whole_keys2=list(map(float, whole_keys))
	timestamp = float(whole_keys[0])
	dt_object = datetime.fromtimestamp(int(timestamp))
	init_hour=dt_object.hour
	init_day=dt_object.day
	init_month=dt_object.month
	init_year=dt_object.year
	init_min=dt_object.minute
	dtime = datetime(init_year, init_month, init_day, 0,0,0)

	directory = model+'_'+variable_type+'_'+event+str(init_year)+'-'+str(init_month)+'-'+str(init_day)+station
	if (not os.path.exists(directory)):
		os.makedirs(directory)

	data_plot=[]
	count=0
	for k in whole_keys:
		tuple_data=data_ordered[k]
		if (np.argmax(tuple_data[1])==events.index(event) and (tuple_data[2]==events_file[events.index(event)])):
			if (variable_type=='duration'):
				data_plot.append(tuple_data[0])
			else:
				data_plot.append(tuple_data[1][events.index(event)])
	fig, axs = plt.subplots(1, 1,
                        figsize =(10, 7),
                        tight_layout = True)


	axs.hist(data_plot, bins = 20)
	plt.ylabel('num of events')
	# Show plot
	if (variable_type=='duration'):
		plt.title('Historam Duration: %s'%(event))

		plt.xlabel('second [s]')
		namefile= directory+"/histogram_duration %s complete day.png"%(event)
		print (namefile)
		plt.savefig(namefile, dpi=200)

	else:
		plt.title('Historam Probabilities: %s '%(event))
		plt.xlabel('probability [%]')
		namefile= directory+"/histogram_probability %s complete day.png"%(event)
		print (namefile)
		plt.savefig(namefile, dpi=200)
	#plt.show()
	plt.close()


def Histogram_plotting_hours(station, data_ordered, variable_type, event, model):

	#timestamp_list=[]
	#events=['SIL','TRE','HYB','EQ','LPE']
	#events_file=[' Sil', 'Tre', 'Hyb', 'Equ', 'Lpe']
	whole_keys= list(data_ordered)
	whole_keys2=list(map(float, whole_keys))
	timestamp = float(whole_keys[0])
	dt_object = datetime.fromtimestamp(int(timestamp))
	init_hour=dt_object.hour
	init_day=dt_object.day
	init_month=dt_object.month
	init_year=dt_object.year
	init_min=dt_object.minute
	dtime = datetime(init_year, init_month, init_day, 0,0,0)
	dtimestamp = dtime.timestamp()
	low_index=0

	directory = model+'_'+variable_type+'_'+event+str(init_year)+'-'+str(init_month)+'-'+str(init_day)+station
	if (not os.path.exists(directory)):
		os.makedirs(directory)

	for i in range (0, 24, 1):
		timestamp_by_hour= dtime + timedelta(hours=i)
		dtimestamp_high=timestamp_by_hour.timestamp()
		index=[ n for n,i in enumerate(whole_keys2) if i> dtimestamp_high][0]
		if (i==23):
			index=-1
		data_plot=[]
		for k in whole_keys[low_index:index]:
			tuple_data=data_ordered[k]
			if (np.argmax(tuple_data[1])==events.index(event) and (tuple_data[2]==events_file[events.index(event)])):
				print (tuple_data)
				if (variable_type=='duration'):
					data_plot.append(tuple_data[0])
				else:
					data_plot.append(tuple_data[1][events.index(event)])

		fig, axs = plt.subplots(1, 1,
							figsize =(10, 7),
							tight_layout = True)

		axs.hist(data_plot, bins = 20)
		plt.ylabel('num of events')
		# Show plot
		if (variable_type=='duration'):
			plt.title('Historam Duration: %s %s hour'%(event,i))

			plt.xlabel('second [s]')
			namefile= directory+"/histogram_duration %s %s hour.png"%(event,i)
			print (namefile)
			plt.savefig(namefile, dpi=200)
		else:
			plt.title('Historam Probabilities: %s %s hour'%(event,i))
			plt.xlabel('probability [%]')
			namefile= directory+"/histogram_probability %s %s hour.png"%(event, i)
			print (namefile)
			plt.savefig(namefile, dpi=200)
		#plt.show()

		plt.close()
		low_index=index
def Histogram_plotting(station, data_ordered, variable_type, date_type, event, model):

	if (date_type=='day'):
		Histogram_plotting_day(station, data_ordered, variable_type, event, model)
	else:
		Histogram_plotting_hours(station, data_ordered, variable_type, event, model)

def Correlate_Events(data_station1, data_station2, data_station3, event, sec):

	count =0
	station_list_len= [len(list(data_station1)), len(list(data_station2)), len(list(data_station3))]
	station_list=[data_station1, data_station2, data_station3]
	station_reference = np.argmax(np.array(station_list_len))
	sort_index = np.argsort(station_list_len)
	print (len(list(data_station1)), len(list(data_station2)), len(list(data_station3)))
	print (sort_index[-1])
	print (station_reference)
	station_reference=station_list[sort_index[-1]]
	whole_keys= list(station_reference)
	for i in range (len(sort_index)-1):
		print (i)
		print ('###################################################################3')
		correlated_station=station_list[sort_index[i]]
		filename='correlation_between_stations'+str(sort_index[-1])+'-'+str(sort_index[i])+'.txt'
		with open(filename, 'w') as file_h:
			whole_keys2= list(correlated_station)
			whole_keys2_float=list(map(float, list(correlated_station)))
			for j in whole_keys:
				tuple_data=station_reference[j]
				if (tuple_data[2]==events_file[events.index(event)]):
					#print (tuple_data)
					timestamp = float(j)
					#print (timestamp)
					dt_object = datetime.fromtimestamp(int(timestamp))
					timestamp_low= dt_object - timedelta(seconds=sec)
					timestamp_high= dt_object + timedelta(seconds=sec)
					dtimestamp_low=timestamp_low.timestamp()
					dtimestamp_high=timestamp_high.timestamp()
					#print ('******************')
					#print (dtimestamp_low)
					#print(dtimestamp_high)
					#print (list(whole_keys2_float))
					#print ('******************')
					try:
  						index_high=[ n for n,i in enumerate(whole_keys2_float) if i> dtimestamp_high][0]
					except:
						index_high=[ n for n,i in enumerate(whole_keys2_float) if i< dtimestamp_high][0]
					try:
  						index_low=[ n for n,i in enumerate(whole_keys2_float) if i> dtimestamp_low][0]
					except:

						index_low=[ n for n,i in enumerate(whole_keys2_float) if i< dtimestamp_low][0]

					if (index_high==index_low):
						index_high=index_high+1
					#print (index_low,index_high)
					#print (whole_keys2[index_low], whole_keys2[index_high])
					#print (whole_keys2[index_low:index_high])
					count+=1
					file_h.write(dt_object.strftime("%Y-%m-%d %H:%M:%S"))
					file_h.write(' ')
					file_h.write(str(tuple_data))
					file_h.write('\n')
					for k in whole_keys2[index_low:index_high]:
						timestamp2 = float(k)
						dt_object2 = datetime.fromtimestamp(int(timestamp2))
						tuple_data_correlated= correlated_station[k]
						file_h.write('\t')
						file_h.write(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
						file_h.write(' ')
						file_h.write(str(tuple_data_correlated))
						file_h.write('\n')


def get_Correlated_Events_previousVersion(data_station1, data_station2, data_station3, event, sec, probability, date):
	station_list_len= [len(list(data_station1)), len(list(data_station2)), len(list(data_station3))]
	station_list=[data_station1, data_station2, data_station3]
	station_reference = np.argmax(np.array(station_list_len))
	sort_index = np.argsort(station_list_len)
	print (len(list(data_station1)), len(list(data_station2)), len(list(data_station3)))
	print (sort_index[-1])
	print (station_reference)
	station_reference=station_list[sort_index[-1]]
	whole_keys= list(station_reference)
	correlated_station=station_list[sort_index[0]]
	correlated_station2=station_list[sort_index[1]]
	filename='good_correlation_detected_(station_reference:'+str(sort_index[-1])+')_pruebaseptiembre.txt'
	with open(filename, 'w') as file_h:
		whole_keys2= list(correlated_station)
		whole_keys2_float=list(map(float, list(correlated_station)))
		whole_keys3= list(correlated_station2)
		whole_keys3_float=list(map(float, list(correlated_station2)))
		text="Reference station "+ str([sort_index[-1]])
		file_h.write(text)
		file_h.write('\n')
		for j in whole_keys:
			tuple_data=station_reference[j]
			if (tuple_data[2]==events_file[events.index(event)]):
				timestamp = float(j)
				dt_object = datetime.fromtimestamp(int(timestamp))
				timestamp_low= dt_object - timedelta(seconds=sec)
				timestamp_high= dt_object + timedelta(seconds=sec)
				dtimestamp_low=timestamp_low.timestamp()
				dtimestamp_high=timestamp_high.timestamp()
				try:
					index_high=[ n for n,i in enumerate(whole_keys2_float) if i> dtimestamp_high][0]
					index_high_station2=[ n for n,i in enumerate(whole_keys3_float) if i> dtimestamp_high][0]
				except:
					index_high=[ n for n,i in enumerate(whole_keys2_float) if i< dtimestamp_high][0]
					index_high_station2=[ n for n,i in enumerate(whole_keys3_float) if i< dtimestamp_high][0]
				try:
					index_low=[ n for n,i in enumerate(whole_keys2_float) if i> dtimestamp_low][0]
					index_low_station2=[ n for n,i in enumerate(whole_keys3_float) if i> dtimestamp_low][0]
				except:
					index_low=[ n for n,i in enumerate(whole_keys2_float) if i< dtimestamp_low][0]
					index_low_station2=[ n for n,i in enumerate(whole_keys3_float) if i< dtimestamp_low][0]
				if (index_high==index_low):
					index_high=index_high+1
				if (index_high_station2==index_low_station2):
					index_high_station2=index_high_station2+1
				file_h.write(dt_object.strftime("%Y-%m-%d %H:%M:%S"))
				file_h.write(' ')
				file_h.write(str(tuple_data))
				file_h.write('\n')
				text_write=False
				text_write2=False
				for k in whole_keys2[index_low:index_high]:
					timestamp2 = float(k)
					dt_object2 = datetime.fromtimestamp(int(timestamp2))
					tuple_data_correlated= correlated_station[k]
					if ((tuple_data_correlated[2]==events_file[events.index(event)]) and (tuple_data_correlated[1][events.index(event)] > probability) and (np.absolute(timestamp-timestamp2)<=2*sec)): 
						file_h.write('\t')
						if (text_write==False):
							text='Correlated station '+str(sort_index[0])
							file_h.write(text)
							file_h.write('\n')
							file_h.write('\t')
							text_write=True
						file_h.write(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
						#time_stamp_station.append(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
						file_h.write(' ')
						file_h.write(str(tuple_data_correlated))
						#tuple_station.append(str(tuple_data_correlated))
						file_h.write('\n')
				for k in whole_keys3[index_low_station2:index_high_station2]:
					timestamp2 = float(k)
					dt_object2 = datetime.fromtimestamp(int(timestamp2))
					tuple_data_correlated= correlated_station2[k]
					if ((tuple_data_correlated[2]==events_file[events.index(event)]) and (tuple_data_correlated[1][events.index(event)] > probability) and (np.absolute(timestamp-timestamp2)<=2*sec)):
						file_h.write('\t')
						if (text_write2==False):
							text='Correlated station '+str(sort_index[1])
							file_h.write(text)
							file_h.write('\n')
							file_h.write('\t')
							text_write2=True

						file_h.write(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
						#time_stamp_station2.append(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
						file_h.write(' ')
						file_h.write(str(tuple_data_correlated))
						#tuple_station2.append(str(tuple_data_correlated))
						file_h.write('\n')


def get_Correlated_Events_INVOLCAN(data_station1, data_station2, data_station3,data_station4,data_station5, event, sec, probability, date, model):
	
	index_high=index_low=None
	index_high_station2=index_low_station2=None
	index_high_station3=index_low_station3=None
	index_high_station4=index_low_station4=None

	station_list_len= [len(list(data_station1)), len(list(data_station2)), len(list(data_station3)), len(list(data_station4)), len(list(data_station5))]
	station_list=[data_station1, data_station2, data_station3, data_station4, data_station5]
	station_reference = np.argmax(np.array(station_list_len))
	sort_index = np.argsort(station_list_len)
	print ('Printing lenght of each station detection...')
	print (len(list(data_station1)), len(list(data_station2)), len(list(data_station3)), len(list(data_station4)), len(list(data_station5)))
	#print (sort_index[-1])
	print ('Reference station: ', station_reference)
	#print (station_reference)
	station_reference=station_list[sort_index[-1]]
	whole_keys= list(station_reference)
	correlated_station=station_list[sort_index[0]]
	correlated_station2=station_list[sort_index[1]]
	correlated_station3=station_list[sort_index[2]]
	correlated_station4=station_list[sort_index[3]]
	filename='good_correlation_detected_N_stations_INVOLCAN_'+event+'(station_reference:'+str(sort_index[-1])+'_'+date+'_'+model+'_'+str(probability)+').txt'
	with open(filename, 'w') as file_h:
		whole_keys2= list(correlated_station)
		whole_keys2_float=list(map(float, list(correlated_station)))
		whole_keys3= list(correlated_station2)
		whole_keys3_float=list(map(float, list(correlated_station2)))
		whole_keys4= list(correlated_station3)
		whole_keys4_float=list(map(float, list(correlated_station3)))
		whole_keys5= list(correlated_station4)
		whole_keys5_float=list(map(float, list(correlated_station4)))
		print ('Printing len of dictionaries: ', len(whole_keys2_float), len (whole_keys3_float), len (whole_keys4_float), len(whole_keys5_float))
		text="Reference station "+ str([sort_index[-1]])
		file_h.write(text)
		file_h.write('\n')
		for j in whole_keys:
			tuple_data=station_reference[j]
			if (tuple_data[2]==events_file[events.index(event)]):
				#print (tuple_data)
				timestamp = float(j)
				dt_object = datetime.fromtimestamp(int(timestamp))
				timestamp_low= dt_object - timedelta(seconds=sec)
				timestamp_high= dt_object + timedelta(seconds=sec)
				dtimestamp_low=timestamp_low.timestamp()
				dtimestamp_high=timestamp_high.timestamp()
				try:
					if (len(whole_keys2_float)>0):
						index_high=[ n for n,i in enumerate(whole_keys2_float) if i> dtimestamp_high][0]
					if (len(whole_keys3_float)>0):
						index_high_station2=[ n for n,i in enumerate(whole_keys3_float) if i> dtimestamp_high][0]
					if (len(whole_keys4_float)>0):
						index_high_station3=[ n for n,i in enumerate(whole_keys4_float) if i> dtimestamp_high][0]
					if (len(whole_keys5_float)>0):
						index_high_station4=[ n for n,i in enumerate(whole_keys5_float) if i> dtimestamp_high][0]
				except:
					if (len(whole_keys2_float)>0):
						index_high=[ n for n,i in enumerate(whole_keys2_float) if i< dtimestamp_high][0]
					if (len(whole_keys3_float)>0):
						index_high_station2=[ n for n,i in enumerate(whole_keys3_float) if i< dtimestamp_high][0]
					if (len(whole_keys4_float)>0):
						index_high_station3=[ n for n,i in enumerate(whole_keys4_float) if i< dtimestamp_high][0]
					if (len(whole_keys5_float)>0):
						index_high_station4=[ n for n,i in enumerate(whole_keys5_float) if i< dtimestamp_high][0]
				try:
					if (len(whole_keys2_float)>0):
						index_low=[ n for n,i in enumerate(whole_keys2_float) if i> dtimestamp_low][0]
					if (len(whole_keys3_float)>0):
						index_low_station2=[ n for n,i in enumerate(whole_keys3_float) if i> dtimestamp_low][0]
					if (len(whole_keys4_float)>0):
						index_low_station3=[ n for n,i in enumerate(whole_keys4_float) if i> dtimestamp_low][0]
					if (len(whole_keys5_float)>0):
						index_low_station4=[ n for n,i in enumerate(whole_keys5_float) if i> dtimestamp_low][0]
				except:
					if (len(whole_keys2_float)>0):
						index_low=[ n for n,i in enumerate(whole_keys2_float) if i< dtimestamp_low][0]
					if (len(whole_keys3_float)>0):
						index_low_station2=[ n for n,i in enumerate(whole_keys3_float) if i< dtimestamp_low][0]
					if (len(whole_keys4_float)>0):
						index_low_station3=[ n for n,i in enumerate(whole_keys4_float) if i< dtimestamp_low][0]
					if (len(whole_keys5_float)>0):
						index_low_station4=[ n for n,i in enumerate(whole_keys5_float) if i< dtimestamp_low][0]


				if (index_high!=None and index_low!=None):
					if (index_high==index_low):
						index_high=index_high+1
				if (index_high_station2!=None and index_low_station2!=None):
					if (index_high_station2==index_low_station2):
						index_high_station2=index_high_station2+1
				if (index_high_station3!=None and index_low_station3!=None):
					if (index_high_station3==index_low_station3):
						index_high_station3=index_high_station3+1
				if (index_high_station4!=None and index_low_station4!=None):
					if (index_high_station4==index_low_station4):
						index_high_station4=index_high_station4+1
						
				file_h.write(dt_object.strftime("%Y-%m-%d %H:%M:%S"))
				file_h.write(' ')
				file_h.write(str(tuple_data))
				file_h.write('\n')
				text_write=False
				text_write2=False
				text_write3=False
				text_write4=False
				
				if(index_high!=None and index_low!=None):
					file_h.write('\t********** STATION 1 **********\n')
					for k in whole_keys2[index_low:index_high]:
						timestamp2 = float(k)
						dt_object2 = datetime.fromtimestamp(int(timestamp2))
						tuple_data_correlated= correlated_station[k]
						if ((tuple_data_correlated[2]==events_file[events.index(event)]) and (tuple_data_correlated[1][events.index(event)] > probability) and (np.absolute(timestamp-timestamp2)<=2*sec)): 
							file_h.write('\t')
							if (text_write==False):
								text='\tCorrelated found at station '+str(sort_index[0])+': '
								file_h.write('################\n')
								#file_h.write('\n')
								#file_h.write('\t')
								text_write=True
							file_h.write(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
							#time_stamp_station.append(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
							file_h.write(' ')
							file_h.write(str(tuple_data_correlated)+'\n')
							#tuple_station.append(str(tuple_data_correlated))
							file_h.write('\t################\n')
						else:
							None
							file_h.write('Found '+ tuple_data_correlated[2]+'(' + dt_object2.strftime("%Y-%m-%d %H:%M:%S")+')\n')
					file_h.write('\t********************\n')
				if(index_high_station2!=None and index_low_station2!=None):
					file_h.write('\t********** STATION 2 **********\n')
					for k in whole_keys3[index_low_station2:index_high_station2]:
						timestamp2 = float(k)
						dt_object2 = datetime.fromtimestamp(int(timestamp2))
						tuple_data_correlated= correlated_station2[k]
						if ((tuple_data_correlated[2]==events_file[events.index(event)]) and (tuple_data_correlated[1][events.index(event)] > probability) and (np.absolute(timestamp-timestamp2)<=2*sec)):
							file_h.write('\t')
							if (text_write2==False):
								text='\tCorrelated found at station '+str(sort_index[1])+': '
								file_h.write('################\n')
								#file_h.write('\n')
								#file_h.write('\t')
								text_write2=True

							file_h.write(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
							#time_stamp_station2.append(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
							file_h.write(' ')
							file_h.write(str(tuple_data_correlated)+'\n')
							#tuple_station2.append(str(tuple_data_correlated))
							file_h.write('\t################\n')
						else:
							None
							file_h.write('Found '+ tuple_data_correlated[2]+'(' + dt_object2.strftime("%Y-%m-%d %H:%M:%S")+')\n')
					file_h.write('\t********************\n')
				if(index_high_station3!=None and index_low_station3!=None):
					file_h.write('\t********** STATION 3 **********\n')
					for k in whole_keys4[index_low_station3:index_high_station3]:
						timestamp2 = float(k)
						dt_object2 = datetime.fromtimestamp(int(timestamp2))
						tuple_data_correlated= correlated_station3[k]
						if ((tuple_data_correlated[2]==events_file[events.index(event)]) and (tuple_data_correlated[1][events.index(event)] > probability) and (np.absolute(timestamp-timestamp2)<=2*sec)):
							file_h.write('\t')
							if (text_write3==False):
								text='\tCorrelated found at station '+str(sort_index[2])+': '
								file_h.write('################\n')
								#file_h.write('\n')
								#file_h.write('\t')
								text_write3=True

							file_h.write(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
							#time_stamp_station2.append(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
							file_h.write(' ')
							file_h.write(str(tuple_data_correlated)+'\n')
							#tuple_station2.append(str(tuple_data_correlated))
							file_h.write('\t################\n')
						else:
							None
							file_h.write('Found '+ tuple_data_correlated[2]+'(' + dt_object2.strftime("%Y-%m-%d %H:%M:%S")+')\n')
					file_h.write('\t********************\n')
				if(index_high_station4!=None and index_low_station4!=None):
					file_h.write('\t********** STATION 4 **********\n')	
					for k in whole_keys5[index_low_station4:index_high_station4]:
						timestamp2 = float(k)
						dt_object2 = datetime.fromtimestamp(int(timestamp2))
						tuple_data_correlated= correlated_station4[k]
						
						if ((tuple_data_correlated[2]==events_file[events.index(event)]) and (tuple_data_correlated[1][events.index(event)] > probability) and (np.absolute(timestamp-timestamp2)<=2*sec)):
							file_h.write('\t')
							if (text_write4==False):
								text='\tCorrelated found at station '+str(sort_index[3])+': '
								file_h.write('################\n')
								file_h.write(text)
								#file_h.write('\n')
								#file_h.write('\t')
								text_write4=True

							file_h.write(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
							#time_stamp_station2.append(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
							file_h.write(' ')
							file_h.write(str(tuple_data_correlated)+'\n')
							#tuple_station2.append(str(tuple_data_correlated))
							file_h.write('\t################\n')
						else:
							None
							file_h.write('\tFound '+ tuple_data_correlated[2]+'(' + dt_object2.strftime("%Y-%m-%d %H:%M:%S")+')\n')
					file_h.write('\t********************\n')


def get_Correlated_Events_INVOLCAN_real_time(event, sec, probability, date, model):
	
	index_high=[]
	index_low=[]
	station_list_len=[]
	correlated_stations=[]
	whole_keys_list=[]
	whole_keys_float_list=[]
	text_write=[]

	for i in range (len(data_ordered)):
		index_high.append(None)
		index_low.append(None)
		text_write.append(False)
	
	for data_station in data_ordered:
		station_list_len.append(len(list(data_station)))

	#station_list_len= [len(list(data_station1)), len(list(data_station2)), len(list(data_station3)), len(list(data_station4)), len(list(data_station5))]
	station_list=data_ordered
	station_reference = np.argmax(np.array(station_list_len))
	sort_index = np.argsort(station_list_len)
	print ('Printing lenght of each station detection...')
	print (station_list_len)
	#print (sort_index[-1])
	print ('Reference station: ', station_reference)
	#print (station_reference)
	station_reference=station_list[sort_index[-1]]
	whole_keys= list(station_reference)
	print (sort_index)
	
	for i in range (len(sort_index)-1):
		correlated_stations.append(station_list[sort_index[i]])
		whole_keys_list.append(list(correlated_stations[i]))
		whole_keys_float_list.append(list(map(float, list(correlated_stations[i]))))

	filename='Coherence_Analysys_'+event+'(station_reference:'+str(sort_index[-1])+'_'+date.strftime('%Y-%m-%d')+'_'+model+'_'+str(probability)+').txt'
	with open(filename, 'w+') as file_h:
		print ('Printing len of dictionaries: ', [len(whole) for whole in whole_keys_float_list])
		text="Reference station "+ str([sort_index[-1]])
		file_h.write(text)
		file_h.write('\n')
		for j in whole_keys:
			tuple_data=station_reference[j]
			if (tuple_data[2]==events_file[events.index(event)]):
				#print (tuple_data)
				timestamp = float(j)
				dt_object = datetime.fromtimestamp(int(timestamp))
				timestamp_low= dt_object - timedelta(seconds=sec)
				timestamp_high= dt_object + timedelta(seconds=sec)
				dtimestamp_low=timestamp_low.timestamp()
				dtimestamp_high=timestamp_high.timestamp()
				
				for index,element in enumerate(whole_keys_float_list):
					if (len(element)>0):
						try:
							index_high[index]=[ n for n,i in enumerate(element) if i> dtimestamp_high][0]
						except:
							index_high[index]=[ n for n,i in enumerate(element) if i< dtimestamp_high][0]


				for index,element in enumerate(whole_keys_float_list):
					if (len(element)>0):
						try:
							index_low[index]=[ n for n,i in enumerate(element) if i> dtimestamp_low][0]
						except:
							index_low[index]=[ n for n,i in enumerate(element) if i< dtimestamp_low][0]

				for i in range (len(index_high)):
					if (index_high[i]!=None and index_low[i]!=None):
						if (index_high[i]==index_low[i]):
							index_high[i]=index_high[i]+1
						
				file_h.write(dt_object.strftime("%Y-%m-%d %H:%M:%S"))
				file_h.write(' ')
				file_h.write(str(tuple_data))
				file_h.write('\n')
				
				for i in range (len(index_high)):
					if(index_high[i]!=None and index_low[i]!=None):
						file_h.write('\t********** STATION '+str (i+1)+' **********\n')
						for k in whole_keys_list[i][index_low[i]:index_high[i]]:
							timestamp2 = float(k)
							dt_object2 = datetime.fromtimestamp(int(timestamp2))
							tuple_data_correlated= correlated_stations[i][k]
							if ((tuple_data_correlated[2]==events_file[events.index(event)]) and (tuple_data_correlated[1][events.index(event)] > probability) and (np.absolute(timestamp-timestamp2)<=2*sec)): 
								file_h.write('\t')
								if (text_write[i]==False):
									text='\tCorrelated found at station '+str(sort_index[0])+': '
									file_h.write('################\n')
									#file_h.write('\n')
									#file_h.write('\t')
									text_write[i]=True
								file_h.write(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
								#time_stamp_station.append(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
								file_h.write(' ')
								file_h.write(str(tuple_data_correlated)+'\n')
								#tuple_station.append(str(tuple_data_correlated))
								file_h.write('\t################\n')
							else:
								file_h.write('Found '+ tuple_data_correlated[2]+'(' + dt_object2.strftime("%Y-%m-%d %H:%M:%S")+')\n')
						file_h.write('\t********************\n')

def get_Correlated_Events_(data_station1, data_station2, data_station3,data_station4,data_station5, event, sec, probability, date, model):
	station_list_len= [len(list(data_station1)), len(list(data_station2)), len(list(data_station3)), len(list(data_station4)), len(list(data_station5))]
	station_list=[data_station1, data_station2, data_station3, data_station4, data_station5]
	station_reference = np.argmax(np.array(station_list_len))
	sort_index = np.argsort(station_list_len)
	print (len(list(data_station1)), len(list(data_station2)), len(list(data_station3)), len(list(data_station4)), len(list(data_station5)))
	print (sort_index[-1])
	print (station_reference)
	station_reference=station_list[sort_index[-1]]
	whole_keys= list(station_reference)
	correlated_station=station_list[sort_index[0]]
	correlated_station2=station_list[sort_index[1]]
	correlated_station3=station_list[sort_index[2]]
	correlated_station4=station_list[sort_index[3]]
	filename='good_correlation_detected_N_stations_INVOLCAN(station_reference:'+str(sort_index[-1])+date+model+str(probability)+').txt'
	with open(filename, 'w') as file_h:
		whole_keys2= list(correlated_station)
		whole_keys2_float=list(map(float, list(correlated_station)))
		whole_keys3= list(correlated_station2)
		whole_keys3_float=list(map(float, list(correlated_station2)))
		whole_keys4= list(correlated_station3)
		whole_keys4_float=list(map(float, list(correlated_station3)))
		whole_keys5= list(correlated_station4)
		whole_keys5_float=list(map(float, list(correlated_station4)))
		text="Reference station "+ str([sort_index[-1]])
		file_h.write(text)
		file_h.write('\n')
		for j in whole_keys:
			tuple_data=station_reference[j]
			if (tuple_data[2]==events_file[events.index(event)]):
				timestamp = float(j)
				dt_object = datetime.fromtimestamp(int(timestamp))
				timestamp_low= dt_object - timedelta(seconds=sec)
				timestamp_high= dt_object + timedelta(seconds=sec)
				dtimestamp_low=timestamp_low.timestamp()
				dtimestamp_high=timestamp_high.timestamp()
				try:
					index_high=[ n for n,i in enumerate(whole_keys2_float) if i> dtimestamp_high][0]
					index_high_station2=[ n for n,i in enumerate(whole_keys3_float) if i> dtimestamp_high][0]
					index_high_station3=[ n for n,i in enumerate(whole_keys4_float) if i> dtimestamp_high][0]
					index_high_station4=[ n for n,i in enumerate(whole_keys5_float) if i> dtimestamp_high][0]
				except:
					index_high=[ n for n,i in enumerate(whole_keys2_float) if i< dtimestamp_high][0]
					index_high_station2=[ n for n,i in enumerate(whole_keys3_float) if i< dtimestamp_high][0]
					index_high_station3=[ n for n,i in enumerate(whole_keys4_float) if i< dtimestamp_high][0]
					index_high_station4=[ n for n,i in enumerate(whole_keys5_float) if i< dtimestamp_high][0]
				try:
					index_low=[ n for n,i in enumerate(whole_keys2_float) if i> dtimestamp_low][0]
					index_low_station2=[ n for n,i in enumerate(whole_keys3_float) if i> dtimestamp_low][0]
					index_low_station3=[ n for n,i in enumerate(whole_keys4_float) if i> dtimestamp_low][0]
					index_low_station4=[ n for n,i in enumerate(whole_keys5_float) if i> dtimestamp_low][0]
				except:
					index_low=[ n for n,i in enumerate(whole_keys2_float) if i< dtimestamp_low][0]
					index_low_station2=[ n for n,i in enumerate(whole_keys3_float) if i< dtimestamp_low][0]
					index_low_station3=[ n for n,i in enumerate(whole_keys4_float) if i< dtimestamp_low][0]
					index_low_station4=[ n for n,i in enumerate(whole_keys5_float) if i< dtimestamp_low][0]
					
				if (index_high==index_low):
					index_high=index_high+1
				if (index_high_station2==index_low_station2):
					index_high_station2=index_high_station2+1
				if (index_high_station3==index_low_station3):
					index_high_station3=index_high_station3+1
				if (index_high_station4==index_low_station4):
					index_high_station4=index_high_station4+1
						
				file_h.write(dt_object.strftime("%Y-%m-%d %H:%M:%S"))
				file_h.write(' ')
				file_h.write(str(tuple_data))
				file_h.write('\n')
				text_write=False
				text_write2=False
				text_write3=False
				text_write4=False
				for k in whole_keys2[index_low:index_high]:
					timestamp2 = float(k)
					dt_object2 = datetime.fromtimestamp(int(timestamp2))
					tuple_data_correlated= correlated_station[k]
					if ((tuple_data_correlated[2]==events_file[events.index(event)]) and (tuple_data_correlated[1][events.index(event)] > probability) and (np.absolute(timestamp-timestamp2)<=2*sec)): 
						file_h.write('\t')
						if (text_write==False):
							text='Correlated station '+str(sort_index[0])
							file_h.write(text)
							file_h.write('\n')
							file_h.write('\t')
							text_write=True
						file_h.write(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
						#time_stamp_station.append(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
						file_h.write(' ')
						file_h.write(str(tuple_data_correlated))
						#tuple_station.append(str(tuple_data_correlated))
						file_h.write('\n')
				for k in whole_keys3[index_low_station2:index_high_station2]:
					timestamp2 = float(k)
					dt_object2 = datetime.fromtimestamp(int(timestamp2))
					tuple_data_correlated= correlated_station2[k]
					if ((tuple_data_correlated[2]==events_file[events.index(event)]) and (tuple_data_correlated[1][events.index(event)] > probability) and (np.absolute(timestamp-timestamp2)<=2*sec)):
						file_h.write('\t')
						if (text_write2==False):
							text='Correlated station '+str(sort_index[1])
							file_h.write(text)
							file_h.write('\n')
							file_h.write('\t')
							text_write2=True

						file_h.write(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
						#time_stamp_station2.append(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
						file_h.write(' ')
						file_h.write(str(tuple_data_correlated))
						#tuple_station2.append(str(tuple_data_correlated))
						file_h.write('\n')
				for k in whole_keys4[index_low_station3:index_high_station3]:
					timestamp2 = float(k)
					dt_object2 = datetime.fromtimestamp(int(timestamp2))
					tuple_data_correlated= correlated_station3[k]
					if ((tuple_data_correlated[2]==events_file[events.index(event)]) and (tuple_data_correlated[1][events.index(event)] > probability) and (np.absolute(timestamp-timestamp2)<=2*sec)):
						file_h.write('\t')
						if (text_write3==False):
							text='Correlated station '+str(sort_index[2])
							file_h.write(text)
							file_h.write('\n')
							file_h.write('\t')
							text_write3=True

						file_h.write(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
						#time_stamp_station2.append(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
						file_h.write(' ')
						file_h.write(str(tuple_data_correlated))
						#tuple_station2.append(str(tuple_data_correlated))
						file_h.write('\n')
						
				for k in whole_keys5[index_low_station4:index_high_station4]:
					timestamp2 = float(k)
					dt_object2 = datetime.fromtimestamp(int(timestamp2))
					tuple_data_correlated= correlated_station4[k]
					if ((tuple_data_correlated[2]==events_file[events.index(event)]) and (tuple_data_correlated[1][events.index(event)] > probability) and (np.absolute(timestamp-timestamp2)<=2*sec)):
						file_h.write('\t')
						if (text_write4==False):
							text='Correlated station '+str(sort_index[3])
							file_h.write(text)
							file_h.write('\n')
							file_h.write('\t')
							text_write4=True

						file_h.write(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
						#time_stamp_station2.append(dt_object2.strftime("%Y-%m-%d %H:%M:%S"))
						file_h.write(' ')
						file_h.write(str(tuple_data_correlated))
						#tuple_station2.append(str(tuple_data_correlated))
						file_h.write('\n')
						
						
						
def Run_All_Month_5_Stations(event, month, year):

	root= '/home/female2020/Software/RNN_Other_Volcanoes/Test_Probabilidad_BZ'
	folder='02_2017_12'
	print('Starting processing for: ', event)
	event='EQ'
	event=event
	days=monthrange(year, month)
	if (month<10):
		str_month='0'+str(month)
	else:
		str_month=str(month)
		
	for i in range (days[1]):
		if (i+1<10):
			day='0'+str(i+1)
		else:
			day= str(i+1)
			
		str_year=str(year)
		date=str_year[len(str_year)-2:]+str_month+day
		date2=str_year+'_'+str_month
		
		path_sta01= root+'01_'+date2+'/Prueba_Bezymyanny_BZ01_'+str_month+'_bz01'+date+'000000_BHZ_probabilities_RNN.txt'
		path_sta06= root+'06_'+date2+'/Prueba_Bezymyanny_BZ06_'+str_month+'_bz06'+date+'000000_BHZ_probabilities_RNN.txt'
		path_sta10= root+'10_'+date2+'/Prueba_Bezymyanny_BZ10_'+str_month+'_bz10'+date+'000000_BHZ_probabilities_RNN.txt'
		path_sta02= root+'02_'+date2+'/Prueba_Bezymyanny_BZ02_'+str_month+'_bz02'+date+'000000_BHZ_probabilities_RNN.txt'
		path_sta08= root+'08_'+date2+'/Prueba_Bezymyanny_BZ08_'+str_month+'_bz08'+date+'000000_BHZ_probabilities_RNN.txt'
		files=[False, False, False, False, False]
		
		if (os.path.exists(path_sta01)):
			Reading_data(root+'01_'+date2+'/Prueba_Bezymyanny_BZ01_'+str_month+'_bz01'+date+'000000_BHZ_probabilities_RNN.txt','1', year,month,i+1, 'rnn',event)
			files[0]=True
		if (os.path.exists(path_sta06)):
			Reading_data(root+'06_'+date2+'/Prueba_Bezymyanny_BZ06_'+str_month+'_bz06'+date+'000000_BHZ_probabilities_RNN.txt','2', year,month,i+1, 'rnn',event)
			files[1]=True
		if(os.path.exists(path_sta10)):
			Reading_data(root+'10_'+date2+'/Prueba_Bezymyanny_BZ10_'+str_month+'_bz10'+date+'000000_BHZ_probabilities_RNN.txt','3', year,month,i+1, 'rnn',event)
			files[2]=True
		if(os.path.exists(path_sta02)):
			Reading_data(root+'02_'+date2+'/Prueba_Bezymyanny_BZ02_'+str_month+'_bz02'+date+'000000_BHZ_probabilities_RNN.txt','4', year,month,i+1, 'rnn',event)
			files[3]=True
		if(os.path.exists(path_sta08)):
			Reading_data(root+'08_'+date2+'/Prueba_Bezymyanny_BZ08_'+str_month+'_bz08'+date+'000000_BHZ_probabilities_RNN.txt','5', year,month,i+1, 'rnn',event)
			files[4]=True
		
		
		#Reading_data(root+'01_'+date2+'/Prueba_Bezymyanny_BZ01_'+str_month+'_bz01'+date+'000000_BHZ_probabilities_RNN.txt','1', year,month,i+1, 'rnn',event)
		#Reading_data(root+'06_'+date2+'/Prueba_Bezymyanny_BZ06_'+str_month+'_bz06'+date+'000000_BHZ_probabilities_RNN.txt','2', year,month,i+1, 'rnn',event)
		#Reading_data(root+'10_'+date2+'/Prueba_Bezymyanny_BZ10_'+str_month+'_bz10'+date+'000000_BHZ_probabilities_RNN.txt','3', year,month,i+1, 'rnn',event)
		#Reading_data(root+'02_'+date2+'/Prueba_Bezymyanny_BZ02_'+str_month+'_bz02'+date+'000000_BHZ_probabilities_RNN.txt','4', year,month,i+1, 'rnn',event)
		#Reading_data(root+'08_'+date2+'/Prueba_Bezymyanny_BZ08_'+str_month+'_bz08'+date+'000000_BHZ_probabilities_RNN.txt','5', year,month,i+1, 'rnn',event)

		if (not False in files):
			data_ordered = collections.OrderedDict(sorted(rnn_equ_station1.items()))
			data_ordered2 = collections.OrderedDict(sorted(rnn_equ_station2.items()))
			data_ordered3 = collections.OrderedDict(sorted(rnn_equ_station3.items()))
			data_ordered4 = collections.OrderedDict(sorted(rnn_equ_station4.items()))
			data_ordered5 = collections.OrderedDict(sorted(rnn_equ_station5.items()))
	
			#2get_Correlated_Events_previousVersion(data_ordered, data_ordered2, data_ordered3, event, 10, 30)
			get_Correlated_Events_(data_ordered, data_ordered2, data_ordered3, data_ordered4, data_ordered5, event, 10, 10, date, 'rnn')
		else:
			print ('Problems reading: ', date)
		rnn_equ_station1.clear()
		rnn_equ_station2.clear()
		rnn_equ_station3.clear()
		rnn_equ_station4.clear()
		rnn_equ_station5.clear()


def Run_All_Month_5_Stations_TCN(event, month, year):

	root= '/home/female2020/Software/CodeTCN/Test_Probabilidad_BZ'
	print('Starting processing for: ', event)
	event='EQ'
	event=event
	days=monthrange(year, month)
	if (month<10):
		str_month='0'+str(month)
	else:
		str_month=str(month)
		
	for i in range (days[1]):
		if (i+1<10):
			day='0'+str(i+1)
		else:
			day= str(i+1)
			
		str_year=str(year)
		date=str_year[len(str_year)-2:]+str_month+day
		date2=str_year+'_'+str_month
		
		path_sta01= root+'01_'+date2+'/Prueba_Bezymyanny_BZ01_'+str_month+'_bz01'+date+'000000_BHZ_probabilities_TCN.txt'
		path_sta06= root+'06_'+date2+'/Prueba_Bezymyanny_BZ06_'+str_month+'_bz06'+date+'000000_BHZ_probabilities_TCN.txt'
		path_sta10= root+'10_'+date2+'/Prueba_Bezymyanny_BZ10_'+str_month+'_bz10'+date+'000000_BHZ_probabilities_TCN.txt'
		path_sta02= root+'02_'+date2+'/Prueba_Bezymyanny_BZ02_'+str_month+'_bz02'+date+'000000_BHZ_probabilities_TCN.txt'
		path_sta08= root+'08_'+date2+'/Prueba_Bezymyanny_BZ08_'+str_month+'_bz08'+date+'000000_BHZ_probabilities_TCN.txt'
		files=[False, False, False, False, False]
		
		if (os.path.exists(path_sta01)):
			Reading_data(root+'01_'+date2+'/Prueba_Bezymyanny_BZ01_'+str_month+'_bz01'+date+'000000_BHZ_probabilities_TCN.txt','1', year,month,i+1, 'tcn',event)
			files[0]=True
		if (os.path.exists(path_sta06)):
			Reading_data(root+'06_'+date2+'/Prueba_Bezymyanny_BZ06_'+str_month+'_bz06'+date+'000000_BHZ_probabilities_TCN.txt','2', year,month,i+1, 'tcn',event)
			files[1]=True
		if(os.path.exists(path_sta10)):
			Reading_data(root+'10_'+date2+'/Prueba_Bezymyanny_BZ10_'+str_month+'_bz10'+date+'000000_BHZ_probabilities_TCN.txt','3', year,month,i+1, 'tcn',event)
			files[2]=True
		if(os.path.exists(path_sta02)):
			Reading_data(root+'02_'+date2+'/Prueba_Bezymyanny_BZ02_'+str_month+'_bz02'+date+'000000_BHZ_probabilities_TCN.txt','4', year,month,i+1, 'tcn',event)
			files[3]=True
		if(os.path.exists(path_sta08)):
			Reading_data(root+'08_'+date2+'/Prueba_Bezymyanny_BZ08_'+str_month+'_bz08'+date+'000000_BHZ_probabilities_TCN.txt','5', year,month,i+1, 'tcn',event)
			files[4]=True
		
		
		#Reading_data(root+'01_'+date2+'/Prueba_Bezymyanny_BZ01_'+str_month+'_bz01'+date+'000000_BHZ_probabilities_RNN.txt','1', year,month,i+1, 'rnn',event)
		#Reading_data(root+'06_'+date2+'/Prueba_Bezymyanny_BZ06_'+str_month+'_bz06'+date+'000000_BHZ_probabilities_RNN.txt','2', year,month,i+1, 'rnn',event)
		#Reading_data(root+'10_'+date2+'/Prueba_Bezymyanny_BZ10_'+str_month+'_bz10'+date+'000000_BHZ_probabilities_RNN.txt','3', year,month,i+1, 'rnn',event)
		#Reading_data(root+'02_'+date2+'/Prueba_Bezymyanny_BZ02_'+str_month+'_bz02'+date+'000000_BHZ_probabilities_RNN.txt','4', year,month,i+1, 'rnn',event)
		#Reading_data(root+'08_'+date2+'/Prueba_Bezymyanny_BZ08_'+str_month+'_bz08'+date+'000000_BHZ_probabilities_RNN.txt','5', year,month,i+1, 'rnn',event)

		if (not False in files):
			data_ordered = collections.OrderedDict(sorted(tcn_equ_station1.items()))
			data_ordered2 = collections.OrderedDict(sorted(tcn_equ_station2.items()))
			data_ordered3 = collections.OrderedDict(sorted(tcn_equ_station3.items()))
			data_ordered4 = collections.OrderedDict(sorted(tcn_equ_station4.items()))
			data_ordered5 = collections.OrderedDict(sorted(tcn_equ_station5.items()))
	
			#2get_Correlated_Events_previousVersion(data_ordered, data_ordered2, data_ordered3, event, 10, 30)
			get_Correlated_Events_(data_ordered, data_ordered2, data_ordered3, data_ordered4, data_ordered5, event, 10, 10, date, 'tcn')
		else:
			print ('Problems reading: ', date)
		tcn_equ_station1.clear()
		tcn_equ_station2.clear()
		tcn_equ_station3.clear()
		tcn_equ_station4.clear()
		tcn_equ_station5.clear()


if __name__ == '__main__':


	#This script has 2 use mode:


	print('Starting processing')
	event='EQ'
	#Run_All_Month_5_Stations(event, 8, 2017)
	Run_All_Month_5_Stations_TCN(event, 8, 2017)
	
	sys.exit()
	#Reading_data('Prueba_Bezymyanny_BZ01_12_bz01171218000000_BHZ_probabilities_RNN.txt','1', 2017,12,18, 'rnn',event)
	#Reading_data('Prueba_Bezymyanny_BZ10_12_bz10171218000000_BHZ_probabilities_RNN.txt','3', 2017,12,18, 'rnn',event)
	#Reading_data('Prueba_Bezymyanny_BZ06_12_bz06171218000000_BHZ_probabilities_RNN.txt','2', 2017,12,18, 'rnn',event)

	# Prueba para 2017 - Agosto - dia 06
    # Prueba para 2017 Octubre dia 10
	#Reading_data('Prueba_Bezymyanny_BZ01_08_bz01170810000000_BHZ_probabilities_RNN.txt','1', 2017,8,10, 'rnn',event)
	#Reading_data('Prueba_Bezymyanny_BZ06_08_bz06170810000000_BHZ_probabilities_RNN.txt','2', 2017,8,10, 'rnn',event)
	#Reading_data('Prueba_Bezymyanny_BZ10_08_bz10170810000000_BHZ_probabilities_RNN.txt','3', 2017,8,10, 'rnn',event)

    # Pueba para 2017 Octubre dia 10
	Reading_data('Prueba_Bezymyanny_BZ01_08_bz01170810000000_BHZ_probabilities_RNN.txt','1', 2017,8,10, 'rnn',event)
	Reading_data('Prueba_Bezymyanny_BZ06_08_bz06170810000000_BHZ_probabilities_RNN.txt','2', 2017,8,10, 'rnn',event)
	Reading_data('Prueba_Bezymyanny_BZ10_08_bz10170810000000_BHZ_probabilities_RNN.txt','3', 2017,8,10, 'rnn',event)
	Reading_data('Prueba_Bezymyanny_BZ02_08_bz02170810000000_BHZ_probabilities_RNN.txt','4', 2017,8,10, 'rnn',event)
	Reading_data('Prueba_Bezymyanny_BZ08_08_bz08170810000000_BHZ_probabilities_RNN.txt','5', 2017,8,10, 'rnn',event)

	data_ordered = collections.OrderedDict(sorted(rnn_equ_station1.items()))
	data_ordered2 = collections.OrderedDict(sorted(rnn_equ_station2.items()))
	data_ordered3 = collections.OrderedDict(sorted(rnn_equ_station3.items()))
	data_ordered4 = collections.OrderedDict(sorted(rnn_equ_station4.items()))
	data_ordered5 = collections.OrderedDict(sorted(rnn_equ_station5.items()))
	
	#2get_Correlated_Events_previousVersion(data_ordered, data_ordered2, data_ordered3, event, 10, 30)
	get_Correlated_Events_(data_ordered, data_ordered2, data_ordered3, data_ordered4, data_ordered5, event, 10, 10)

	#Correlate_Events(data_ordered, data_ordered2, data_ordered3, event, 10)

	sys.exit()
	#Reading_data('Prueba_Bezymyanny_BZ01_12_bz01171218000000_BHZ_probabilities_RNN.txt','1', 2018,12,17, 'rnn','EQ')
	#data_ordered2 = collections.OrderedDict(sorted(rnn_equ_station1.items()))

	Histogram_plotting('station1', data_ordered, 'duration')
	Histogram_plotting('station1', data_ordered, 'probability','hours', event, 'rnn')
	Histogram_plotting('station1', data_ordered, 'probability','day', event,'rnn')
	Histogram_plotting('station1', data_ordered, 'duration','hours', event, 'rnn')
	Histogram_plotting('station1', data_ordered, 'duration','day', event,'rnn')

	sys.exit()

