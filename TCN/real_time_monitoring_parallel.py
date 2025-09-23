import numpy as np
import sys
import tensorflow.keras as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Sequential
from Auxiliar_Functions_TCN import *
from tcn import TCN, tcn_full_summary
import tensorflow as tnf
import os
import time
import sched

scheduler = sched.scheduler(time.time, time.sleep)

def repeat_task(frequency,arguments):

    task1=scheduler.enter(1, 1, real_time_monitoring_TCN_parallel, arguments)
    task2=scheduler.enter(frequency, 1, repeat_task, (frequency,arguments))

    #if(time.gmtime().tm_hour <= 10 and time.gmtime().tm_min >=42):
    #    scheduler.cancel(task2)
    #    exit(0)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if __name__ == '__main__':
    
    print ('starting...')

    if (len(sys.argv) >1):
        aux_file=sys.argv[1]
    else:
        aux_file='../config_real_time.ini'

    n_filter, num_station, station_names,coherence_event, station_path, frequency, model_path, real_time, dates, model=read_Config_File_TCN_parallel(aux_file)

    print("==> Building a TCN with %s filters per layer" %n_filter)

    expert_by_station=[]

    for i in range (num_station):

        ml_model = tnf.keras.models.load_model(model_path[i])
        expert_by_station.append(ml_model)

    expert_by_station[0].summary()

    print ('Starting near real time monitoring for stations: ', station_names)
    print ('Reading data and  creating features vectors...')

    if (eval(real_time)):

        files = [os.path.exists(path) for path in station_path[0]]
        if (not False in files):
            repeat_task(frequency, (station_path[0], station_names, coherence_event, dates[0], model, expert_by_station))
            scheduler.run()
        else:
            print ('File to ge real time raw data does not exits...')
            print ('aborting...')
            sys.exit(0)
        
    else:
        for index, ana_day in enumerate (station_path):
            print (ana_day, station_names, coherence_event, dates[index])
            files = [os.path.exists(path) for path in ana_day]
            if (not False in files):
                real_time_monitoring_TCN_parallel(ana_day, station_names, coherence_event, dates[index], model,expert_by_station)
        exit(0)

    sys.exit()





