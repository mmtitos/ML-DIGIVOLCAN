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


# Main program. We can select between use a classical LSTM or a dilated version using 3 hidden layers
#We can also select if we want to train a new networks or just test using the trained version 

if __name__ == '__main__':


    #Read config file to get configuration

    if (len(sys.argv) >1):
        aux_file=sys.argv[1]
    else:
        aux_file='../config_real_time.ini'

    n_filter, num_station, weakly, min_chunk, prob_threshold, station_names,coherence_event, station_path, frequency, model_paths, real_time, dates, model =read_Config_File_TCN_weakly(aux_file)

    print("==> Building a TCN with %s filters per layer" %n_filter)

    expert_by_station=[]

    for i in range (num_station):

        ml_model = tnf.keras.models.load_model(model_paths[i])
        expert_by_station.append(ml_model)

    expert_by_station[0].summary()

    print ('Starting weakly retraining for stations: ', station_names)

    if (eval(weakly)):

        print ('Starting RE-TRAINING process using a weakly supervised approach...')
        for index, ana_day in enumerate (station_path):
            print (ana_day, station_names, coherence_event, dates[index])
            files = [os.path.exists(path) for path in station_path[0]]
            if (not False in files):
                weakly_reatrain_parallel(ana_day, min_chunk ,prob_threshold, station_names, dates[index], model,expert_by_station)
            else:
                print ('File to get raw data does not exits...')
                print ('aborting...')
                sys.exit(0)
        
    else:
        print ('Please, change weakly mode in the config file to process with weakly supervised retraining.')
        print ('Aborting...')
        sys.exit(0)

    sys.exit()


    for i in range (num_station):

        ml_model = Classifier(48, n_hidden, n_layers, n_classes, cell_type=cell_type)
        print (model_paths[i])

        ml_model.load_state_dict(torch.load(model_paths[i], map_location=torch.device('cpu')))
        if use_cuda:
            ml_model.cuda()
        expert_by_station.append(ml_model)
    
    #expert_by_station[0].eval()
    summary(expert_by_station[0])


    if (eval(weakly)==True):
        print ('Starting RE-TRAINING process using a weakly supervised approach...')

        for index, ana_day in enumerate (station_path):
            print (ana_day, station_names, coherence_event, dates[index])
            files = [os.path.exists(path) for path in ana_day]
            if (not False in files):
                weakly_reatrain_parallel(ana_day, min_chunk ,prob_threshold, station_names, dates[index], model,expert_by_station)
    else:

        print ('Please, change weakly mode in the config file to process with weakly supervised retraining.')
        print ('Aborting...')
        sys.exit(0)

    
    sys.exit()
