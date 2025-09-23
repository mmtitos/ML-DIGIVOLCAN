from Auxiliar_Functions import *
from coherence_analysis import *
import sys
import torch
from torchsummary import summary
import torch.nn as nn
from datetime import datetime
import time
import sched
import multiprocessing


print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION')

print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
#print('__Devices')
#print('Active CUDA Device: GPU', torch.cuda.current_device())

#print ('Available devices ', torch.cuda.device_count())
#print ('Current cuda device ', torch.cuda.current_device())
#print(torch.cuda.device_count())
#print (torch.cuda.get_device_capability(0))
#print(torch.cuda.get_device_capability(1))
#mydevice=torch.device('cuda:1')
#print('Active CUDA Device: GPU', torch.cuda.current_device())
use_cuda = torch.cuda.is_available()

# some global execution parameters --->NOT CHANGE
n_classes = 5
cell_type = "LSTM"
batch_size=1
learning_rate = 1.0e-3
training_iters = 30000
training_iters = 500
display_step = 25
display_step = 1

scheduler = sched.scheduler(time.time, time.sleep)

def repeat_task(frequency,arguments):

    task1=scheduler.enter(1, 1, real_time_monitoring_parallel, arguments)
    task2=scheduler.enter(frequency, 1, repeat_task, (frequency,arguments))

    #if(time.gmtime().tm_hour <= 10 and time.gmtime().tm_min >=42):
    #    scheduler.cancel(task2)
    #    exit(0)




#def process(station_path, station_names, coherence_event, dates, ml_model):


#p = multiprocessing.Pool()
#for f in glob.glob(folder+"*.csv"):
    # launch a process for each file (ish).
    # The result will be approximately one process per CPU core available.
#    p.apply_async(process, [f]) 

#p.close()
#p.join()



# Main program. We can select between use a classical LSTM or a dilated version using 3 hidden layers
#We can also select if we want to train a new networks or just test using the trained version 

if __name__ == '__main__':


    #Read config file to get configuration

    if (len(sys.argv) >1):
        aux_file=sys.argv[1]
    else:
        aux_file='../config_real_time.ini'
    
    n_layers,n_hidden, num_station, station_names,coherence_event, station_path, frequency, model_paths, real_time, dates, model =read_Config_File_parallel(aux_file)

    print ('starting...')

    device = torch.device("cuda" if use_cuda else "cpu")
    print (device)

    expert_by_station=[]

    print("==> Building a dRNN with %s cells" %cell_type)

    for i in range (num_station):

        ml_model = Classifier(48, n_hidden, n_layers, n_classes, cell_type=cell_type)
        print (model_paths[i])

        if (n_layers==3):
            ml_model.load_state_dict(torch.load(model_paths[i], map_location=torch.device('cpu')))
        if (n_layers==1):
            ml_model.load_state_dict(torch.load(model_paths[i], map_location=torch.device('cpu')))
        expert_by_station.append(ml_model)
    
    summary (expert_by_station[0])

    #ml_model.eval()
    #summary(ml_model)
    #if use_cuda:
        #ml_model.cuda()
    
    print ('Starting near real time monitoring for stations: ', station_names)
    print ('Reading data and  creating features vectors...')

    if (eval(real_time)):

        files = [os.path.exists(path) for path in station_path[0]]
        if (not False in files):
            repeat_task(frequency, (station_path[0], station_names, coherence_event, dates[0],model, expert_by_station))
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
                real_time_monitoring_parallel(ana_day, station_names, coherence_event, dates[index], model,expert_by_station)
        exit(0)
