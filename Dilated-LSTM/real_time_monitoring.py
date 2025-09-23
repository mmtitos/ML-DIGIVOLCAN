from Auxiliar_Functions import *
from coherence_analysis import *
import sys
import torch
from torchsummary import summary
import torch.nn as nn
from datetime import datetime
import time
import sched


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

    task1=scheduler.enter(1, 1, real_time_monitoring, arguments)
    task2=scheduler.enter(frequency, 1, repeat_task, (frequency,arguments))

    #if(time.gmtime().tm_hour <= 10 and time.gmtime().tm_min >=42):
    #    scheduler.cancel(task2)
    #    exit(0)




# Main program. We can select between use a classical LSTM or a dilated version using 3 hidden layers
#We can also select if we want to train a new networks or just test using the trained version 

if __name__ == '__main__':


    #Read config file to get configuration

    n_layers,n_hidden, num_station, station_names,coherence_event, station_path, frequency, model_path, real_time, dates, model=read_Config_File()

    print ('starting...')

    device = torch.device("cuda" if use_cuda else "cpu")
    print (device)

    expert_by_station=[]

    print("==> Building a dRNN with %s cells" %cell_type)

    ml_model = Classifier(48, n_hidden, n_layers, n_classes, cell_type=cell_type)

    if (n_layers==3):
        ml_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    if (n_layers==1):
        ml_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    ml_model.eval()
    summary(ml_model)
    if use_cuda:
        ml_model.cuda()
    
    print ('Starting near real time monitoring for stations: ', station_names)
    print ('Reading data and  creating features vectors...')

    if (eval(real_time)):

        files = [os.path.exists(path) for path in station_path[0]]
        if (not False in files):
            repeat_task(frequency, (station_path[0], station_names, coherence_event, dates[0], model,ml_model))
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
                real_time_monitoring(ana_day, station_names, coherence_event, dates[index], model ,ml_model)
        exit(0)
