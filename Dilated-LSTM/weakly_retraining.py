from Auxiliar_Functions import *
from coherence_analysis import *
import sys
import torch
from torchsummary import summary
import configparser
import torch.nn as nn

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


# Main program. We can select between use a classical LSTM or a dilated version using 3 hidden layers
#We can also select if we want to train a new networks or just test using the trained version 

if __name__ == '__main__':


    #Read config file to get configuration

    if (len(sys.argv) >1):
        aux_file=sys.argv[1]
    else:
        aux_file='../config_real_time.ini'

    n_layers,n_hidden, num_station, weakly, min_chunk, prob_threshold, station_names,coherence_event, station_path, frequency, model_paths, real_time, dates, model =read_Config_File_Weakly(aux_file)


    print ('starting...')

    device = torch.device("cuda" if use_cuda else "cpu")
    print (device)

    expert_by_station=[]

    print("==> Building a dRNN with %s cells" %cell_type)

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
                print ('File to get raw data does not exits...')
                print ('aborting...')
                sys.exit(0)
    else:

        print ('Please, change weakly mode in the config file to process with weakly supervised retraining.')
        print ('Aborting...')
        sys.exit(0)

    sys.exit()
