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
print('__Devices')
print('Active CUDA Device: GPU', torch.cuda.current_device())

print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.device_count())
print (torch.cuda.get_device_capability(0))
#print(torch.cuda.get_device_capability(1))
mydevice=torch.device('cuda:1')
print('Active CUDA Device: GPU', torch.cuda.current_device())
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

    config = configparser.ConfigParser()
    config.read('config.ini')
    mode=config['DEFAULT']['mode']
    n_layers=int(config[mode]['n_layers'])
    min_chunk= int(config[mode]['min_chunk'])

    print ('starting...')

    device = torch.device("cuda" if use_cuda else "cpu")
    print (device)

    if (n_layers==1):
        n_hidden = 210
    elif(n_layers==3):
        n_hidden = 50
    else:
        print ('Currently, only a classical LSTM with one hidden layer or a Dilated LSTM with 3 hidden layers available')
        print ('Exiting...')
        exit(0)

    print("==> Building a dRNN with %s cells" %cell_type)

    ml_model = Classifier(48, n_hidden, n_layers, n_classes, cell_type=cell_type)

    if (n_layers==3):
        ml_model.load_state_dict(torch.load('/home/manuel/Documents/Dilated_RNN/pytorch-dilated-rnn-deception/best_model_3layers'))
    if (n_layers==1):
        ml_model.load_state_dict(torch.load('/home/manuel/Documents/Dilated_RNN/pytorch-dilated-rnn-deception/best_model_1l_P0'))

    ml_model.eval()
    summary(ml_model)
    if use_cuda:
        ml_model.cuda()

    if (mode=='weakly'):
        print ('Starting RE-TRAINING process using a weakly supervised approach...')

        print ('Reading data and  creating features vectors from PLPI and PPMA stations...')
        labels_plpi,dataset_plpi= compute_features('../2021_HHZ/C7.PLPI..HHZ.D.2021.255',min_chunk,norm_var=True, norm_colum=False)
        labels_ppma,dataset_ppma= compute_features('../2021_HHZ/PPMA/C7.PPMA..HHZ.D.2021.255',min_chunk,norm_var=True, norm_colum=False)

        print ('creating tensors to be used in GPU devices ...')

        dataset_plpi_tensor= Create_List_Tensor (dataset_plpi)
        dataset_ppma_tensor= Create_List_Tensor (dataset_ppma)

        print ('Creating data set using only best events...')

        
        datatraining_plpi_confidence, label_plpi_confidence, chunk_selected_plpi=create_Data_Base_Using_Grammar(ml_model, dataset_plpi_tensor,labels_plpi , 0.55)
        data_training_ppma_confidence, label_ppma_confidence, chunk_selected_ppma=create_Data_Base_Using_Grammar(ml_model, dataset_ppma_tensor,labels_ppma , 0.55)

        print ('Re-training using weakly supervised data set...')

        Retraining_model_by_station(ml_model, datatraining_plpi_confidence, label_plpi_confidence, dataset_plpi_tensor, labels_plpi, "Prueba_Lapalma_involcan_plpi_after_weakly.txt")
        Retraining_model_by_station(ml_model, data_training_ppma_confidence, label_ppma_confidence, dataset_ppma_tensor, labels_ppma, "Prueba_Lapalma_involcan_ppma_after_weakly.txt")
        
    
    else:

        print ('Starting RE-TRAINING process using a coherence analysis method...')
        print ('Reading data and  creating features vectors from PLPI and PPMA stations...')

        
    
        labels_plpi,dataset_plpi= compute_features('../2021_HHZ/C7.PLPI..HHZ.D.2021.255',min_chunk,norm_var=True, norm_colum=False)
        labels_ppma,dataset_ppma= compute_features('../2021_HHZ/PPMA/C7.PPMA..HHZ.D.2021.255',min_chunk,norm_var=True, norm_colum=False)

        print ('creating tensors to be used in GPU devices ...')

        dataset_plpi_tensor= Create_List_Tensor (dataset_plpi)
        dataset_ppma_tensor= Create_List_Tensor (dataset_ppma)

        # Get detections for PLPI and PPMA stations
    
        print ('Geting recognitions, probabilities and activations for PLPI and PPMA stations...')

        detection_plpi, probabilities_plpi, activation_plpi= get_recognition(dataset_plpi_tensor,labels_plpi,ml_model)
        detection_ppma, probabilities_ppma, activation_ppma= get_recognition(dataset_ppma_tensor,labels_ppma,ml_model)

        print ('Creating pseudo catalog files...')

        Find_instant_Events_probability(detection_plpi,probabilities_plpi ,"Prueba_Lapalma_involcan_plpi_coherence.txt")
        Find_instant_Events_probability(detection_ppma,probabilities_ppma ,"Prueba_Lapalma_involcan_ppma_coherence.txt")
        
        
        event='EQ'
        #event='LPE'
        print('Starting processing the coherence analysis for: '+ event)


        Reading_data('Prueba_Lapalma_involcan_plpi_coherence.txt','1', 2021,9,12, 'rnn',event)
        Reading_data('Prueba_Lapalma_involcan_ppma_coherence.txt','2', 2021,9,12, 'rnn',event)
        #Reading_data('Prueba_Bezymyanny_BZ10_08_bz10170810000000_BHZ_probabilities_RNN.txt','3', 2017,8,10, 'rnn',event)
        #Reading_data('Prueba_Bezymyanny_BZ02_08_bz02170810000000_BHZ_probabilities_RNN.txt','4', 2017,8,10, 'rnn',event)
        #Reading_data('Prueba_Bezymyanny_BZ08_08_bz08170810000000_BHZ_probabilities_RNN.txt','5', 2017,8,10, 'rnn',event)

        data_ordered = collections.OrderedDict(sorted(rnn_equ_station1.items()))
        data_ordered2 = collections.OrderedDict(sorted(rnn_equ_station2.items()))
        data_ordered3 = collections.OrderedDict(sorted(rnn_equ_station3.items()))
        data_ordered4 = collections.OrderedDict(sorted(rnn_equ_station4.items()))
        data_ordered5 = collections.OrderedDict(sorted(rnn_equ_station5.items()))

	
        #getting correlation file showing the event detected in at least 2 stations
        year=2021
        month=9
        #days=monthrange(year, month)
        day=12

        if (month<10):
            str_month='0'+str(month)
        else:
            str_month=str(month)
		
        if (day<10):
            day='0'+str(day)
        else:
            day= str(day)
			
        str_year=str(year)
        date=str_year[len(str_year)-2:]+str_month+day

        get_Correlated_Events_INVOLCAN(data_ordered, data_ordered2, data_ordered3, data_ordered4, data_ordered5, event, 10, 10, date, 'rnn')

    
    sys.exit()
