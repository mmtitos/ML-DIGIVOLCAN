import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import drnn
import os
import pandas as pd
import sys
import scipy.io
import drnn
from torchsummary import summary
import _pickle as pickle
import pickle as cPickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt
torch.backends.cudnn.enabled = False
from sklearn.preprocessing import StandardScaler
import sys
scaler = StandardScaler()
import datetime
import scipy
from scipy import signal
from scipy.signal import butter, lfilter, hilbert
import obspy as obspy
from scipy.optimize import curve_fit
from scipy.spatial import distance
from subprocess import call
sys.path.append('../')
from features import *
import torch.optim as optim
import torch.nn as nn
import configparser
from coherence_analysis import *
from datetime import date, timedelta
import multiprocessing


#Defining a classifier class to be used in the main program

class Classifier(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_layers, n_classes, cell_type="GRU"):
        super(Classifier, self).__init__()

        self.drnn = drnn.DRNN(n_inputs, n_hidden, n_layers, dropout=0, cell_type=cell_type)
        self.softmax= nn.Softmax(dim=2)
        #self.linear = nn.Linear(n_hidden, n_classes)

    def forward(self, inputs):
        layer_outputs, _ , linear_output= self.drnn(inputs)
        #pred = self.linear(layer_outputs[-1])
        #pred = self.softmax(linear_output)
        pred= linear_output
        return pred



def compute_features(filename, min, norm_var=True, norm_colum=False):

    """Compute Log filter bank features from a seismic signal. 4 second windows; 3.5 second overlap. 16 filter

    :param filename: the seismic signal from which to compute features. Should be an N*1 array
    :param norm_var: variance normalization. By default is TRUE
    :param norm_colum: colum normalization. By default is False    
    :returns: A numpy array of size (NUMFRAMES by num filter*3 (delta+delta delata)) containing features. Each row holds 1 feature vector.
    """  
	
    print ("... computing filter bank features for each seismic trace")
    
    if (min==0):
        dataset = []
        labels = []
        signal, y = calculate_features(filename,0)
        dataset.append(signal)
        labels.append(y)
    else:
        dataset, labels = calculate_features(filename, min)

    delta_delta = []
    accelerations = []

    # this might not be the best thing to do, but will do as a proof of concept.
    print ("... Calculating d+dd and their accelerations...")
    for x in range(len(dataset)):
        current_signal = dataset[x]
        # this function compute delta features from a feature vector sequence.
        # First argument is a numpy array of size (NUMFRAMES x number of features). Each row holds 1 feature vector.
    # N: For each frame, calculate delta features, based on previous N frames
        deltas = base.delta(current_signal, 2)
        accelerations = base.delta(deltas, 2)
        #print "Signal shape: " + str(current_signal.shape)
        derivatives_accelerations = np.hstack((deltas, accelerations))
        #print "Signal + derivatives shape: " + str(derivatives_accelerations.shape)
        #print "Signal + derivatives + accelerations " + str(np.hstack((current_signal,derivatives_accelerations)).shape)
        new_signal = np.hstack((current_signal, derivatives_accelerations))

        dataset[x] = new_signal
    
    print ("... Creating dataset of features: Done...")
    
    if (norm_var):
        
        mean = []
        stds = []
        for j, keq in enumerate(dataset):
           if j == 0:
              total = keq
           else:
              total = np.concatenate((total, keq), axis=0)

        for k in range(total.shape[1]):
           mean.append(np.mean(total[:, k]))
           stds.append(np.std(total[:, k]))

        for m, seq in enumerate(dataset):
           for i in range(seq.shape[1]):
               seq[:, i] = (seq[:, i]-mean[i])/stds[i]
        
    if (norm_colum):
        norm_colum = Leer_Norm("Norm_Colum.txt")
        minimo = list(map(float, norm_colum[0]))
        maximo = list(map(float, norm_colum[1]))

        for l, keq in enumerate(dataset):
            for n in range(keq.shape[1]):
                keq[:, n] = (keq[:, n]-minimo[n])/(maximo[n]-minimo[n])

    return labels, dataset

def compute_features_real_time(filename ,norm_var=True, norm_colum=False):

    """Compute Log filter bank features from a seismic signal. 4 second windows; 3.5 second overlap. 16 filter

    :param filename: the seismic signal from which to compute features. Should be an N*1 array
    :param norm_var: variance normalization. By default is TRUE
    :param norm_colum: colum normalization. By default is False    
    :returns: A numpy array of size (NUMFRAMES by num filter*3 (delta+delta delata)) containing features. Each row holds 1 feature vector.
    """  
	
    print ("... computing filter bank features for each seismic trace")
    
    dataset, labels = calculate_features_real_time(filename)

    delta_delta = []
    accelerations = []

    # this might not be the best thing to do, but will do as a proof of concept.
    print ("... Calculating d+dd and their accelerations...")
    for x in range(len(dataset)):
        current_signal = dataset[x]
        # this function compute delta features from a feature vector sequence.
        # First argument is a numpy array of size (NUMFRAMES x number of features). Each row holds 1 feature vector.
    # N: For each frame, calculate delta features, based on previous N frames
        deltas = base.delta(current_signal, 2)
        accelerations = base.delta(deltas, 2)
        #print "Signal shape: " + str(current_signal.shape)
        derivatives_accelerations = np.hstack((deltas, accelerations))
        #print "Signal + derivatives shape: " + str(derivatives_accelerations.shape)
        #print "Signal + derivatives + accelerations " + str(np.hstack((current_signal,derivatives_accelerations)).shape)
        new_signal = np.hstack((current_signal, derivatives_accelerations))

        dataset[x] = new_signal
    
    print ("... Creating dataset of features: Done...")
    
    if (norm_var):
        
        mean = []
        stds = []
        for j, keq in enumerate(dataset):
           if j == 0:
              total = keq
           else:
              total = np.concatenate((total, keq), axis=0)

        for k in range(total.shape[1]):
           mean.append(np.mean(total[:, k]))
           stds.append(np.std(total[:, k]))

        for m, seq in enumerate(dataset):
           for i in range(seq.shape[1]):
               seq[:, i] = (seq[:, i]-mean[i])/stds[i]
        
    if (norm_colum):
        norm_colum = Leer_Norm("Norm_Colum.txt")
        minimo = list(map(float, norm_colum[0]))
        maximo = list(map(float, norm_colum[1]))

        for l, keq in enumerate(dataset):
            for n in range(keq.shape[1]):
                keq[:, n] = (keq[:, n]-minimo[n])/(maximo[n]-minimo[n])

    return labels, dataset

def calculate_features_real_time(filename):

    """Compute Log filter bank features from a seismic signal. 4 second windows; 3.5 second overlap. 16 filter

    :param filename: the seismic signal from which to compute features. Should be an N*1 array.   
    :returns: A numpy array of size (NUMFRAMES by num filter*3 (delta+delta delata)) containing features. Each row holds 1 feature vector.
    """ 

    f4 = "values_fbank.p"
    params = cPickle.load(open(f4, 'rb'))
    signal, fs = read_signal_obspy_filter(filename)
    labels = []
    features = []

    if len(params) == 8:
        samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph = params[
            0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]
        
    print(f"Params: {samplerate} Hz, WinLen{winlen} sec, WinStep {winstep} sec, Num_Filter {nfilt}, {nfft} FFT, {lowfreq} LowFreq, {highfreq} HighFreq, {preemph} Preempha")

    lbl = 2
    samplerate=fs

    feat = np.float32(logfbank(signal, samplerate, winlen,
                                winstep, nfilt, nfft, lowfreq, highfreq, preemph))

    etiquetas = np.int32(np.array([lbl] * feat.shape[0]))
    labels.append(etiquetas)
    features.append(feat)

    return features, labels

def calculate_features(filename, min):

    """Compute Log filter bank features from a seismic signal. 4 second windows; 3.5 second overlap. 16 filter

    :param filename: the seismic signal from which to compute features. Should be an N*1 array.   
    :returns: A numpy array of size (NUMFRAMES by num filter*3 (delta+delta delata)) containing features. Each row holds 1 feature vector.
    """ 

    f4 = "values_fbank.p"
    params = cPickle.load(open(f4, 'rb'))
    signal, fs = read_signal_obspy_filter(filename)
    labels = []
    features = []

    if len(params) == 8:
        samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph = params[
            0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]
        
    print(f"Params: {samplerate} Hz, WinLen{winlen} sec, WinStep {winstep} sec, Num_Filter {nfilt}, {nfft} FFT, {lowfreq} LowFreq, {highfreq} HighFreq, {preemph} Preempha")

    lbl = 2
    samplerate=fs

    if (min>0):
        data_windowed= Split_data(signal, min, fs)
       
        lbl = 2
        for signal in data_windowed:
            feat = np.float32(logfbank(signal, samplerate, winlen,
                                   winstep, nfilt, nfft, lowfreq, highfreq, preemph))

            etiquetas = np.int32(np.array([lbl] * feat.shape[0]))
            labels.append(etiquetas)
            #values = np.delete(values,range(3))
            features.append(feat)
    else:

        feat = np.float32(logfbank(signal, samplerate, winlen,
                                   winstep, nfilt, nfft, lowfreq, highfreq, preemph))

        etiquetas = np.int32(np.array([lbl] * feat.shape[0]))
        labels.append(etiquetas)
        features.append(feat)
    
    if (min==0):

        for j, keq in enumerate(features):
            if j == 0:
                total = keq
            else:
                total = np.concatenate((total, keq), axis=0)

        for l, meq in enumerate(labels):
            if l == 0:
                y = meq
            else:
                y = np.concatenate((y, meq), axis=0)

        return total, y
    else:
        return features, labels

def read_signal_obspy_filter(filename):

    """Read the seismic signal using obspy library. 
    :returns: A numpy array containing the samples in time domain.
    """ 

    print ('Reading' + filename+' using obspy library...')
    st= obspy.read(filename)
    tr=st[0]
    tr_filt=tr.copy()
    tr_filt.filter("highpass", freq=1.0, corners=4, zerophase=True)
    data=tr_filt.data-np.mean(tr_filt.data)
    return tr_filt.data, tr_filt.stats.sampling_rate
    #return data, tr_filt.stats.sampling_rate

def Create_List_Tensor(data):

    """Create a tensor data structure to be used in GPU devices

    :param data : the list containing the set of signals. Min size should be 1   
    :returns: A tensor data structure of size Num_signals.
    """ 

    num_instances=[]
    list_tensor=[]
    for seq in range (len(data)):

        #seq_tensor= torch.zeros((1, data[seq].shape[0], 48)).float().cuda()
        seq_tensor= torch.zeros((1, data[seq].shape[0], 48)).float()
        num_instances.append(data[seq].shape[0])
        seq_tensor[0][:]=torch.FloatTensor(data[seq])
        seq_tensor = seq_tensor.transpose(0,1)
        list_tensor.append(seq_tensor)
    #print (sum(num_instances))
    return list_tensor

def Create_List_Tensor_label(data):

    """Create a tensor data structure to be used in GPU devices

    :param data : the list containing the set of signals labels. Min size should be 1   
    :returns: A tensor data structure of size Num_signals.
    """

    list_tensor=[]
    for seq in range (len(data)):

        #seq_tensor= torch.zeros((1, data[seq].shape[0])).long().cuda()
        seq_tensor= torch.zeros((1, data[seq].shape[0])).long()

        seq_tensor[0][:]=torch.LongTensor(data[seq])
        seq_tensor = seq_tensor.squeeze(0)
        #print (seq_tensor.shape)
        list_tensor.append(seq_tensor)
    return list_tensor


def get_recognition(data,ml_model):

    """Get recognition in term of windows, probabilities for each window and level of activation per hidden units (neuron).
    :param data : the list containing the set of signals. Min size should be 1
    :param model : model pre-trained used to guide the recognition    
    :returns: 3 arrays containing: label each window, probabilities and activation for each window.
    """ 

    pred_labe= []
    #activations_data=[]
    output_prob=[]
    softmax= nn.Softmax(dim=2)
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,seq in enumerate(data):
            # calculate outputs by running images through the network
            outputs= ml_model(seq)
            probabilities=softmax(outputs)
            #outputs_copy= torch.clone(outputs)
            outputs = outputs.transpose(0, 1)
            outputs = torch.squeeze(outputs, 0)
            # the class with the highest energy is what we choose as prediction
            pred_y = torch.max(outputs, 1)[1].data.squeeze().tolist()
            pred_labe.append(pred_y)
            
            probabilities = probabilities.transpose(0, 1)
            probabilities = torch.squeeze(probabilities, 0)
            output_prob.append(probabilities.cpu().detach().numpy())

            '''
            for layer in range (len(activations)):
                if (len(activations_data)<len(activations)):
                    activations_data.append([])
                activations[layer] = torch.squeeze(activations[layer], 1)
                activations_data[layer].append(activations[layer].cpu().detach().numpy())
            '''

    return pred_labe, output_prob

def Recognize_data(data_test, labels ,model):

    """Get recognition in term of windows, probabilities for each window and level of activation per hidden units (neuron).
    :param data_test : the list containing the set of signals. Min size should be 1
    :param model : model pre-trained used to guide the recognition    
    :returns: 3 arrays containing: label each window, probabilities and activation for each window.
    """ 
    pred_labe, output_prob, activations_data= get_recognition(data_test, labels ,model)

    return pred_labe, output_prob, activations_data


def Find_instant_Events_probability(prediction, probabilities, namefile):

    """Create a file containing the events detected within a given seismic trace.
    :param prediction: N*1 array of predicted labels
    :param probabilities : N*5 array of probabilities associated with all the predicted labels
    :param namefile: name of the created file containing the pseudo-catalog  
    :returns: psuedo catalog of detected events
    """ 

    print ('creating pseudo catalog...')
    win_len=[2, 40, 14, 4, 4]
    events= [" Sil", "Tre", "Hyb", "Equ", "Lpe", "EDS"]
    events_list=Event_Limits2(prediction)
    lower=''
    upper=''
    #print (events_list)
    with open(namefile,"w+") as f:
    
        for i in range(len(events_list)):
            gramatic_vector=[]
            probabilities_ind=probabilities[i]
            #f.write("Predi_frames-> "+ np.array_str(prediction[i])+"\n")
            #f.write("\n")
            #create label gramatic
            intervals_label=len(events_list[i])
            down=0
            for j in range (intervals_label):
                if (down>0):
                    lower=str((int(down)-1)*0.5+4)

                else:
                    lower=str(0)
                upper= str((int(events_list[i][j])-1)*0.5+4+4)
                #print (upper, down)
                #Compute win len and event type
                if(events_list[i][j]-down>win_len[prediction[i][events_list[i][j]-1]]):
                    f.write(" predi-> ")
                    f.write(events[prediction[i][events_list[i][j]-1]]+"\t")
                    gramatic_vector.append(events[prediction[i][events_list[i][j]-1]])
                    #f.write('| '+lower+'---'+upper)
                    #f.write("\n")
                    #down=events_list[i][j]
                else:
                    f.write(" predi-> ")
                    f.write(events[5]+"\t")
                    gramatic_vector.append(events[5])
                    #f.write('| '+lower+'---'+upper)
                    #f.write("\n")
                    #down=events_list[i][j]

                f.write('| '+lower+'---'+upper)
                f.write("\n")
                f.write("     SIL\t TRE\t     HYB\t   EQ\t      LPE"+"\n")
                #probabilities_frame=probabilities_ind[interval_down:interval_up]
                probabilities_frame=probabilities_ind[down:int(events_list[i][j])]
                #print (down,int(events_list[i][j])-1 )
                means=np.mean(probabilities_frame, axis=0)
                f.write(np.array_str(means)+"\n")
                #print (means)
                f.write("\n")
                down=events_list[i][j]

            if(intervals_label==0):
                intervals_label=1
                events_list[i].append(0)

            if (down>0):
                lower=str((int(down)-1)*0.5+4)

            else:
                lower=str(0)
            upper= 'until end'
            if(len(prediction[i])-events_list[i][intervals_label-1] > win_len[prediction[i][len(prediction[i])-1]]):
                f.write(" predi-> ")
                gramatic_vector.append(events[prediction[i][len(prediction[i])-1]])
                f.write(events[prediction[i][len(prediction[i])-1]])
                #f.write('| '+lower+'---'+upper)
                #f.write("\n")
            else:
                f.write(" predi-> ")
                f.write(events[5]+"\t")
                #f.write('| '+lower+'---'+upper)
                gramatic_vector.append(events[5])
                #f.write("\n")

            f.write('| '+lower+'---'+upper)
            f.write("\n")
            f.write("     SIL\t TRE\t     HYB\t   EQ\t      LPE"+"\n")
            probabilities_frame=probabilities_ind[events_list[i][intervals_label-1]:len(prediction[i])]
            #print (probabilities_frame.shape)
            #print (down,int(events_list[i][j])-1 )
            means=np.mean(probabilities_frame, axis=0)
            f.write(np.array_str(means)+"\n")
            #print (means)
            f.write("\n")
        f.close()


def create_Data_Base_Using_Grammar(ml_model, dataset, labels, prob_threshold):

    """Create a dataset containing the events detected within a given seismic trace up to a probability threshold. It is used
    for weakly supervised approaches.
    :param ml_model: machine learning system pre-trained used to psuedo label the seismic trace
    :param daset : seismic trace parammeterized
    :param labels: labels for the parammeterized data. No used in this script since we look to pseudo label the data  
    :returns: pseudo data and pseudo labels detected
    """ 
	
    data_training_confidence=[]
    label_training_confidence=[]
    probabilities_confidence=[]
    chunk_selected=[]


    pred_label, probabilities=get_recognition(dataset,ml_model)

    events_delimited=Event_Limits2_chunk(pred_label)
	
    for i,records in enumerate(events_delimited):
        down=0
        suggested_label=[]
        threshold=[]
        threshold_value=[]
        for j,event in enumerate (records):
            prob_event=np.mean(probabilities[i][down:event], axis=0)
            threshold_value.append(prob_event[np.argmax(prob_event)])
            if (prob_event[np.argmax(prob_event)]>prob_threshold):
                suggested_label.append(np.argmax(prob_event))
                threshold.append(True)
            else:
                threshold.append(False)
			#print (down,event)
			#print(prob_event)

            down=event
		#prob_event=probabilities[i][down:]
        prob_event=np.mean(probabilities[i][down:], axis=0)
        threshold_value.append(prob_event[np.argmax(prob_event)])
        if (prob_event[np.argmax(prob_event)]>prob_threshold):
            suggested_label.append(np.argmax(prob_event))
            threshold.append(True)
        else:
            threshold.append(False)
        if (not False in threshold):
            data_training_confidence.append(dataset[i])
            label_training_confidence.append(labels[i])
            probabilities_confidence.append(probabilities[i])
            down_label=0
            for k,index in enumerate (records):

                label_training_confidence[-1][down_label:index]=suggested_label[k]
                down_label=index
            label_training_confidence[-1][down_label:]=suggested_label[-1]
            chunk_selected.append(True)
        else:
            chunk_selected.append(False)
	
    label_training_confidence= Create_List_Tensor_label(label_training_confidence)

    print ('New dataset size is: ', len(data_training_confidence))
    print ('Original dataset size was: ', len(dataset))
    
    return data_training_confidence, label_training_confidence, chunk_selected


def Retraining_model_by_station(model, dataset, labels, original_dataset, original_labels, namefile):

    """Retrain a given model given a pseudo labelled dataset.
    :param ml_model: machine learning system pre-trained used to psuedo label the seismic trace
    :param daset : seismic trace parammeterized
    :param labels: labels for the parammeterized data. No used in this script since we look to pseudo label the data  
    :returns: model retrained
    """ 

    from sklearn.metrics import accuracy_score
    
    training_iters = 5
    display_step = 25
    display_step = 1
    previous=0
    best_accuracy=0

    data_training, data_test=split_dataset(dataset, 80)
    label_training, label_test=split_dataset(labels, 80)


    #optimizer = optim.Adam(model.parameters(), lr=0.004)
    optimizer= optim.SGD(model.parameters(), lr=0.004)
    #optimizer= optim.Adagrad(model.parameters(), lr=0.004)
    criterion = nn.CrossEntropyLoss()


    print('Starting the Re-training process...')
    for iter in range(training_iters):
        optimizer.zero_grad()
        total_error=[]
        for i, (seq, labels) in enumerate(zip(data_training, label_training)):
            pred= model.forward(seq)
            pred = torch.squeeze(pred, 1)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            total_error.append(loss.cpu().detach().numpy())
        
        if (iter + 1) % display_step == 0:

            print("Iter " + str(iter + 1) + ", Average Loss: " + "{:.6f}".format(np.mean(total_error)))
            pred_label, probabilities=get_recognition(data_test,model)
            #print (torch.cat(label_test).cpu().numpy().shape)
            #print (np.concatenate( pred_label, axis=0 ))
            previous=accuracy_score(torch.cat(label_test).cpu().numpy(), np.concatenate( pred_label, axis=0 ))
            print ('accuracy: ', previous)
            if (previous>best_accuracy):
                best_accuracy=previous
                print('Saving the model...')
                #torch.save(model.state_dict(), '/home/manuel/Documents/Dilated_RNN/pytorch-dilated-rnn-deception/best_model_TF_POPO_5classes_20_2approach')
    
    
    pred_label_after_weakly, probabilities_after_weakly=get_recognition(original_dataset,model)

    Find_instant_Events_probability(pred_label_after_weakly,probabilities_after_weakly ,namefile)


def real_time_monitoring(station_path, station_names, coherence_event, dates, model,ml_model):
    
    files_coherence=[]

    for station, name in zip (station_path, station_names):

        joint_detection=[]
        joint_probabilities=[]
        print (station, name)
        start_time = time.perf_counter()
        labels,dataset= compute_features_real_time(station,norm_var=True, norm_colum=False)

        #windowing signal for recognizing by slots of time
        #dataset=windowing_signal(dataset[0], 7200)

        print ('creating tensors to be used in GPU devices ...')
        dataset_tensor= Create_List_Tensor (dataset)

        end_time = time.perf_counter()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        print("Elapsed time for computing features: ", elapsed_time)

        # Get detections for PLPI and PPMA stations

        start_time = time.perf_counter()
        print ('Getting recognitions, probabilities and activations...')
        detection_, probabilities_= get_recognition(dataset_tensor,ml_model)

        # for recognizing by slot of time
        #joint_detection.append(sum(detection_, []))
        #joint_probabilities.append(np.vstack(probabilities_))

        end_time = time.perf_counter()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        print("Elapsed time for getting detection and probabilities: ", elapsed_time)

        print ('Creating pseudo catalog file...')
        extention='.txt'
        extention_csv='.csv'
        aux='pseudo_catalog_'
        #date=datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        date='_'+dates.strftime('%Y-%m-%d')

        #namefile = f'{aux}{name}{extension}'
        files_coherence.append(f'{aux}{name}{model}{date}{extention}')


        start_time = time.perf_counter()
        Find_instant_Events_probability(detection_,probabilities_,f'{aux}{name}{model}{date}{extention}')
        ##windowing signal for recognizing by slot of time
        #Find_instant_Events_probability(joint_detection,joint_probabilities,f'{aux}{name}{date}{extention}')

        end_time = time.perf_counter()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        print("Elapsed time for creating detection files: ", elapsed_time)

        #write file csv
        DF = pd.DataFrame(np.squeeze(probabilities_, axis=0))
        #windowing signal for recognizing by slot of time
        #DF = pd.DataFrame(joint_probabilities[0]) 
        # save the dataframe as a csv file
        DF.to_csv(f'{aux}{name}{model}{date}{extention_csv}', mode='w')
    
    for event in coherence_event:

        start_time = time.perf_counter()

        for j in range(len(station_names)):
            rnn_stations.append({})

        for i,file in enumerate (files_coherence):
            Reading_data_real_time(file, i, dates.year,dates.month,dates.day, 'rnn',event)
            data_ordered.append(collections.OrderedDict(sorted(rnn_stations[i].items()))) 

        #date=get_date(2021,9,12)
        #get_date(datetime.now().year,datetime.now().month,datetime.now().day)
        
        get_Correlated_Events_INVOLCAN_real_time(event, 10, 10, dates, 'rnn')

        rnn_stations.clear()
        data_ordered.clear()

        end_time = time.perf_counter()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        print("Elapsed time for creating coherence file for: "+event, elapsed_time)

def process(station_path, station_name, dates,files_coherence,model, ml_model):
        
    start_time = time.perf_counter()
    labels,dataset= compute_features_real_time(station_path,norm_var=True, norm_colum=False)

    #windowing signal for recognizing by slots of time
    #dataset=windowing_signal(dataset[0], 7200)

    print ('creating tensors to be used in GPU devices ...')
    dataset_tensor= Create_List_Tensor (dataset)

    end_time = time.perf_counter()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time for computing features: ", elapsed_time)

    # Get detections for PLPI and PPMA stations

    start_time = time.perf_counter()
    print ('Getting recognitions, probabilities and activations...')
    detection_, probabilities_= get_recognition(dataset_tensor,ml_model)

    # for recognizing by slot of time
    #joint_detection.append(sum(detection_, []))
    #joint_probabilities.append(np.vstack(probabilities_))

    end_time = time.perf_counter()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time for getting detection and probabilities: ", elapsed_time)

    print ('Creating pseudo catalog file...')
    extention='.txt'
    extention_csv='.csv'
    aux='pseudo_catalog_'
    #date=datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    date='_'+dates.strftime('%Y-%m-%d')

    #namefile = f'{aux}{name}{extension}'
    files_coherence.append(f'{aux}{station_name}{model}{date}{extention}')


    start_time = time.perf_counter()
    Find_instant_Events_probability(detection_,probabilities_,f'{aux}{station_name}{model}{date}{extention}')
    ##windowing signal for recognizing by slot of time
    #Find_instant_Events_probability(joint_detection,joint_probabilities,f'{aux}{name}{date}{extention}')

    end_time = time.perf_counter()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time for creating detection files: ", elapsed_time)

    #write file csv
    DF = pd.DataFrame(np.squeeze(probabilities_, axis=0))
    #windowing signal for recognizing by slot of time
    #DF = pd.DataFrame(joint_probabilities[0]) 
    # save the dataframe as a csv file
    DF.to_csv(f'{aux}{station_name}{model}{date}{extention_csv}', mode='w')


def process_retraining(station_path,min_chunk,prob_threshold,station_name, dates,model, ml_model):
        
    start_time = time.perf_counter()
    labels,dataset= compute_features(station_path,min_chunk,norm_var=True, norm_colum=False)

    #windowing signal for recognizing by slots of time
    #dataset=windowing_signal(dataset[0], 7200)

    print ('creating tensors to be used in GPU devices ...')
    dataset_tensor= Create_List_Tensor (dataset)

    end_time = time.perf_counter()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time for computing features: ", elapsed_time)


    start_time = time.perf_counter()
    print ('Getting weakly dataset for re-training the expert for the selected station: ', station_name)
    datatraining_confidence, label_confidence, chunk_selected=create_Data_Base_Using_Grammar(ml_model, dataset_tensor,labels , prob_threshold)

    end_time = time.perf_counter()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time for getting weakly dataset: ", elapsed_time)

    print ('Training model using weakly dataset...')

    start_time = time.perf_counter()
    extention='.txt'
    aux='pseudo_catalog_after_weakly_retraining_'
    #date=datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    date='_'+dates.strftime('%Y-%m-%d')

    Retraining_model_by_station(ml_model, datatraining_confidence, label_confidence, dataset_tensor, labels, f'{aux}{station_name}{model}{date}{extention}')
    end_time = time.perf_counter()
    print("Elapsed time for weakly retraining: ", elapsed_time)


def real_time_monitoring_parallel(station_path, station_names, coherence_event, dates, model,expert_by_station):
    
    #from multiprocessing import Process
    files_coherence=[]

    for index_station,(station, name) in enumerate(zip (station_path, station_names)):   
        print (station, name, dates)
        # launch a process for each file (ish).
        # The result will be approximately one process per CPU core available.
        process (station, name,dates, files_coherence ,model,expert_by_station[index_station])
        #processes.append(Process(target=process_station, args=(station, name,dates, files_coherence ,expert_by_station[index_station])))
        # launch a process for each file (ish).
        # The result will be approximately one process per CPU core available.

    #for process in processes:
        #process.start()

    # now wait for them to finish
    #for process in processes:
        #process.join()

    
    for event in coherence_event:

        start_time = time.perf_counter()

        for j in range(len(station_names)):
            rnn_stations.append({})

        for i,file in enumerate (files_coherence):
            Reading_data_real_time(file, i, dates.year,dates.month,dates.day, 'rnn',event)
            data_ordered.append(collections.OrderedDict(sorted(rnn_stations[i].items()))) 

        #date=get_date(2021,9,12)
        #get_date(datetime.now().year,datetime.now().month,datetime.now().day)
        
        get_Correlated_Events_INVOLCAN_real_time(event, 10, 10, dates, 'rnn')

        rnn_stations.clear()
        data_ordered.clear()
        end_time = time.perf_counter()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        print("Elapsed time for creating coherence file for: "+event, elapsed_time)

def weakly_reatrain_parallel(station_path, min_chunk ,prob_threshold, station_names, dates, model,expert_by_station):
    
    #from multiprocessing import Process

    for index_station,(station, name) in enumerate(zip (station_path, station_names)):   
        print (station, name, dates)
        # launch a process for each file (ish).
        # The result will be approximately one process per CPU core available.
        process_retraining (station, min_chunk, prob_threshold,name,dates ,model,expert_by_station[index_station])
        #processes.append(Process(target=process_station, args=(station, name,dates, files_coherence ,expert_by_station[index_station])))
        # launch a process for each file (ish).
        # The result will be approximately one process per CPU core available.

    #for process in processes:
        #process.start()

    # now wait for them to finish
    #for process in processes:
        #process.join()

def Leer_Norm(filename):
    norm_values = []
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.rstrip("\n")
                values = line.split()
                norm_values.append(values)
        f.close()
    return norm_values

def Event_Limits2(X_test):
    list_of_events=list()
    for i in range(len(X_test)):
        event=list()
        for j in range(len(X_test[i])-1):
            if(X_test[i][j]!=X_test[i][j+1]):
                event.append(j+1)
        list_of_events.append(event)
    return list_of_events

def Event_Limits2_chunk(X_test):
    list_of_events=list()
    for i in range(len(X_test)):
        event=list()
        for j in range(len(X_test[i])-1):
            if(X_test[i][j]!=X_test[i][j+1]):
                event.append(j+1)
        list_of_events.append(event)
    return list_of_events

def Split_data (data_training, min, fs):   
    min_to_sample= (min*60)*fs
    print (data_training.shape)
    data_splitted=np.array_split(data_training, np.rint(data_training.shape[0]/min_to_sample))
    #data_splitted=np.array_split(data_training[0], np.rint(data_training[0].shape[0]/min_to_win))
    return data_splitted

def split_dataset(dataset, percentaje):

    ntraining = int(len(dataset)*percentaje)/100
    training = dataset[0:int(ntraining)]
    test = dataset[int(ntraining):len(dataset)]

    return training, test

def get_date(year, month, day):

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
    
    return date

def windowing_signal(array, lenght):
    windows = []
    for i in range(0, len(array), lenght):
        window = array[i:i + lenght]
        windows.append(window)
    # Si la última ventana es más pequeña que el tamaño deseado, añádela
    #if len(array) % lenght != 0:
    #    window = array[len(array) - (len(array) % lenght):]
    #    windows.append(window)
    return windows

def read_Config_File():
    
    reading=[]
    station_path=[]
    config = configparser.ConfigParser()
    config.read('../config_real_time.ini')
    n_layers=int(config['DEFAULT']['n_layers'])
    num_station= int(config['DEFAULT']['number_stations'])
    station_names=config['stations']['names'].split(',')
    coherence_event= config['coherence_analysis']['events'].split(',')
    real_time= config['monitoring']['real_time']
    alternative_dates= config['monitoring']['alternative_date']
    relative_path=config['data_path']['relative']
    component= config['data_path']['component']
    network= config['data_path']['network']
    dates = []

    if (n_layers==1):
        n_hidden = 210
    elif(n_layers==3):
        n_hidden = 50
    else:
        print ('Currently, only a classical LSTM with one hidden layer or a Dilated LSTM with 3 hidden layers available')
        print ('Exiting...')
        exit(0)

    frequency=int(config['frequency']['seconds'])
    
    if (n_layers==1):
        model_path='..'+config['model_path']['path_lstm']
        model='_lstm'
    else:
        model_path='..'+config['model_path']['path_dilated']
        model='_drnn'

    if(eval(real_time)==False):

        alternative_dates_splitted=alternative_dates.split(':')

        aux=alternative_dates_splitted[0].split('/')
        start_dt=date(int(aux[2]), int(aux[1]), int(aux[0]))

        if(len(alternative_dates_splitted)>1):

            aux=alternative_dates_splitted[1].split('/')
            end_dt=date(int(aux[2]), int(aux[1]), int(aux[0]))

            # difference between current and previous date
            delta = timedelta(days=1)
            # store the dates between two dates in a list
            
            while start_dt <= end_dt :
                # add current date to list by converting  it to iso format
                dates.append(start_dt)
                # increment start date by timedelta
                start_dt += delta
        else:

            dates.append(start_dt)
        
    else:
        dates.append (date.today())
        #print (date.today().timetuple().tm_yday)

    for my_date in dates:
        station_path=[]
        yday=my_date.timetuple().tm_yday
        if (yday<100):
            yday_str='0'+str(yday)
        else:
            yday_str=str(yday)

        for i in range(num_station):
            path=relative_path+str(my_date.year)+'/'+network+'/'+station_names[i]+'/'+component+'/'
            namefile=network+'.'+station_names[i]+'..'+component+'.'+str(my_date.year)+'.'+yday_str
            station_path.append(path+namefile)
            #station_path.append(config['data_path'][station_names[i]])
            #rnn_stations.append({})
        reading.append(station_path)   
    print (reading)

    return n_layers, n_hidden, num_station, station_names,coherence_event, reading, frequency, model_path, real_time, dates, model

def read_Config_File_parallel(aux_file):
    
    reading=[]
    station_path=[]
    config = configparser.ConfigParser()
    config.read(aux_file)
    n_layers=int(config['DEFAULT']['n_layers'])
    num_station= int(config['DEFAULT']['number_stations'])
    station_names=config['stations']['names'].split(',')
    coherence_event= config['coherence_analysis']['events'].split(',')
    real_time= config['monitoring']['real_time']
    alternative_dates= config['monitoring']['alternative_date']
    relative_path=config['data_path']['relative']
    component= config['data_path']['component']
    network= config['data_path']['network']
    dates = []
    model_paths=[]

    if (n_layers==1):
        n_hidden = 210
    elif(n_layers==3):
        n_hidden = 50
    else:
        print ('Currently, only a classical LSTM with one hidden layer or a Dilated LSTM with 3 hidden layers available')
        print ('Exiting...')
        exit(0)

    frequency=int(config['frequency']['seconds'])
    
    for i in range (num_station):
        if (n_layers==1):
            model_path=config['model_path']['path_lstm']
            model='_lstm'
        else:
            model_path=config['model_path']['path_dilated']
            model='_drnn'
        model_paths.append('../'+station_names[i]+model_path)
    

    if(eval(real_time)==False):

        alternative_dates_splitted=alternative_dates.split(':')

        aux=alternative_dates_splitted[0].split('/')
        start_dt=date(int(aux[2]), int(aux[1]), int(aux[0]))

        if(len(alternative_dates_splitted)>1):

            aux=alternative_dates_splitted[1].split('/')
            end_dt=date(int(aux[2]), int(aux[1]), int(aux[0]))

            # difference between current and previous date
            delta = timedelta(days=1)
            # store the dates between two dates in a list
            
            while start_dt <= end_dt :
                # add current date to list by converting  it to iso format
                dates.append(start_dt)
                # increment start date by timedelta
                start_dt += delta
        else:

            dates.append(start_dt)
        
    else:
        dates.append (date.today())
        #print (date.today().timetuple().tm_yday)

    for my_date in dates:
        station_path=[]
        yday=my_date.timetuple().tm_yday
        if (yday<100):
            yday_str='0'+str(yday)
        else:
            yday_str=str(yday)

        for i in range(num_station):
            path=relative_path+str(my_date.year)+'/'+network+'/'+station_names[i]+'/'+component+'/'
            namefile=network+'.'+station_names[i]+'..'+component+'.'+str(my_date.year)+'.'+yday_str
            station_path.append(path+namefile)
            #station_path.append(config['data_path'][station_names[i]])
            #rnn_stations.append({})
        reading.append(station_path)   
    print (reading)

    return n_layers, n_hidden, num_station, station_names,coherence_event, reading, frequency, model_paths, real_time, dates, model

def read_Config_File_Weakly(aux_file):
    
    reading=[]
    station_path=[]
    config = configparser.ConfigParser()
    config.read(aux_file)
    n_layers=int(config['DEFAULT']['n_layers'])
    num_station= int(config['DEFAULT']['number_stations'])
    weakly=config['weakly']['weakly']
    min_chunk= int(config['weakly']['min_chunk'])
    prob_threshold=float(config['weakly']['prob_threshold'])
    station_names=config['stations']['names'].split(',')
    coherence_event= config['coherence_analysis']['events'].split(',')
    real_time= config['monitoring']['real_time']
    alternative_dates= config['monitoring']['alternative_date']
    relative_path=config['data_path']['relative']
    component= config['data_path']['component']
    network= config['data_path']['network']
    dates = []
    model_paths=[]

    if (n_layers==1):
        n_hidden = 210
    elif(n_layers==3):
        n_hidden = 50
    else:
        print ('Currently, only a classical LSTM with one hidden layer or a Dilated LSTM with 3 hidden layers available')
        print ('Exiting...')
        exit(0)

    frequency=int(config['frequency']['seconds'])
    
    for i in range (num_station):
        if (n_layers==1):
            model_path=config['model_path']['path_lstm']
            model='_lstm'
        else:
            model_path=config['model_path']['path_dilated']
            model='_drnn'
        model_paths.append('../'+station_names[i]+model_path)
    

    if(eval(real_time)==False):

        alternative_dates_splitted=alternative_dates.split(':')

        aux=alternative_dates_splitted[0].split('/')
        start_dt=date(int(aux[2]), int(aux[1]), int(aux[0]))

        if(len(alternative_dates_splitted)>1):

            aux=alternative_dates_splitted[1].split('/')
            end_dt=date(int(aux[2]), int(aux[1]), int(aux[0]))

            # difference between current and previous date
            delta = timedelta(days=1)
            # store the dates between two dates in a list
            
            while start_dt <= end_dt :
                # add current date to list by converting  it to iso format
                dates.append(start_dt)
                # increment start date by timedelta
                start_dt += delta
        else:

            dates.append(start_dt)
        
    else:
        dates.append (date.today())
        #print (date.today().timetuple().tm_yday)

    for my_date in dates:
        station_path=[]
        yday=my_date.timetuple().tm_yday
        if (yday<100):
            yday_str='0'+str(yday)
        else:
            yday_str=str(yday)

        for i in range(num_station):
            path=relative_path+str(my_date.year)+'/'+network+'/'+station_names[i]+'/'+component+'/'
            namefile=network+'.'+station_names[i]+'..'+component+'.'+str(my_date.year)+'.'+yday_str
            station_path.append(path+namefile)
            #station_path.append(config['data_path'][station_names[i]])
            #rnn_stations.append({})
        reading.append(station_path)   
    print (reading)

    return n_layers, n_hidden, num_station,weakly, min_chunk, prob_threshold,station_names,coherence_event, reading, frequency, model_paths, real_time, dates, model

