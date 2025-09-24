# ML-DIGIVOLCAN
# Multi station volcano seismic monitoring system based on transfer-learning
A multi-station volcano seismic monitoring approach based on transfer learning techniques. We applied a Dilated RNN–LSTM or a classical RNN-LSTM architectures —both trained with a master dataset and catalogue belonging to Deception Island volcano (Antarctica), as blind-recognizers to a new volcanic environment (Canary Island). 

# Re-training approaches
Systems were re-trained under 2 different approaches: 

### Coherence Analysis
A multi correlation-based approach (i.e., only seismic traces detected at the same time at different seismic stations are selected). The subset of the selected events recognized are analysed by the experts, labelled, and used for re-train the models.

### Weakly Supervises Analysis
Weakly supervised learning approaches have the remarkable capability of simultaneously identifying unannotated seismic traces in the catalogue and correcting mis-annotated seismic traces. The pre-trained system is used as a pseudo-labeller within the framework of weakly supervised learning. We applied a concept drift framework in which no prior knowledge about the volcano or the availability of any seismic catalogue was considered. To do this, a weakly supervised
data-driven instance selection algorithm was applied as follows:
#####
  (a) A subset of the target dataset (in our case 40% of the total) was analysed by the pre-trained source.
#####
  (b) Discarding information contained within the available seismic catalogue, for each detected event, the confidence of the detection was analysed using a probabilistic event detection matrix with per-class membership outputted by the softmax layer. The per-class recognition probabilities for each event type were extremely high, showing that the systems were perfectly fitted to the master database. This action also allowed us to quantify the severity of drift between datasets. Here, we assumed that low per-class probabilities reflect a change in the description of the analysed information. It is important to note that both problems, how to define an accurate and robust dissimilarity measurement and use a specific hypothesis to evaluate
the statistical significance of the change observed, are not strictly necessary since the dissimilarity between the volcanic environments is well-known.
#####
  (c) A drift adaptation mechanism based on an adaptive threshold was then adopted. Those events whose average number of per-class probability was greater than a given threshold were selected and included as training instances.
  #####
  (d) Finally, the pre-trained systems were re-trained using the selected instances considering as labels the pseudo-labels obtained by the pre-trained models during the recognition task.


## Dilated/Classical RNN-LSTM and TCN

The Dilated RNN-LSTM architecture implemented in this work uses as baseline the open source code located at Github [https://github.com/zalandoresearch/pytorch-dilated-rnn]. Users can change the number of layer easily getting the model deeper.

The TCN architecture implemented in this work uses as baseline the open source code located at Github [https://github.com/philipperemy/keras-tcn]

Inputs are 4-second windows parameterized as 48-features vectors corresponding to a bank of filters (16 filters) as well as their first and second derivatives.

The following works were especially helpful for understanding the parameterization scheme:   

Titos, M., Bueno, A., García, L., Benítez, M. C., & Ibáñez, J. (2018). Detection and classification of continuous volcano-seismic signals with recurrent neural networks. *IEEE Transactions on Geoscience and Remote Sensing*, **57**(4), 1936–1948.  

Titos, M., Carthy, J., García, L., Barnie, T., & Benítez, C. (2024). Dilated-RNNs: A deep approach for continuous volcano-seismic events recognition. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, **17**, 11857–11865. https://doi.org/10.1109/JSTARS.2024.3421921 

### Installation

To run the software, you can install the required dependencies using the provided `spec-file_torch.txt` file in the GitHub repository:  

```bash
conda create --name myenv --file spec-file_torch.txt
``` 

## Using the Models

### Configuration File
The system is configured through a **configuration file**, where parameters are modified in different header sections:

- **[DEFAULT]**  
  Defines the recognition model parameters, such as:  
  - `n_layers`: number of layers  
  - `tcn_n_filters`: number of filters in the TCN  

- **[weakly]**  
  Controls the weakly supervised re-training process:  
  - `weakly`: whether weakly supervised re-training is enabled  
  - `min_chunk`: size of the seismic segments to be recognized (in minutes)  
  - `prob_threshold`: probability threshold to include or discard recognized events in the database  

- **[model_path]**  
  Specifies the location of the pre-trained models.  
  By default, these correspond to the models included in the repository.  

- **[stations]**  
  Lists the stations to be analyzed.  
  In the provided example, these are stations from *La Palma volcano* in the Canary Islands.  

- **[data_path]**  
  Defines the data directories and acquisition details:  
  - `relative`: path to the seismic data  
  - `network`: seismic network code  
  - `component`: component to be analyzed  

- **[frequency]**  
  Sets the refresh rate in real-time mode (in seconds).  
  This parameter determines how often the system performs a new recognition and re-training cycle.  

- **[monitoring]**  
  Defines whether the system runs in real-time or offline mode:  
  - `real_time`: `True` for real-time operation, `False` for offline  
  - `alternative_date`: in offline mode, the analysis is performed between the dates specified here

## Running the Models

To run the models, simply adjust the **configuration file** and move into the corresponding directory (`Dilated-LSTM` or `TCN`).  

- For **real-time recognition and catalog creation**, run:  

```bash
python3 real_time_monitoring.py
```

- For **recognition, catalog creation, and system re-training (using the weakly supervised pretraining approach)**, run:  

```bash
python3 weakly_retraining.py
```
## Acknowledgments

This software was developed as part of the DigiVolCan project - A digital infrastructure for forecasting volcanic eruptions in the Canary Islands.
The results are the outcome of collaboration between the University of Granada, the Canary Islands Volcanological Institute (INVOLCAN), the Institute of Technological and Renewable Energies (ITER), the University of La Laguna, and Aragón Photonics.

![Project Screenshot](./logo.png)

This work has been funded by the Spanish Project PID2022-143083NB-I00, “LEARNING”, funded by MICIU/AEI /10.13039/501100011033 and by FEDER, UE. This work has been partially funded by the Spanish Project: PLEC2022-009271 “DigiVolCan”, funded by MCIN/AEI, funded by MCIN/AEI/10.13039/501100011033 and by EU NextGenerationEU/PRTR, 10.13039/501100011033 and SDAS (B-TIC-542-UGR20). The authors would also like to acknowledge Instituto Andaluz de Geofísica,  Instituto Volcanológico de Canarias (INVOLCAN), Raúl Arámbula-Mendoza and CENAPRED for providing the data from Deception Island, Popocatépetl and Tajogaite volcanoes supporting this work. 


