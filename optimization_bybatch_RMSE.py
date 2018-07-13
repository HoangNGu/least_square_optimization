import cvxEDA
from numpy import array
import numpy as np
from matplotlib import pyplot as plt
import pylab as pla
import EDA_myAPI as eda
import os
from math import log
from math import exp
import time
from scipy import signal
import math
#RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt

#rms = sqrt(mean_squared_error(y_actual, y_predicted))

#IMPORTING FILES (CSV,txt,..)
start = time.time()

workingdir = r"./data/"
csvfile = sorted(os.listdir(workingdir))
for folder, member in enumerate(csvfile):
    csvfile[folder] = workingdir + csvfile[folder]

label_dir = r"./label_oasis/"
csv_oasis = os.listdir(label_dir)
for folder, member in enumerate(csv_oasis):
    csv_oasis[folder] = label_dir + csv_oasis[folder] 
    
Name_oasis,Valence_mean_oasis,Valence_SD_oasis,Valence_N_oasis,Arousal_mean_oasis,_oasisArousal_SD_oasis,Arousal_N_oasis = [],[],[],[],[],[],[]
Name_oasis,Valence_mean_oasis,Valence_SD_oasis,Valence_N_oasis,Arousal_mean_oasis,_oasisArousal_SD_oasis,Arousal_N_oasis = eda.OpenCsvFile_oasis(csv_oasis[0])

with open(r'filenames_randorder.txt', 'r') as f:
    myNames = [line.strip() for line in f]

list_name, list_arousal, list_valence = [] , [] , []
for i in range(len(myNames)):
    list_name.append(Name_oasis[int(myNames[i])-1])
    list_arousal.append(Arousal_mean_oasis[int(myNames[i])-1])
    list_valence.append(Valence_mean_oasis[int(myNames[i])-1])

#numberOfDataSet= len(csvfile)
numberOfDataSet = 4 # number of participants
len_experience = 20100 # 10 sec countdown*10hz + 200 pictures * 10 seconds  * 10 Hz
len_stimulus = 10 #length of stimulus in seconds
delta = 0.1 # delta = 1/Fs
threshold = 0.01 #threshold to define a SCR. If the amplitude of a peak is at least 0.01 it is identified as a SCR
windowSizeLatency =  5#Number of images used to compute the mean of every first peak latency 


x = [[] for i in range(numberOfDataSet)] 
y = [[] for i in range(numberOfDataSet)]
ya = []
yn = []
yntemp=[]

for i in range(0,numberOfDataSet):
   eda.OpenCsvFile(csvfile[i],x[i],y[i])
   x[i] = [float(j) for j in x[i]] # time
   y[i] = [float(j) for j in y[i]] # potential
   ya.append(array(y[i]))

ya[0] = ya[0][1800:1800 + len_experience] #include the 10 sec countdown. 
ya[1] = ya[1][470:470+len_experience] #toshiki
ya[2] = ya[2][440:440 + len_experience]  # yusuke
ya[3] = ya[3][110:110 + len_experience]  # kento



############# FILTER #################################################################
###median filter
##WindowSize = 100
##yafilt = eda.medianFilter(ya,WindowSize)
##ya2filt = eda.medianFilter(ya2,WindowSize)
yafilt = ya  # no filter needed according to cvxEDA paper
############# N0RMALIZATION ##########################################################
#zscore normalization
##yn = (yafilt - yafilt.mean()) / yafilt.std()
for i in range(numberOfDataSet):
   yn.append(eda.zscoreNormalisation(yafilt[i]))
###Min-max normalization
yntemp = list(yn)
yn =[]
for i in range(numberOfDataSet):
   yn.append(eda.minmaxNormalisation(yntemp[i]))
############# WINDOW SIZING###########################################################
RawDataList_10s = [array([]) for i in range(numberOfDataSet)] #This list is a list of list containing rawdata component value in 8 sec time windows after each stimulus
RawDataList_20imgs = [array([]) for i in range(numberOfDataSet)]
RawDataList_WinSize = [array([]) for i in range(numberOfDataSet)]

nbImage = 18 #Change this variable to change the number of images in the window
WinStart = 3 #Use in WinSize, define the elapse time between stimulus and the start of the window
WinStop = 10 #Use in WinSize, define the elapse time between stimulus and the end of the window

#Computing mean over window size for arousal and valence
meanvalence_window = []
meanarousal_window = []
for i in eda.my_range(0, (len(list_arousal)-nbImage), nbImage):
    listtemp_arousal = []
    listtemp_valence = []
    for j in range(nbImage):
        listtemp_arousal.append(list_arousal[j+i])
        listtemp_valence.append(list_valence[j+i])
    meanarousal_window.append(np.mean(listtemp_arousal))
    meanvalence_window.append(np.mean(listtemp_valence))

for i in range(numberOfDataSet): #this will be change so the length is automaticaly detected.
   RawDataList_10s[i] =(eda.WindowSizing(0,len_experience-(len_stimulus*10),100,yn[i])) #begining of each stimulus, whole length, we start after 10 secs countdownand time window of 10s after each stimulus
   RawDataList_20imgs[i]=(eda.WindowSizing_ImagNumber(0,len_experience-(len_stimulus*10),nbImage,yn[i]))
   RawDataList_WinSize[i]=(eda.WindowSizing_WinStartStop(0,len_experience-(len_stimulus*10),100,WinStart,WinStop,yn[i]))


############# USING CVXEDA LIBRARY ###################################################
phasicList_10s_reconstruct = [[] for i in range(numberOfDataSet)] #This list is a list of list containing phasic component value in 8 sec time windows after each stimulus
phasicList_20imgs_reconstruct = [[] for i in range(numberOfDataSet)]  #This list is a list of list containing phasic component value in 160 time windows (20 images)
phasicList_WinSize_reconstruct = [[] for i in range(numberOfDataSet)]  #This list is a list of list containing phasic component value in 160 time windows (20 images)

TonicList_10s_reconstruct = [[] for i in range(numberOfDataSet)]
TonicList_20imgs_reconstruct = [[] for i in range(numberOfDataSet)]
TonicList_WinSize_reconstruct = [[] for i in range(numberOfDataSet)]

phasicNoSparseList_10s_reconstruct = [[] for i in range(numberOfDataSet)]
phasicNoSparseList_20imgs_reconstruct = [[] for i in range(numberOfDataSet)]
phasicNoSparseList_WinSize_reconstruct = [[] for i in range(numberOfDataSet)]


#10s window
r_10s,p_10s,t_10s,l_10s,d_10s,e_10s,obj_10s = [[[] for j in range(len(RawDataList_10s[i]))] for i in range(numberOfDataSet)], [[[] for j in range(len(RawDataList_10s[i]))] for i in range(numberOfDataSet)], [[[] for j in range(len(RawDataList_10s[i]))] for i in range(numberOfDataSet)],[[[] for j in range(len(RawDataList_10s[i]))] for i in range(numberOfDataSet)], [[[] for j in range(len(RawDataList_10s[i]))] for i in range(numberOfDataSet)], [[[] for j in range(len(RawDataList_10s[i]))] for i in range(numberOfDataSet)],[[[] for j in range(len(RawDataList_10s[i]))] for i in range(numberOfDataSet)]

for i in range(0,numberOfDataSet):
    for j in range(0,len(RawDataList_10s[i])):
        [r_10s[i][j], p_10s[i][j], t_10s[i][j], l_10s[i][j], d_10s[i][j], e_10s[i][j], obj_10s[i][j]] = cvxEDA.cvxEDA(RawDataList_10s[i][j], delta)
        phasicList_10s_reconstruct[i].extend(array(p_10s[i][j]))
        TonicList_10s_reconstruct[i].extend(array(t_10s[i][j]))
        phasicNoSparseList_10s_reconstruct[i].extend(array(r_10s[i][j]))
        

#Images number window
r_20imgs,p_20imgs,t_20imgs,l_20imgs,d_20imgs,e_20imgs,obj_20imgs = [[[] for j in range(len(RawDataList_20imgs[i]))] for i in range(numberOfDataSet)], [[[] for j in range(len(RawDataList_20imgs[i]))] for i in range(numberOfDataSet)],[[[] for j in range(len(RawDataList_20imgs[i]))] for i in range(numberOfDataSet)], [[[] for j in range(len(RawDataList_20imgs[i]))] for i in range(numberOfDataSet)], [[[] for j in range(len(RawDataList_20imgs[i]))] for i in range(numberOfDataSet)],[[[] for j in range(len(RawDataList_20imgs[i]))] for i in range(numberOfDataSet)], [[[] for j in range(len(RawDataList_20imgs[i]))] for i in range(numberOfDataSet)]

for i in range(0,numberOfDataSet):
   for j in range(0,len(RawDataList_20imgs[i])):
      [r_20imgs[i][j],p_20imgs[i][j],t_20imgs[i][j],l_20imgs[i][j],d_20imgs[i][j],e_20imgs[i][j],obj_20imgs[i][j]] = cvxEDA.cvxEDA(RawDataList_20imgs[i][j], delta)
      phasicList_20imgs_reconstruct[i].extend(array(p_20imgs[i][j]))
      TonicList_20imgs_reconstruct[i].extend(array(t_20imgs[i][j]))
      phasicNoSparseList_20imgs_reconstruct[i].extend(array(r_20imgs[i][j]))


#Win size window
r_WinSize,p_WinSize,t_WinSize,l_WinSize,d_WinSize,e_WinSize,obj_WinSize = [[[] for j in range(len(RawDataList_WinSize[i]))] for i in range(numberOfDataSet)], [[[] for j in range(len(RawDataList_WinSize[i]))] for i in range(numberOfDataSet)],[[[] for j in range(len(RawDataList_WinSize[i]))] for i in range(numberOfDataSet)], [[[] for j in range(len(RawDataList_WinSize[i]))] for i in range(numberOfDataSet)], [[[] for j in range(len(RawDataList_WinSize[i]))] for i in range(numberOfDataSet)],[[[] for j in range(len(RawDataList_WinSize[i]))] for i in range(numberOfDataSet)], [[[] for j in range(len(RawDataList_WinSize[i]))] for i in range(numberOfDataSet)]

for i in range(0,numberOfDataSet):
   for j in range(0,len(RawDataList_WinSize[i])):
      [r_WinSize[i][j],p_WinSize[i][j],t_WinSize[i][j],l_WinSize[i][j],d_WinSize[i][j],e_WinSize[i][j],obj_WinSize[i][j]] = cvxEDA.cvxEDA(RawDataList_WinSize[i][j], delta)
      phasicList_WinSize_reconstruct[i].extend(array(p_WinSize[i][j]))
      TonicList_WinSize_reconstruct[i].extend(array(t_WinSize[i][j]))
      phasicNoSparseList_WinSize_reconstruct[i].extend(array(r_WinSize[i][j]))

############# FEATURES EXTRACTION #####################################################################################
#### List use for the 10s windows lists
# =============================================================================
# phasicList_10s = array( phasicList_10s)
# =============================================================================
phasicMeanList = [array([]) for i in range(numberOfDataSet)]
phasicSTDList = [array([]) for i in range(numberOfDataSet)]
phasicMaxList = [array([]) for i in range(numberOfDataSet)]
phasicNBpeakList = [array([]) for i in range(numberOfDataSet)]
phasicPeakAmplitude = [array([]) for i in range(numberOfDataSet)]
phasicLatency = [array([]) for i in range(numberOfDataSet)]

#### List use for the 20imgs windows lists
# =============================================================================
# phasicList_20imgs = array(phasicList_20imgs)
# =============================================================================
phasicNBpeakList_20imgs = [array([]) for i in range(numberOfDataSet)]
phasicPeakAmplitude_20imgs  = [array([]) for i in range(numberOfDataSet)]
phasicLatency_20imgs = [array([]) for i in range(numberOfDataSet)]

#### List use for the WinSize windows lists
# =============================================================================
# phasicList_WinSize = array(phasicList_WinSize)
# =============================================================================
phasicNBpeakList_WinSize = [array([]) for i in range(numberOfDataSet)]
phasicPeakAmplitude_WinSize = [array([]) for i in range(numberOfDataSet)]
phasicLatency_WinSize = [array([]) for i in range(numberOfDataSet)]

for i in range(numberOfDataSet):
   #### features extraction for 8 seconds Window
   [phasicMeanList[i], phasicSTDList[i], phasicMaxList[i], phasicNBpeakList[i], phasicPeakAmplitude[i], phasicLatency[i]]= eda.featuresExtraction(array(p_10s[i]),threshold)
   #### features extraction for 20 images Window
   [_,_,_, phasicNBpeakList_20imgs[i], phasicPeakAmplitude_20imgs[i], phasicLatency_20imgs[i]]= eda.featuresExtraction(array(p_20imgs[i]),threshold)
   #### features extraction for WinSize window
   [_,_,_, phasicNBpeakList_WinSize[i], phasicPeakAmplitude_WinSize[i], phasicLatency_WinSize[i]]= eda.featuresExtraction(array(p_WinSize[i]),threshold)
#
#DELETING PICTURE WHERE AMPLITUDE IS 0
#participant_number = 1
RMSE_estimate = []
for participant_number in range(1,5):
    RMSE_temp_participant = []
    for i in range(100):
        index = [i for i in range(len(list_arousal))]
        RMSE_error = []
        save_arousal_estimate = []
        save_amplitude_batch = []
        optimizedCoefficient = []
        len_per_batch = []
        
        # OPTIMIZING EQUATION
        #Filtering median filter
        rawphasicpeak = list(phasicPeakAmplitude[participant_number-1])
        phasicPeakAmplitude[participant_number-1] = signal.medfilt(phasicPeakAmplitude[participant_number-1])
        for c in eda.my_range(0,len(phasicPeakAmplitude[participant_number-1])-nbImage,nbImage):
            # Optimization is run batch by batch, we extract the batch of 18 images from the lists.
            amplitude_batch_temp = list(phasicPeakAmplitude[participant_number-1][c:c+18])
            save_amplitude_batch.append(list(amplitude_batch_temp))
            
            arousal_batch_temp = list(list_arousal[c:c+18])
            index_batch_temp = list(index[c:c+18])
            
            #DELETE ELEMENT EQUAL TO 0
            index_del = []
            for j in range(len(amplitude_batch_temp)):
                if amplitude_batch_temp[j] == 0:
                    index_del.append(j)
                        
            for index_l in sorted(index_del, reverse=True):
                    del amplitude_batch_temp[index_l]
            
            for index_l in sorted(index_del, reverse=True):
                    del arousal_batch_temp[index_l]
                    del index_batch_temp[index_l]
                    
            len_per_batch.append(len(amplitude_batch_temp))
            
            if (len(amplitude_batch_temp) >= 6): #check if the list is long enough for optimization (len(half)>=3)
                for h in range(300):
                    # Permutation  for the halves of the matrix
                    amplitude_shuffled_batch_temp = np.array(amplitude_batch_temp)
                    arousal_mean_shuffled_batch_temp = np.array(arousal_batch_temp)
                    index_shuffled_batch_temp = np.array(index_batch_temp)
                    
                    perm = np.random.permutation(len(amplitude_batch_temp))
                    np.take(amplitude_batch_temp,perm,axis=0,out=amplitude_shuffled_batch_temp)
                    np.take(arousal_batch_temp,perm,axis=0,out=arousal_mean_shuffled_batch_temp)
                    np.take(index_batch_temp,perm,axis=0,out=index_shuffled_batch_temp)
                    
        
                    #We cut the existing list in halves for each bin
                    halves1_amplitude, halves2_amplitude= [],[]
                    
                    half1_arousal = arousal_mean_shuffled_batch_temp[:int(len(arousal_mean_shuffled_batch_temp)/2)]
                    half2_arousal = arousal_mean_shuffled_batch_temp[int(len(arousal_mean_shuffled_batch_temp)/2):]
        
                    
                    half1_index = index_shuffled_batch_temp[:int(len(index_shuffled_batch_temp)/2)]
                    half2_index = index_shuffled_batch_temp[int(len(index_shuffled_batch_temp)/2):]
                    
                    halves1_amplitude = amplitude_shuffled_batch_temp[:int(len(amplitude_shuffled_batch_temp)/2)]
                    halves2_amplitude = amplitude_shuffled_batch_temp[int(len(amplitude_shuffled_batch_temp)/2):]
                                      
                    import scipy.optimize
                    alphas, betas, Cs = [],[],[]
          
        
                    def eq( p , arousal, index, peaksAvgMagn):
                        a,b,C = p
                        err = []
                        for i in range(len(index)):
                            f = arousal[i] * exp(a*(index[i])+b)-peaksAvgMagn[i] + C
        #                    f = log(arousal[i]) + a*(index[i]) + b - log(peaksAvgMagn[i] + C)
                            err.append(f)
                        return (err) 
                    
                    p0 = [0,0,-30]
                    
                    results = scipy.optimize.leastsq(eq, p0, args=(half1_arousal, half1_index, halves1_amplitude))
                
                    alphas.append(results[0][0])
                    betas.append(results[0][1])
                    Cs.append(results[0][2])
                                    
                alpha_mean = np.mean(alphas)
                beta_mean = np.mean(betas)
                C_mean = np.mean(Cs)
                
                alpha_std = np.std(alphas)
                beta_std = np.std(betas)
                C_std = np.std(Cs)
        
                optimizedCoefficient.append([alpha_mean,beta_mean,C_mean])
                
        #        Estimation of arousal using alpha beta and c
                Arousal_Estimate = []
                
                for i in range(len(half2_arousal)):
                    arousal_est_temp = ( halves2_amplitude[i] - C_mean) / exp((alpha_mean*(half2_index[i]))+beta_mean) #A = (AvgAmp - C)/ exp(at+b)
                    Arousal_Estimate.append((arousal_est_temp))
                save_arousal_estimate.append(Arousal_Estimate)
        
                
                for i in range(len(Arousal_Estimate)):
                    if 7.0 <= Arousal_Estimate[i]:
                        Arousal_Estimate[i] = 7
                    elif Arousal_Estimate[i] <= 1.0:
                        Arousal_Estimate[i] = 1.0    
                
                #SAGR
                error = sqrt(mean_squared_error(half2_arousal, Arousal_Estimate))
                RMSE_error.append(error)
            else:
                RMSE_error.append(math.nan)
        RMSE_error = [x for x in RMSE_error if str(x) != 'nan']
        RMSE_temp_participant.append(np.mean(RMSE_error))
    RMSE_estimate.append(np.mean(RMSE_temp_participant))

print(np.mean(RMSE_estimate))
print(np.std(RMSE_estimate))
        



#for RMSE
#A estimated in [1,7] no scaling
#crop A estimated [0,inf] to [1,7]
#RUN 1
#2.50010419842
#0.160341619814

#RUN 2
#2.50824845258
#0.148134992802

#RUN 3
#2.53244452408
#0.130149632046

#2.51664376113
#0.133092294069

#2.51728685894
#0.132115692481

#2.51837727518
#0.14642633601

#2.4997368118
#0.136503820344

#2.51671196527
#0.13831063428

#2.51683788498
#0.138266054455

#2.50672176576
#0.133978455451