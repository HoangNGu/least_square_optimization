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

with open(r'list_name.txt', 'r') as f:
    myNames = [line.strip() for line in f]

list_arousal, list_valence = [] , []
for i in range(len(myNames)):
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
##WindowSize = 1001
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

nbImage = 5 #Change this variable to change the number of images in the window
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
#%%
#DELETING PICTURE WHERE AMPLITUDE IS 0
participant_number = 4
index = [i for i in range(len(list_arousal))]
#TEST: DELETE ELEMENT EQUAL TO 0
index_del = [[] for i in range(numberOfDataSet)]
for i in range(numberOfDataSet):
    for j in range(len(phasicPeakAmplitude[i])):
        if phasicPeakAmplitude[i][j] == 0:
            index_del[i].append(j)
            
        
for i in range(numberOfDataSet):
    for index_l in sorted(index_del[i], reverse=True):
        del phasicPeakAmplitude[i][index_l]

for index_l in sorted(index_del[participant_number-1], reverse=True):
        del list_arousal[index_l]
        del index[index_l]

mean_SAGR = []
for b in range(10):
    estimate_error = []
    for o in range(50):
        #Preparing array for permutation to shuffle the data
        amplitude_shuffled, arousal_mean_shuffled, index_shuffled = [], [], []
        
        amplitude_shuffled.append(np.array(phasicPeakAmplitude[participant_number-1]))
        arousal_mean_shuffled = np.array(list_arousal)
        index_shuffled = np.array(index)
        #%% OPTIMIZING EQUATION
        global_alpha = []
        global_beta = []
        #global_gamma = []
        #global_theta = []
        for m in range(1):
        #    Permutation 
            perm = np.random.permutation(len(list_arousal))
            np.take(phasicPeakAmplitude[participant_number-1],perm,axis=0,out=amplitude_shuffled[0])
            np.take(list_arousal,perm,axis=0,out=arousal_mean_shuffled)
            np.take(index,perm,axis=0,out=index_shuffled)
            
            
            #We cut the existing list in halves (100/100)
            halves1_amplitude, halves2_amplitude= [],[]
            
            half1_arousal = arousal_mean_shuffled[:int(len(arousal_mean_shuffled)/2)]
            half2_arousal = arousal_mean_shuffled[int(len(arousal_mean_shuffled)/2):]
            half1_index = index_shuffled[:int(len(index_shuffled)/2)]
            half2_index = index_shuffled[int(len(index_shuffled)/2):]
            
            halves1_amplitude.append(amplitude_shuffled[0][:int(len(amplitude_shuffled[0])/2)])
            halves2_amplitude.append(amplitude_shuffled[0][int(len(amplitude_shuffled[0])/2):])
                
            
            
            import scipy.optimize
            alphas, betas, gammas,thetas = [],[],[],[]
            halves1_amplitude_shuffled,half1_arousal_shuffled,half1_index_shuffled = [],[],[]
            halves1_amplitude_shuffled.append(np.array(halves1_amplitude[0]))
            half1_arousal_shuffled = np.array(half1_arousal)
            half1_index_shuffled = np.array(half1_index)
            for n in range(1):
            
                perm = np.random.permutation(len(half1_arousal))
                
                np.take(halves1_amplitude[0],perm,axis=0,out=halves1_amplitude_shuffled[0])
                np.take(half1_arousal,perm,axis=0,out=half1_arousal_shuffled)
                np.take(half1_index,perm,axis=0,out=half1_index_shuffled)
                def eq( p , arousal, index, peaksAvgMagn):
                    a,b = p
                    err = []
                    for i in range(len(index)):
                        ind = index[i]
                        f = log(arousal[i])+a*(ind)+b-log(peaksAvgMagn[i])
                        err.append(f)
                    return (err) 
                
                p0 = [0,0]
                
                results = scipy.optimize.leastsq(eq, p0, args=(half1_arousal_shuffled, half1_index_shuffled, halves1_amplitude_shuffled[0]))
            
                alphas.append(results[0][0])
                betas.append(results[0][1])
        #        gammas.append(results[0][2])
        #        thetas.append(results[0][3])
                
            global_alpha.append(np.mean(alphas))
            global_beta.append(np.mean(betas))
        #    global_gamma.append(np.mean(gammas))
        #    global_theta.append(np.mean(thetas))
            
        alpha_mean = np.mean(global_alpha)
        beta_mean = np.mean(global_beta)
        #gamma_mean = np.mean(global_gamma)
        #theta_mean = np.mean(global_theta)
        
        alpha_std = np.std(global_alpha)
        beta_std = np.std(global_beta)
        #gamma_std = np.std(global_gamma)
        #theta_std = np.std(global_theta)
    
        Arousal_Estimate = []
        half2_arousal = eda.rescaling(half2_arousal,1,7)
        
        for i in range(len(half2_arousal)):
            arousal_est_temp = halves2_amplitude[0][i] / exp(alpha_mean*(half2_index[i])+beta_mean)#A = AvgAmp/ exp(at+b)
            Arousal_Estimate.append(arousal_est_temp)
        
        Arousal_Estimate = eda.rescaling(Arousal_Estimate,1,7)
        
        #SAGR
        error = eda.SAGR(Arousal_Estimate,half2_arousal)
        
        estimate_error.append(error)
        
    mean_SAGR.append(np.mean(estimate_error))