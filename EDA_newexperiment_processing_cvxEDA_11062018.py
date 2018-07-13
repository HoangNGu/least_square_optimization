import cvxEDA
from numpy import array
import numpy as np
from matplotlib import pyplot as plt
import pylab as pla
import EDA_myAPI as eda
import os
from math import log


workingdir = r"./data/"
csvfile = os.listdir(workingdir)
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

list_arousal, list_valence = [] , []
for i in range(len(myNames)):
    list_arousal.append(Arousal_mean_oasis[int(myNames[i])])
    list_valence.append(Valence_mean_oasis[int(myNames[i])])


#numberOfDataSet= len(csvfile)
numberOfDataSet = 4
len_experience = 20100
len_stimulus = 10 #length of stimulus in seconds
delta = 0.1 # delta = 1/Fs
threshold = 0.01 #threshold to define a SCR. If the amplitude of a peak is at least 0.01 it is identified as a SCR

x = [[] for i in range(numberOfDataSet)]
y = [[] for i in range(numberOfDataSet)]
ya = []
yn = []
yntemp=[]

for i in range(0,numberOfDataSet):
   eda.OpenCsvFile(csvfile[i],x[i],y[i])
   x[i] = [float(j) for j in x[i]]
   y[i] = [float(j) for j in y[i]]
   ya.append(array(y[i]))

ya[0] = ya[0][1800:1800 + len_experience] #include the 10 sec countdown. 
ya[1] = ya[1][470:470+len_experience] 
ya[2] = ya[2][440:440 + len_experience]  
ya[3] = ya[3][110:110 + len_experience] 



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



nbImage = 18 #Change this variable to change the number of images in the window
WinStart = 3 #Use in WinSize, define the elapse time between stimulus and the start of the window
WinStop = 10 #Use in WinSize, define the elapse time between stimulus and the end of the window


#Computing Arousal sum for nbImages
sumArousal = []
for i in eda.my_range(0,len(list_arousal)-nbImage,nbImage):
    sum_temp = 0
    for j in range(nbImage):
        sum_temp += list_arousal[i+j]
    sumArousal.append(sum_temp)

plt.figure()
plt.plot(sumArousal)
plt.show()
        
    

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
#plot
     
#create x values for ploting
tm = []
tmRebuilt = []
for i in range(0, numberOfDataSet):
   tm.append(eda.create_XAxis(yafilt[i],delta))
   tmRebuilt.append(eda.create_XAxis(TonicList_10s_reconstruct[i],delta))

# Four subplots, the axes array is 1-d
f, axarr = plt.subplots(4, sharex=True)

for j in range(1):
   axarr[0].plot(tm[j],yn[j])
   axarr[0].set(xlabel='t (sec)', ylabel='y(t) ($μ$S)')
   axarr[0].yaxis.label.set_size(18)
   axarr[0].xaxis.label.set_size(18)
   axarr[0].grid()
for j in range(1):
   axarr[1].plot(tmRebuilt[j],TonicList_10s_reconstruct[j])
   axarr[1].set(xlabel='t (sec)', ylabel='r(t) ($μ$S)')
   axarr[1].yaxis.label.set_size(18)
   axarr[1].xaxis.label.set_size(18)
   axarr[1].grid()
for j in range(1):
   axarr[2].plot(tmRebuilt[j],phasicNoSparseList_10s_reconstruct[j])  
   axarr[2].set(xlabel='t (sec)', ylabel='p(t) ($μ$S)')
   axarr[2].yaxis.label.set_size(18)
   axarr[2].xaxis.label.set_size(18)
   axarr[2].grid()
for j in range(1):
   axarr[3].plot(tmRebuilt[j],phasicList_10s_reconstruct[j])
   axarr[3].set(xlabel='t (sec)', ylabel='y(t) ($μ$S)')
   axarr[3].yaxis.label.set_size(18)
   axarr[3].xaxis.label.set_size(18)
   axarr[3].grid()
      
#axarr[0].set_title('(a)')
#axarr[1].set_title('(b)')
#axarr[2].set_title('(c)')
#axarr[3].set_title('Sparse SMNA driver of phasic component')


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

for i in range(0, numberOfDataSet):
   #### features extraction for 8 seconds Window
   [phasicMeanList[i], phasicSTDList[i], phasicMaxList[i], phasicNBpeakList[i], phasicPeakAmplitude[i], phasicLatency[i]]= eda.featuresExtraction(array(p_10s[i]),threshold)
   #### features extraction for 20 images Window
   [_,_,_, phasicNBpeakList_20imgs[i], phasicPeakAmplitude_20imgs[i], phasicLatency_20imgs[i]]= eda.featuresExtraction(array(p_20imgs[i]),threshold)
   #### features extraction for WinSize window
   [_,_,_, phasicNBpeakList_WinSize[i], phasicPeakAmplitude_WinSize[i], phasicLatency_WinSize[i]]= eda.featuresExtraction(array(p_WinSize[i]),threshold)
  
############# Plot results ############################################################################################################

############# Number of peaks in window ###############################################################################################

tmphasicAmplitude,tmphasicAmplitude_20imgs,tmphasicAmplitude_WinSize,tmPeaks, tmPeaks_20imgs, tmPeaks_WinSize, tmphasicLatency, tmphasicLatency_20imgs, tmphasicLatency_WinSize= [], [], [], [], [], [],[], [], []
for i in range(0,numberOfDataSet):
   #Create the xAxis fo peaks number
   tmPeaks.append(pla.arange(1., len(phasicNBpeakList[i])+1.))
   tmPeaks_20imgs.append(pla.arange(1., len(phasicNBpeakList_20imgs[i])+1.))
   tmPeaks_WinSize.append(pla.arange(1., len(phasicNBpeakList_WinSize[i])+1.))
   #Create the xAxis for the phasic Latency
   tmphasicLatency.append(pla.arange(1., len(phasicLatency[i])+1.))
   tmphasicLatency_20imgs.append(pla.arange(1., len(phasicLatency_20imgs[i])+1.))
   tmphasicLatency_WinSize.append(pla.arange(1., len(phasicLatency_WinSize[i])+1.))
   #Create the xAxis for the phasic Amplitude
   tmphasicAmplitude.append(pla.arange(1., len(phasicPeakAmplitude[i])+1.))
   tmphasicAmplitude_20imgs.append(pla.arange(1., len(phasicPeakAmplitude_20imgs[i])+1.))
   tmphasicAmplitude_WinSize.append(pla.arange(1., len(phasicPeakAmplitude_WinSize[i])+1.))
   
#### 10s window 
f2, axarr = plt.subplots(4, sharex=True)  #plot nb of peak when using 10s window size
for i in range(numberOfDataSet):
   axarr[i].bar(tmPeaks[i],phasicNBpeakList[i])
   axarr[i].set_title('dataset'+str(i+1)+'_10s Window size_number of peak')  

#### _20imgs
f3, axarr = plt.subplots(4, sharex=True)  #plot nb of peak when using 20 imgs window size
for i in range(numberOfDataSet):
   plt.ylim(0,200)
#   set lim to 200
   axarr[i].bar(tmPeaks_20imgs[i],phasicNBpeakList_20imgs[i])
   axarr[i].set_title('dataset'+str(i+1)+'_'+str(nbImage)+'imgs_number of peak')

###### _WinSize
f4, axarr = plt.subplots(4, sharex=True)  #plot nb of peak when using WinSize window size
for i in range(numberOfDataSet):
   axarr[i].bar(tmPeaks_WinSize[i],phasicNBpeakList_WinSize[i])
   axarr[i].set_title('dataset'+str(i+1)+'_WinSize_number of peak')


#Cumulative sum of nb of peaks for nBimage

cumulativeSum = []
for i in range(numberOfDataSet):
    cumulativeSum.append(np.cumsum(phasicNBpeakList[i]))
    
x = [i for i in range(len(list_arousal))]

plt.figure()
for i in range(numberOfDataSet):
    plt.step(x,cumulativeSum[i], label='participant'+str(i+1))
plt.legend()
plt.grid()
plt.show()

############### LATENCY: plot of all first peak latency in the window size
##10sec Window
## create x values
#f5, axarr = plt.subplots(4, sharex = True)
#for i in range(numberOfDataSet):
#   axarr[i].bar(tmphasicLatency[i],phasicLatency[i])
#   axarr[i].set_title('Latency_dataset'+str(i+1)+'_10sWindow')
#
#
##20 images Window
#f6, axarr = plt.subplots(4, sharex = True)
#for i in range(numberOfDataSet):
#   axarr[i].bar(tmphasicLatency_20imgs[i],phasicLatency_20imgs[i])
#   axarr[i].set_title('Latency_dataset'+str(i+1)+'_'+str(nbImage)+' imgs Window')
#
#
##WinSize (!) the latency plot here is between the first SCR peaks and the start of the window (WinStart) and not
##the latency between the first peaks of the window and the stimulus
#f7, axarr = plt.subplots(4, sharex = True)
#for i in range(numberOfDataSet):
#   axarr[i].bar(tmphasicLatency_WinSize[i],phasicLatency_WinSize[i])
#   axarr[i].set_title('Latency_dataset'+str(i+1)+'_WinSize')

################ MEAN LATENCY for a windowSizeLatency img window size ########################################
##SEPARATE THE LIST IN SECTION TO CALCULATE MEAN, STD,... OF FEATURES IN EVERY CHUNKS OF DATA AT THE BEGINING, MIDDLE, END, ... of EXP
#windowSizeLatency =  5#Number of images used to compute the mean of every first peak latency 
#
#chunkLatency = [[] for i in range(numberOfDataSet)]
#for k in range(0,numberOfDataSet):
#   chunkLatency[k] = [phasicLatency[k][i:i+windowSizeLatency] for i in range(0, len(phasicLatency[k]), windowSizeLatency)] #We cut the original data into chunk of 50 images
#
#chunkLatency1mean = [[] for i in range(numberOfDataSet)]
#
#for j in range(numberOfDataSet):
#   for i in range(len(chunkLatency[j])-1) :
#      chunkLatency1mean[j].append(array(chunkLatency[j][i]).mean()) #We add to chunkLatency1mean the average latency of the first peak of each 50 images chunks. For example, chunkLatencymean[0]
#                                                                     # contains the average latency of the first peaks of the first 50 images.
#
#
#tmphasicLatencymean = []
#for i in range(numberOfDataSet):
#   tmphasicLatencymean.append(pla.arange(1., len(chunkLatency1mean[i])+1.))
#
#
#f8, axarr = plt.subplots(4, sharex = True)
#for i in range(numberOfDataSet):
#   axarr[i].bar(tmphasicLatencymean[i],chunkLatency1mean[i])
#   axarr[i].set_title('Latency_mean_dataset'+str(i+1)+'_'+str(windowSizeLatency)+'img Window')


############### AMPLITUDE: plot of all first peak amplitude in the window size
##10sWindows
#f9, axarr = plt.subplots(4, sharex = True)
#for i in range(numberOfDataSet):
#   axarr[i].bar(tmphasicAmplitude[i],phasicPeakAmplitude[i])
#   axarr[i].set_title('Amplitude_dataset_'+str(i+1)+'_10secWin')
##5images Windows
#f10, axarr = plt.subplots(4, sharex = True)
#for i in range(numberOfDataSet):
#   axarr[i].bar(tmphasicAmplitude_20imgs[i],phasicPeakAmplitude_20imgs[i])
#   axarr[i].set_title('Amplitude_dataset_'+str(i+1)+'_'+str(nbImage)+'window')
##Winsize Window
#f11, axarr = plt.subplots(4, sharex = True)
#for i in range(numberOfDataSet):
#   axarr[i].bar(tmphasicAmplitude_WinSize[i],phasicPeakAmplitude_WinSize[i])
#   axarr[i].set_title('Amplitude_dataset_'+str(i+1)+'WinSize')
#plt.show()


############### Scatterplot - x_axis time , y_axis Arousal or Valence, size of dot proportional to amplitude,latency, ...
#### 5 images window
####################Size of dot proportional to amplitude
#
#for i in range(3):
#    plt.figure()
#    plt.title("Dot size proportional to average amplitude in 5s picture bin")
#    plt.xlabel('timebin')
#    plt.ylabel('Arousal')
#    x = pla.arange(1., len(meanarousal_window)+1.)
#    c_map = ['seismic','Blues','Greens']
##    trr = phasicPeakAmplitude_20imgs[i]
##    normalized = (trr-min(trr))/(max(trr)-min(trr))
#    s = [(k)*1000 for k in phasicPeakAmplitude_20imgs[i]]
#    plt.scatter(x, meanarousal_window, s, marker= 'o', alpha=0.5, cmap = c_map[0],edgecolors='black')
#plt.show()
####################Size of dot proportional to latency
#plt.figure()
#plt.title("Dot size proportional to average latency of first peaks in 5s picture bin")
#plt.xlabel('timebin')
#plt.ylabel('Arousal')
#x = pla.arange(1., len(meanarousal_window)+1.)
#for i in range(3):
#    s = [(k)*100 for k in phasicLatency_20imgs[i]]
#    plt.scatter(x, meanarousal_window, s, marker= 'o', alpha=0.5, cmap = 'seismic',edgecolors='black')
#plt.colorbar()
#plt.show()
####################Size of dot proportional to number of peak
#plt.figure()
#plt.title("Dot size proportional to average nb of peaks in 5s picture bin")
#plt.xlabel('timebin')
#plt.ylabel('Arousal')
#x = pla.arange(1., len(meanarousal_window)+1.)
#for i in range(3):   
#    s = [(k)*100 for k in phasicNBpeakList_20imgs[i]]
#    plt.scatter(x, meanarousal_window, s, marker= 'o', alpha=0.5, cmap = 'seismic',edgecolors='black')
#plt.colorbar()
#plt.show()
#
#### per picture
####################Size of dot proportional to amplitude
#
#for i in range(3):   
#    plt.figure()
#    plt.title("Dot size proportional to average amplitude per picture")
#    plt.xlabel('timebin')
#    plt.ylabel('Arousal')
#    x = pla.arange(1., len(list_arousal)+1.)
#    colors = ['r','g','b']
##    trr = phasicPeakAmplitude[i]
##    normalized = (trr-min(trr))/(max(trr)-min(trr))
#    s = [(k)*1000 for k in phasicPeakAmplitude[i]]
#    plt.scatter(x, list_arousal, s, marker= 'o', alpha=0.5, cmap = 'seismic',edgecolors='black')
#plt.show()
####################Size of dot proportional to latency
#plt.figure()
#plt.title("Dot size proportional to average latency of first peaks per picture")
#plt.xlabel('timebin')
#plt.ylabel('Arousal')
#x = pla.arange(1., len(list_arousal)+1.)
#colors = ['r','g','b']
#for i in range(3):   
##    trr = phasicLatency[i]
##    normalized = (trr-min(trr))/(max(trr)-min(trr))
#    s = [(k)*100 for k in phasicLatency[i]]
#    plt.scatter(x, list_arousal, s,c = s, marker= 'o', alpha=0.5, cmap = 'seismic',edgecolors='black')
#plt.colorbar()
#plt.show()
####################Size of dot proportional to number of peak
#plt.figure()
#plt.title("Dot size proportional to average nb of peaks per picture")
#plt.xlabel('timebin')
#plt.ylabel('Arousal')
#x = pla.arange(1., len(list_arousal)+1.)
#for i in range(3):   
##    trr = phasicNBpeakList[i]
##    normalized = (trr-min(trr))/(max(trr)-min(trr))
#    s = [(k)*100 for k in phasicNBpeakList[i]]
#    plt.scatter(x, list_arousal, s,c = s, marker= 'o', alpha=0.5, cmap = 'seismic',edgecolors='black')
#plt.colorbar()
#plt.show()





