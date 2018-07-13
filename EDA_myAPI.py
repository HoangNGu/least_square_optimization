import csv
import pylab as pla
from numpy import array
from PeakDetect import peakdet
import numpy as np
import signal

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # #  GLOBAL FUNCTIONS # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#### Global variable ####
global numberOfDataSet

#### Range for loop ####

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

#### Open directory ####

def OpenCsvFile (csvfile, x, y):
   f=open(csvfile,'r') # opens file for reading
   empty_lines = 0
   reader = csv.reader(f, delimiter=';')
   for i in range(20):
      next(reader)
   print('this is reader')
   print(reader)
   for row in reader:
    if not row:
        empty_lines += 1           
    else:       
        x.append(row[0])
        y.append(row[1])

def OpenCsvFile_oasis (csvfile):
   f=open(csvfile,'r') # opens file for reading
   reader = csv.reader(f, delimiter=',')
   Name,Valence_mean,Valence_SD,Valence_N,Arousal_mean,Arousal_SD,Arousal_N = [],[],[],[],[],[],[]
   next(reader)
   for row in reader:
      Name.append(row[1])
      Valence_mean.append(float(row[4]))
      Valence_SD.append(float(row[5]))
      Valence_N.append(float(row[6]))
      Arousal_mean.append(float(row[7]))
      Arousal_SD.append(float(row[8]))
      Arousal_N.append(float(row[9]))
   f.close()
   return (Name,Valence_mean,Valence_SD,Valence_N,Arousal_mean,Arousal_SD,Arousal_N)
#### Create x axis for a list ####

def create_XAxis(y,delta):
   tm = pla.arange(1.,len(y)+1.) * delta
   return(tm)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # FILTERING # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def medianFilter (y,Windowsize):
   ya= signal.medfilt(y, Windowsize)
   return(ya)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # NORMALIZATION # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#### Z-score normalisation ####

def zscoreNormalisation(y):
   yn = (y-y.mean())/ y.std()
   return(yn)

#### Min-max normalization ####

def minmaxNormalisation(y):
   yn = (y-y.min())/(y.max() - y.min())
   return(yn)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # WINDOW SIZING # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def WindowSizing(start,stop,step, ListToCut):
## this function create a sliding window startint at t0, of size step and going from start to step,
## incrementing of step each time  
   ListPhaseIter = []
   ListOut = []
   for i in my_range(start,stop-step, step): #begining of each stimulus, whole length, we start after 10 secs countdown
      for j in range(i,i+step): #time window of 8s after each stimulus
         ListPhaseIter.append(float(ListToCut[j]))
      ListOut.append(ListPhaseIter)
      ListPhaseIter = []
   return(ListOut)

def WindowSizing_ImagNumber(start,stop,ImgNumber, ListToCut):
   step = ImgNumber*10*10 #Each stimulus is 8seconds long * 10 to get number of points
   ListPhaseIter = []
   ListOut = []
   for i in my_range(start-1,stop-step,step): #begining of each stimulus, whole length, we start after 10 secs countdown
      for j in my_range(i,i+step,1): #time window of 8s after each stimulus
         ListPhaseIter.append(float(ListToCut[j]))
      ListOut.append(ListPhaseIter)
      ListPhaseIter = []
   return(ListOut)

def WindowSizing_WinStartStop(start,stop,step,WinStart,WinStop, ListToCut):
   ListPhaseIter = []
   ListOut = []
   WinStart = 10*WinStart #seconds to number of points
   WinStop = 10*WinStop
   WinSize = WinStop - WinStart
   for i in my_range(start-1,stop-step,step): #begining of each stimulus, whole length, we start after 10 secs countdown
      for j in my_range(i+WinStart,i+WinSize,1): #time window of 8s after each stimulus
         ListPhaseIter.append(float(ListToCut[j]))
      ListOut.append(ListPhaseIter)
      ListPhaseIter = []
   return(ListOut)

def WindowSizing_DifferentStep(ListStep, ListToCut):
    x = 0
    y = 0
    ListAppend = [] #contains only question phase
    ListAppendStory = [] #contains only story phase
    ListAppendWholeList = [] #contains both story and question phase
    for i in range(0,len(ListStep)):
        x = y + ListStep[i]
        ListAppendStory.append(ListToCut[int(y):int(x)])
        ListAppendWholeList.append(ListToCut[int(y):int(x)])
        y = x + 17.
        ListAppend.append(ListToCut[int(x):int(y)])
        ListAppendWholeList.append(ListToCut[int(x):int(y)])
    return ListAppend, ListAppendStory, ListAppendWholeList
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # FEATURES EXTRACTION # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def featuresExtraction(List,threshold):
    '''This function return the following features from List:Mean,Standard Deviation(STD), the Maximimum of amplitude of the peaks, The number of peaks, The amplitude of the peaks, the Latency of the first peak'''
    mean = []
    STD = []
    MAX = []
    PeakNb = []
    PeakAmplitude = []
    FirstPeakAmplitude = []
    Latency = []
    for k in range(0, len(List)):
        second_onwards = False
        maxtab = []
        mintab = []
        mean.append(List[k].mean())
        STD.append(List[k].std())
        MAX.append(List[k].max())
        Latency.append(float((onsetDetection(List[k],threshold)[0])/10))
        maxtab,mintab = peakdet(List[k],threshold) #Consider a peak if onset of at least 0.01
        PeakNb.append(len(maxtab)) #len(maxtab) is equal to the number of peaks 
        
        if not (list(maxtab)):
            PeakAmplitude.append(0)
            FirstPeakAmplitude.append(0)
                    
        else:
            templist = []
            for j in range(0,len(maxtab)):
                templist.append(maxtab[j][1]) #Stock the peak amplitude value
                if not second_onwards:
                    FirstPeakAmplitude.append(maxtab[0][1])
                    second_onwards = True
            if (templist):
                PeakAmplitude.append(np.mean(templist))
            
    return mean,STD,MAX,PeakNb,PeakAmplitude,Latency #,FirstPeakAmplitude
      
def frequencyOfPeaks(List,time, threshold):
    '''time is in seconds'''
    ListFrequency= []
    PeakNb = []
    for k in range(0,len(List)):
        maxtab,mintab = peakdet(List[k],threshold) #Consider a peak if onset of at least 0.01
        PeakNb.append(len(maxtab))
    for i in range(0,len(List)):
        ListFrequency.append(float((time*PeakNb[i])/(len(List[i]*10))))
    return ListFrequency

def onsetDetection(List, threshold):
    global xOnset
    global yOnset
    i = 0
    '''return the onset (higher than the threshold) of the first peak in the list'''
    while(List[i]<threshold and i<len(List)-1):
        i = i+1
    if(i == len(List)-1 and List[i]<threshold): #no peak detected
        xOnset = 0
        yOnset = 0
    else:
        xOnset = float(i)
        yOnset = float(List[i])
    
    return float(xOnset),float(yOnset)

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]



def SAGR(predictions,labels):
    '''Computes Sign Agreement Metric (SAGR)'''
    sumF = 0
    n = len(predictions)
    for i in range(n):
        if(np.sign(predictions[i])==np.sign(labels[i])):
            #if prediction/labels == 0, sign return 0
            sumF += 1
    return(sumF/n)
    
    
def rescaling(listN,currentmin,currentmax):
    '''Rescales the List around 0'''
    neutral = (currentmax+currentmin)/2
    for i in range(len(listN)):
        listN[i] = listN[i]-neutral
    return listN  




def check(value):
    if -1.0 <= value <= 1.0:
        return True
    return False



def OpenCsvFile_Gaped(csvfile):
    f= open(csvfile,'r')
    reader = csv.reader(f, delimiter=',')
    Name,Valence_mean,Arousal_mean = [],[],[]
    next(reader)
    for row in reader:
        Name.append(row[0])
        Valence_mean.append(float(row[1]))
        Arousal_mean.append(float(row[2]))
    f.close()
    return(Name,Valence_mean,Arousal_mean)


def normalizerange(listN,newmin,newmax):
#    normalize(listN)# normalized between 0 and 1, so translate and strect
    for i in range(len(listN)):
        listN[i] = (newmax-newmin) * listN[i] + newmin
#        return listN

def normalize(listN):
    '''Performs a min-max normalization'''
    # normalizes between 0 and 1
    maxList = max(listN)
    minList = min(listN)
    for i in range(len(listN)):
        listN[i] = (listN[i]-minList)/(maxList - minList)
    return listN


   
   
