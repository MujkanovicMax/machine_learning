import numpy as np
import matplotlib.pyplot as plt

def gauss(x,mu,sig):
    
    return 1/np.sqrt(2*np.pi*sig*sig)*np.exp(-(x-mu)*(x-mu)/(2*sig*sig))

###ex1###
###a1###


#data = np.loadtxt('../LMU_DA_ML_19Adv/rohr1.dat')

#mean = np.nanmean(data)
#std = np.nanstd(data)

#fig,ax = plt.subplots(1,5,figsize=(15,5))
#hist = ax[0].hist(data)
#ax[0].set_title("rohr1 hist")



#x,y = np.loadtxt('../LMU_DA_ML_19Adv/rohr2.dat',unpack=True)

#xmean = np.nanmean(x)
#ymean = np.nanmean(y)

#xstd = np.nanstd(x)
#ystd = np.nanstd(y)
#corr = np.correlate(x,y)

#histx = ax[1].hist(x)
#ax[1].set_title("rohr2 xhist")
#histy = ax[2].hist(y)
#ax[2].set_title("rohr2 yhist")

#pointplot = ax[3].plot(x,y,".b")
#ax[3].set_title("rohr2 pointplot")

#hist2d = ax[4].hist2d(x,y)
#ax[4].set_title("rohr2 2d hist")

#fig.tight_layout()

#plt.savefig("ex1a1.pdf")
#plt.close("all")

####a2###

#data = np.loadtxt("../LMU_DA_ML_19Adv/faithful.csv", delimiter=",",skiprows=1)
#dur=data[:,1]
#wait=data[:,2]
#meandur = np.nanmean(dur)
#meanwait = np.nanmean(wait)
#stddur = np.nanstd(dur)
#stdwait = np.nanstd(wait)

#gaussdur = gauss(dur,meandur,stddur)
#gausswait = gauss(wait,meanwait,stdwait)

#fig,ax = plt.subplots(1,3,figsize=(15,5))

#histdur = ax[0].hist(dur)
##ax[0].plot(dur,gaussdur)
#ax[0].set_title("duration hist")

#histwait = ax[1].hist(wait)
##ax[1].plot(wait,gausswait)
#ax[1].set_title("wait-time hist")

#scatter = ax[2].scatter(dur,wait)
#ax[2].set_title("duration-wait scatter")

#plt.savefig("ex1a2.pdf")
#plt.tight_layout()
#plt.close("all")

###a3###

# load dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
import pandas as pd
df = pd.DataFrame( cancer.data, columns=cancer.feature_names)

df.info()
df.describe()

fig,ax=plt.subplots(5,6,figsize=(30,20))

ax = np.ravel(ax)

for i,col in enumerate(df.columns):
    ax[i].plot(range(df.shape[0]),df[col],linewidth=0.4)
    ax[i].set_title(str(col))
    ax[i].grid(True,ls=":",lw=0.2)
    
plt.tight_layout()
plt.savefig("cancerdata.pdf")
plt.close("all")
    







