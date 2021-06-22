import sys, os, re, copy, time
import numpy as np

class calcaverageclass:
    def __init__(self):

        self.reset()
        self.set_str_format()
        self.c4_smalln = [0.0, 0.0, 0.7978845608028654, 0.8862269254527579, 0.9213177319235613, 0.9399856029866251, 0.9515328619481445]

    def c4(self,n):
        #http://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
        if n<=6:
            return(self.c4_smalln[n])
        else:
            return(1.0 - 1.0/(4.0*n) - 7.0/(32.0*n*n) - 19.0/(128.0*n*n*n))

    def reset(self):
        self.mean = None
        self.mean_err = None
        self.stdev = None
        self.stdev_err = None
        self.X2norm = None

        # if speed is an issue, you can disable stdev and X2 calculation if not needed in errorcut
        self.calc_stdev_X2_flag = True

        self.Nchanged=self.Nused=self.Nskipped=0

        self.converged = False
        self.i = 0

        self.use = None
        self.clipped = None

    def set_str_format(self,format_floats='%f',format_ints='%d'):
        self.str_format1 = "mean:%s(%s) stdev:%s(%s) Nchanged:%s Nused:%s Nskipped:%s" % (format_floats,format_floats,format_floats,format_floats,format_ints,format_ints,format_ints)
        self.str_format2 = "mean:%s(%s) stdev:%s X2norm:%s Nchanged:%s Nused:%s Nskipped:%s" % (format_floats,format_floats,format_floats,format_floats,format_ints,format_ints,format_ints)

    def __str__(self):
        if self.mean != None and self.stdev != None and self.mean_err != None:
            if self.stdev_err!=None:
                #s = "i:%02d mean:%f(%f) stdev:%f(%f) Nchanged:%d Nused:%d Nskipped:%d" % (self.i,self.mean,self.mean_err,self.stdev,self.stdev_err,self.Nchanged,self.Nused,self.Nskipped)
                format = "i:%02d "+self.str_format1
                s = format % (self.i,self.mean,self.mean_err,self.stdev,self.stdev_err,self.Nchanged,self.Nused,self.Nskipped)
            else:
                #s = "i:%02d mean:%f(%f) stdev:%f Nchanged:%d Nused:%d Nskipped:%d" % (self.i,self.mean,self.mean_err,self.stdev,self.Nchanged,self.Nused,self.Nskipped)
                format = "i:%02d "+self.str_format2
                if self.X2norm!=None:
                    s = format % (self.i,self.mean,self.mean_err,self.stdev,self.X2norm,self.Nchanged,self.Nused,self.Nskipped)
                else:
                    if self.stdev!=None:
                        s = format % (self.i,self.mean,self.mean_err,self.stdev,0.0,self.Nchanged,self.Nused,self.Nskipped)
                    else:
                        s = format % (self.i,self.mean,self.mean_err,0.0,0.0,self.Nchanged,self.Nused,self.Nskipped)
            return(s)
        else:
            print(" mean:",self.mean," mean_err:",self.mean_err," stdev:",self.stdev," Nchanged:",self.Nchanged," Nused:",self.Nused," Nskipped:",self.Nskipped)
            return('ERROR')

    def results2texttable(self,t,key=None,meancol='mean',meanerrcol='mean_err',stdevcol='stdev',stdeverrcol='stdev_err',Nusedcol='Nused',Nskippedcol='Nskipped',Pskippedcol='Pskipped',Pgoodcol='Pgood',convergedcol='converged',iterationcol='i',format='%.2f',initcols=False):
        if initcols or (not (meancol in t.cols)):
            t.configcols([meancol,meanerrcol,stdevcol,stdeverrcol],'f',format,visible=1)
            t.configcols([Pgoodcol,Pskippedcol],'f','%.3f',visible=1)
            t.configcols([Nusedcol,Nskippedcol,convergedcol,iterationcol],'d',visible=1)
        if key==None:
            key = t.newrow({})
        #    key = t.newrow({meancol:self.mean,meanerrcol:self.mean_err,stdevcol:self.stdev,stdeverrcol:self.stdev_err,Nusedcol:self.Nused,Nskippedcol:self.Nskipped,iterationcol:self.i})
        #else:
        #    t.add2row(key,{meancol:self.mean,meanerrcol:self.mean_err,stdevcol:self.stdev,stdeverrcol:self.stdev_err,Nusedcol:self.Nused,Nskippedcol:self.Nskipped,iterationcol:self.i})
        t.add2row(key,{meancol:self.mean,meanerrcol:self.mean_err,stdevcol:self.stdev,stdeverrcol:self.stdev_err,Nusedcol:self.Nused,Nskippedcol:self.Nskipped,iterationcol:self.i})
        if self.Nused!=None and (self.Nused+self.Nskipped)>0:
            t.setentry(key,Pskippedcol,100.0*self.Nskipped/(self.Nused+self.Nskipped))
        if self.Ntot>0:
            t.setentry(key,Pgoodcol,100.0*self.Nused/self.Ntot)
        if self.converged:
            t.setentry(key,convergedcol,1)
        else:
            t.setentry(key,convergedcol,0)
        return(key)

    def calcaverage_errorcut(self,data,mask=None,noise=None,mean=None,Nsigma=None,medianflag=False,verbose=0):
        if mean!=None:
            self.mean=mean

        # If 3sigma cut and second iteration (i.e. we have a stdev from the first iteration), skip bad measurements.
        if Nsigma!=None and self.stdev!=None:
            useold = copy.deepcopy(self.use)

            # old
            #self.use = np.where(abs(data-self.mean)<=Nsigma*noise,True,False)
            #if mask!=None:
            #    self.use = self.use &  (np.logical_not(mask))
            #self.Nchanged = len(data[np.where(useold!=self.use)])


            # New
            if mask is None:
                self.use = np.where(abs(data-self.mean)<=Nsigma*noise,True,False)
            else:
                self.use = np.where((abs(data-self.mean)<=Nsigma*noise) & (np.logical_not(mask)),True,False)

            self.Nchanged = len(np.where(useold!=self.use)[0])

            del useold
        else:
            # use all data.
            if mask is None:
                self.use = np.ones(data.shape, dtype=bool)
            else:
                self.use = np.ones(data.shape, dtype=bool) &  (np.logical_not(mask))
            self.Nchanged = 0

        if data[self.use].size-1.0>0:
            if medianflag:
                self.mean = np.median(data[self.use])
                #self.mean_err = None
                self.stdev =  np.sqrt(1.0/(data[self.use].size-1.0)*np.sum(np.square(data[self.use] - self.mean)))/self.c4(data[self.use].size)
                self.mean_err = self.stdev/np.sqrt(data[self.use].size-1)
                self.X2norm = None
            else:
                c1 = np.sum(1.0*data[self.use]/np.square(noise[self.use]))
                c2 = np.sum(1.0/np.square(noise[self.use]))
                self.mean = c1/c2
                self.mean_err = np.sqrt(1.0/c2)
                if self.calc_stdev_X2_flag and data[self.use].size>1:
                    self.X2norm = 1.0/(data[self.use].size-1.0)*np.sum(np.square((data[self.use] - self.mean)/noise[self.use]))
                else:
                    self.X2norm = None

            if self.calc_stdev_X2_flag and data[self.use].size>1:
                self.stdev = np.sqrt(1.0/(data[self.use].size-1.0)*np.sum(np.square(data[self.use] - self.mean)))/self.c4(data[self.use].size)
            else:
                self.stdev = None
        else:
            self.mean = None
            self.mean_err = None
            self.X2norm = None
            self.stdev = None


        # old
        #self.Nused = len(data[self.use])
        #self.Ntot = data.size
        #self.Nskipped = data.size-self.Nused
        #if mask!=None:
        #    self.Nskipped -= len(np.where(mask>0)[0])


        self.Nused = data[self.use].size
        self.Ntot = data.size
        self.Nskipped = data.size-self.Nused
        if not (mask is None):
            #old
            #self.Nskipped -= (data.size-data[np.logical_not(mask)].size)

            # new: more efficient
            self.Nskipped -= len(np.where(mask)[0])

    def calcaverage_sigmacut(self,data,mask=None,mean=None,stdev=None,Nsigma=None,fixmean=None,medianflag=False,verbose=0):
        if mean!=None:
            self.mean=mean
        if stdev!=None:
            self.stdev=stdev
        if fixmean!=None:
            self.mean=fixmean

        # If 3sigma cut and second iteration (i.e. we have a stdev from the first iteration), skip bad measurements.
        if Nsigma!=None and self.stdev!=None:
            useold = copy.deepcopy(self.use)

            #self.use = np.where(abs(data-self.mean)<=Nsigma*self.stdev,True,False)
            #
            #if mask!=None:
            #    self.use = self.use &  (np.logical_not(mask))

            if mask is None:
                self.use = np.where(abs(data-self.mean)<=Nsigma*self.stdev,True,False)
            else:
                self.use = np.where((abs(data-self.mean)<=Nsigma*self.stdev) & (np.logical_not(mask)),True,False)

            # old
            # self.Nchanged = len(data[np.where(useold!=self.use)])

            # New
            self.Nchanged = len(np.where(useold!=self.use)[0])
            del useold
        else:
            # use all data.
            if mask is None:
                self.use = np.ones(data.shape, dtype=bool)
            else:
                self.use = np.ones(data.shape, dtype=bool) &  (np.logical_not(mask))


            self.Nchanged = 0

        data4use = data[self.use]

        if fixmean==None:
            if medianflag:
                self.mean = np.median(data4use)
            else:
                self.mean = np.mean(data4use)

        if data4use.size>1:
            #print '1,',time.asctime()
            self.stdev = np.sqrt(1.0/(data4use.size-1.0)*np.sum(np.square(data4use - self.mean)))/self.c4(data4use.size)
            #print '2,',time.asctime()
        else:
            self.stdev = None


                # np.std = sqrt(1/N*sum((x-x_average)^2), NOT 1/(N-1)
                #self.stdev = np.std(data4use)

            #else:
            #    if data4use.size>1:
            #        self.stdev = np.sqrt(data4use.size/(data4use.size-1.0)*np.mean((data4use - self.mean)*(data4use - self.mean)))
            #    else:
            #        self.stdev = None

        self.Nused = data4use.size
        self.Ntot = data.size
        if self.Nused>0 and (not(self.stdev is None)):
            self.mean_err = 1.0*self.stdev/np.sqrt(self.Nused)
            self.stdev_err = 1.0*self.stdev/np.sqrt(2.0*self.Nused)
        else:
            self.mean_err = None
            self.mean = None
            self.stdev_err  = None
            self.stdev = None
            self.Nskipped = 0
            return(1)

        self.Nskipped = data.size-self.Nused
        if not (mask is None):
            # old: inefficent memory use
            # self.Nskipped -= (data.size-data[np.logical_not(mask)].size)

            # new: more efficient
            self.Nskipped -= len(np.where(mask)[0])

        del data4use

    def calcaverage_sigmacutloop(self,data,mask=None,noise=None,Nsigma=3.0,Nitmax=10,fixmean=None,verbose=0,saveused=False,median_firstiteration=True):
        """
        mask must have same dimensions than data. If mask[x]=True, then data[x] is not used.
        noise must have same dimensions than data. If noise != None, then the error weighted mean is calculated.
        if saveused, then self.use contains array of datapoints used, and self.clipped the array of datapoints clipped
        median_firstiteration: in the first iteration, use the median instead the mean. This is more robust if there is a population of bad measurements
        """

        self.reset()
        #self.i=0
        #self.converged=False
        while ((self.i<Nitmax) or (Nitmax==0)) and (not self.converged):
            medianflag = median_firstiteration and (self.i==0) and (Nsigma!=None)
            if noise is None:
                self.calcaverage_sigmacut(data,mask=mask,Nsigma=Nsigma,fixmean=fixmean,medianflag=medianflag,verbose=verbose)
            else:
                self.calcaverage_errorcut(data,mask=mask,noise=noise,Nsigma=Nsigma,medianflag=medianflag,verbose=verbose)
            #if verbose>=2:
             #   print(self.__str__())
            # Not converged???
            if self.stdev==None or self.stdev==0.0 or self.mean==None:
                self.converged=False
                break
            # Only do a sigma cut if wanted
            if Nsigma == None or Nsigma == 0.0:
                self.converged=True
                break
            # No changes anymore? If yes converged!!!
            if (self.i>0) and (self.Nchanged==0):
                self.converged=True
                break
            self.i+=1

        if saveused:
            if mask is None:
                self.clipped = np.logical_not(self.use)
            else:
                self.clipped = np.logical_not(self.use) &  np.logical_not(mask)
        else:
            del(self.use)
        return(0)
