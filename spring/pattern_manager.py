import numpy as np
from spring_core.pattern import downscale
import matplotlib.pyplot as plt
import matplotlib

def resize_pattern(data, mask, newsize):
    
    tempdata = np.copy(data)        
    tempdata = tempdata+1e3
    tempdata[mask>0.5]=-1
    
    newdata =   downscale(tempdata, [newsize, newsize])    
    
    newmask = np.zeros(newdata.shape)
    newmask[newdata<0]=1
    
    newdata = newdata-1e3

    return newdata, newmask


def cut(data, center, size):
    
    frame = int(data.shape[0]*2)
    tempsize = int(data.shape[0]+2*frame)
    tempdata = np.zeros([tempsize, tempsize])
    tempdata[frame:data.shape[0]+frame, frame:data.shape[1]+frame] = data
    tempcenter = [center[0]+frame, center[1]+frame]
    newdata = np.copy(tempdata[tempcenter[0]-size//2:tempcenter[0]+size//2, tempcenter[1]-size//2:tempcenter[1]+size//2])
    
    return newdata


class Pattern:
    """
    Class for the handling of diffraction patterns. 
    
    :param pattern: 2D numpy array containing the diffraction pattern
    :param mask: 2D numpy array containing the mask. Indexes where ``mask[i,j]=0`` indicate good pixels. Indexes where ``mask[i,j]=1`` are instead excluded. the *mask* and the *pattern* must have the same dimensions.
    :param center: list of length 2 containing the y and x coordinates of the center of the *pattern*
    :param cropsize: The pattern is cropped to a shape ``[cropsize,cropsize]``. If ``cropsize = None``, it is set equal to ``max(pattern.shape)``.
    :param pid: List of identifiers for the diffraction pattern. For example, if pattern has been acquired in run 34 with pulse-id 197843921, it is convenient to set ``pid=[34,197843921]``.
    :param size: If not ``None``, the pattern is rescaled to a final shape ``[size,size]`` after cropping.
    :param satvalue: Indicate which is the saturation value of the detector. If different from ``None``, pixels above this value are masked. 
    :param metadata: Additional experimental metadata that is convenient to save with the reconstruction, provided as a dictionary.  
    
    """
    def __init__(self, pattern, mask, center: list[2], cropsize: int = None, pid: list =[0], size: int = None, satvalue: float = None, metadata={}):
        
        self.origpattern = np.copy(pattern)
        self.origmask = np.copy(mask)
        self.pid = pid
        self.metadata = metadata
        data = np.copy(pattern)   
        pixmask = np.copy(mask)
        
        
        pixmask[np.isnan(data)] = 1
        
        ######
        if cropsize is None:
            cropsize = np.amax(data.shape)        
        if size is None:
            size = cropsize
        
        data = cut(data, center, cropsize)
        pixmask = cut(pixmask, center, cropsize)

    
        satmask = np.zeros(pixmask.shape)
        if not (satvalue is None):
            satmask[data>satvalue] = 1
            pixmask[satmask==1] = 1
            
            
            
        data[pixmask==1] = -1
        data[satmask==1] = -1

        resizedmask = np.copy(pixmask)
        if size<cropsize:
            data, resizedmask=resize_pattern(data, pixmask, size)
            satmask=downscale(satmask, [size, size]) 
            pixmask=downscale(pixmask, [size, size]) 
            
        
        pixmask[pixmask>0.25]=1
        pixmask[pixmask<=0.25]=0
        pixmask[np.logical_and(pixmask<0.5, resizedmask>0.25)]=1
        
        satmask[satmask>0.05]=1
        satmask[satmask<=0.05]=0

        data[data<0]=0
        data[pixmask==1]=-1
        data[satmask==1]=-2
        
        self.pattern = data

        
    def get(self):
        """
        Get the prepared pattern as a 2D numpy array.
        """
        return self.pattern
     
    def get_pid(self):
        return self.pid

    def get_metadata(self):
        return self.metadata
    
    
    def plot(self, ax: matplotlib.axes.Axes = None, cmap='inferno', circles: int = 0, cropsize: int = -1 ):
        """
        Plot the diffraction pattern, in logarithmic colorscale.
        
        :param ax: matplotlib axes where the pattern is plotted. If none, axes are created within the function and the data is automatically plotted. 
        :param cmap: colormap for the plot. It must be a valid `matplotlib` colormap
        :param circles: number of circles to draw on the pattern, centered in the central coordinates of the pattern matrix. It serves as a guide for the eyes to identify the correct pattern center. 
        :param cropsize: The pattern is cropped to the given linear dimension for plotting. 
        """
        
        if ax is None:
            fig, axx = plt.subplots(1,1,figsize=(4,4))
        
        else:
            axx = ax
        axx.clear()
        patternplot = np.copy(self.pattern)
        mask = patternplot<0
        patternplot[mask]=np.nan
        vmin = np.nanmean(patternplot[:patternplot.shape[0]//8,::patternplot.shape[0]//8])*0.5
        patternplot[mask]=vmin
        patternplot[patternplot<vmin]=vmin
        
        
        if cropsize>0 and cropsize<patternplot.shape[0]:
            patternplot = cut(patternplot, [patternplot.shape[0]//2,patternplot.shape[1]//2] , cropsize)
        
        axx.imshow(patternplot, norm=matplotlib.colors.LogNorm(vmin=vmin), cmap=cmap)
        
        
        if circles>0:
            maxrad = patternplot.shape[0]/2
            radii = np.linspace(0, maxrad, num=circles+2)[1:-1]
            for rad in radii:
                circ = matplotlib.patches.Circle((maxrad,maxrad),rad, ec='lime', fill=False)
                axx.add_patch(circ)        
        
        
        if ax is None:
            titlestring=''
            if len(self.pid)>0:
                titlestring = "{:d}".format(self.pid[0])
                for i in range(1, len(self.pid)):
                    titlestring+= " - {:d}".format(self.pid[i])
            
            axx.set_title(titlestring)

            
            


            plt.show()
        

    def plot_mask(self, ax : matplotlib.axes.Axes = None, cmap='viridis'):
        """
        Plot the mask of the pattern. Masked pixels get a value of 1, while saturated pixels get a value of 2. Valid pixels have a value of 0.
        
        :param ax: matplotlib axes where the mask is plotted. If none, axes are created within the function and the data is automatically plotted. 
        :param cmap: colormap for the plot. It must be a valid `matplotlib` colormap
        """
        
        
        if ax is None:
            fig, axx = plt.subplots(1,1,figsize=(4,4))
        
        else:
            axx = ax
        axx.clear()
        maskplot = np.zeros(self.pattern.shape)
        maskplot[self.pattern==-1]=1
        maskplot[self.pattern==-2]=2
        axx.imshow(maskplot, cmap=cmap)
        if ax is None:
            titlestring=''
            if len(self.pid)>0:
                titlestring = "{:d}".format(self.pid[0])
                for i in range(1, len(self.pid)):
                    titlestring+= " - {:d}".format(self.pid[i])
            
            axx.set_title(titlestring)

            plt.show()
        
        
        
        
        
        
