import os
import h5py
import numpy as np
from spring.settings_manager import Settings
from spring.stats_manager import Stats

from spring_core import pattern
import matplotlib.pyplot as plt
import matplotlib
import PIL.Image
from copy import deepcopy

def hsl_to_hsv(h, s, l):
    l[l==0]=1e-8
    v = (2*l + s*(1-np.abs(2*l-1)))/2
    #print(v)
    s = np.nan_to_num(2*(v-l)/(v))
    #print(s)
    #s = np.zeros(v.shape)
    #s[v>0] = np.nan_to_num(2*(v[v>0]-l[v>0])/(v[v>0]))
    #s[v==0] = 0
    return h, s, v


def colorize(z):
  n,m = z.shape

  A = (np.angle(z) + np.pi) / (2*np.pi)
  A = (A + 0.5) % 1.0
  B = np.ones(z.shape)*1.
  H = (abs(z)/ np.abs(z).max())**0.66 *0.7
  
  A,B,H = hsl_to_hsv(A, B, H)
  A = A*255
  B = B*255
  H = H*255
  
  image = PIL.Image.fromarray(np.stack((A, B, H), axis=-1).astype(np.uint8), "HSV") # HSV has range 0..255 for all channels
  image = image.convert(mode="RGB")

  return np.array(image)


class Result:
    """
    Handle a MPR result.
    
    :param pid: If specified, load the reconstruction file corresponding to the given pid. Otherwise, an empty reconstruction is initialized.
    :param workdir: Specify the path of the folder where the reconstruction is saved
    :param tag: Specify a tag assigned to the reconstruction (given to the `MPR.run` method as parameter).
    """
    def __init__(self, pid: list = None, workdir: str ='./', tag: str=''):
        
        if pid is None:
            
            self.pattern = None
            self.density_best=None
            self.support_best=None
            self.density_avg=None
            self.support_avg=None
            self.pid = None
            self.stats = None
            self.settings = None
            self.metadata = None
            self.density_all = None
            self.support_all = None
            self.proc_id = os.getpid()
            self.step=None
            self.total=None
        else:
            self.load(pid, workdir, tag)
        
        self.x_min, self.y_min = None, None
        self.x_max, self.y_max = None, None
        
  
     
    def set(self, pattern, density_best, support_best, density_avg, support_avg, pid, stats, settings, metadata={}, density_all=None, support_all = None, step=None, total=None ):
        self.pattern = pattern
        self.density_best=density_best
        self.support_best=support_best
        self.density_avg=density_avg
        self.support_avg=support_avg
        self.pid = pid
        self.stats = stats
        self.settings = settings
        self.metadata = metadata
        self.density_all = density_all
        self.support_all = support_all
        self.step=step
        self.total=total
        self.x_min=None
        
    def copy(self, inres):
        self.pattern = deepcopy(inres.pattern)
        self.density_best = deepcopy(inres.density_best)
        self.support_best = deepcopy(inres.support_best)
        self.density_avg= deepcopy(inres.density_avg)
        self.support_avg=deepcopy(inres.support_avg)
        self.pid = deepcopy(inres.pid)
        self.stats = deepcopy(inres.stats)
        self.settings = deepcopy(inres.settings)
        self.metadata = deepcopy(inres.metadata)
        self.density_all = deepcopy(inres.density_all)
        self.support_all = deepcopy(inres.support_all)
        self.step=inres.step
        self.total=inres.total
        self.x_min = None
        
        
    def save(self, workdir: str ='./', tag: str=''):
        """
        Save the reconstruction on disk.
        
        :param workdir: path to the ourput folder
        :param tag: tag to add to the filename       
        """
        fname = self.get_filename(self.pid, tag)
        
        fullname = os.path.join(workdir, fname)
        
        try:
            self.savefile(fullname)
        except:
            print("Failed to save file {}".format(fullname))
        
    def savefile(self, fname: str):
        """
        Save the reconstruction with the specified filename.
        
        :param filename: name of the output file.
        """
        
            
        with h5py.File(fname, "w") as h5file:
            #print("File opened")
            h5file.create_dataset("pid", data=self.pid)
            h5file.create_dataset("settings", data=self.settings.getjson())

            h5file.create_dataset("pattern", data=np.nan_to_num(self.pattern))
            h5file.create_dataset("density_best", data=np.nan_to_num(self.density_best))
            h5file.create_dataset("support_best", data=np.nan_to_num(self.support_best))   

            h5file.create_dataset("density_avg", data=np.nan_to_num(self.density_avg))
            h5file.create_dataset("support_avg", data=np.nan_to_num(self.support_avg))  
            
            if self.density_all is not None:
                h5file.create_dataset("density_all", data=self.density_all)
                
            if self.support_all is not None:
                h5file.create_dataset("support_all", data=self.support_all)
            
            for key in self.metadata.keys():
                #print(key, metadata[key])
                h5file.create_dataset("metadata/"+str(key), data=self.metadata[key])

            if self.stats is not None:
                
                statsdict = self.stats.get()
                              
                for key1 in statsdict.keys():
                    for key2 in statsdict[key1].keys():
                        h5file.create_dataset("stats/{}/{}".format(key1,key2), data=statsdict[key1][key2])
                    
            
            h5file.create_dataset("process/generation", data=self.step)
            h5file.create_dataset("process/totgenerations", data=self.total)
            h5file.create_dataset("process/proc_id", data=self.proc_id)

            h5file.close()

        #os.replace(outfiletemp, outfile) 

        
        return
    

    def get_filename(self, pid:list, tag=""):
        """
        Get the filename for a specific pid.
        
        :param pid: list of integers specifing the pid
        :param tag: tag to add to the filename 
        
        """
        
        outname=''
        for pp in pid:
            outname+="{:d}_".format(pp)
        if len(tag)>0:
            outname+=tag+"_"
        #if save_steps:
            #filename+="step{:04d}_".format(self.generation)
        outname += "rec.h5"

        return outname
    
    def load(self, pid: list, workdir: str ='./', tag: str=''):
        fname = self.get_filename(pid, tag)
        
        fullname = os.path.join(workdir, fname)
        
        try:
            self.loadfile(fullname)
        except:
            print("Failed to load file {}".format(fullname))
        
        
    def loadfile(self, fname: str):
        """
        Load a reconstruction with the specified filename.
        
        :param fname: name of the reconstruction file
        """
        
        with h5py.File(fname, "r") as h5file:
            #print("File opened")
            self.pattern = h5file['pattern'][()]
            self.density_best=h5file['density_best'][()]
            self.support_best=h5file['support_best'][()]
            self.density_avg=h5file['density_avg'][()]
            self.support_avg=h5file['support_avg'][()]
            self.pid = h5file['pid'][()]
                        
            self.settings = Settings()
            setstring=str(h5file['settings'][()].decode('UTF-8'))
            #print(setstring)
            self.settings.setjson(jsonstr=setstring)
            
            self.metadata = {}
            
            if "metadata" in h5file.keys():
                for k in h5file["metadata"].keys():
                    metadata[k]  =h5file["metadata"][k][()]
                #print(k, metadata[k])
            
            statsdict = {}
            for k1 in h5file["stats"].keys():
                statsdict[k1] = {}
                for k2 in h5file["stats"][k1].keys():
                    statsdict[k1][k2] = h5file["stats"][k1][k2][()]
            self.stats = Stats()
            self.stats.set(statsdict)
                
            if "density_all" in h5file.keys():
                self.density_all = h5file['density_all'][()]
            if "support_all" in h5file.keys():
                self.support_all = h5file['support_all'][()]

            self.step=h5file["process/generation"][()]
            self.total=h5file["process/totgenerations"][()]
            self.proc_id = h5file["process/proc_id"][()]
        
        self.x_min, self.y_min = None, None
        self.x_max, self.y_max = None, None
    
    #def plot(self):
        
    def print_status(self):
        """
        Print textual information about the reconstruction status.
        """
        print("Generation {:d}/{:d}".format(self.step, self.total))
        self.stats.print()

        

    def get_crop(self):
        
        coords = np.argwhere(self.support_best)
        self.x_min, self.y_min = coords.min(axis=0)
        self.x_max, self.y_max = coords.max(axis=0)
        if self.x_max-self.x_min<self.y_max-self.y_min:
            centeral = (self.x_max+self.x_min)//2
            self.x_min = centeral - (self.y_max-self.y_min)//2
            self.x_max = centeral + (self.y_max-self.y_min)//2
        elif self.x_max-self.x_min>self.y_max-self.y_min:
            centeral = (self.y_max+self.y_min)//2
            self.y_min = centeral - (self.x_max-self.x_min)//2
            self.y_max = centeral + (self.x_max-self.x_min)//2
        self.margins = int( (self.x_max-self.x_min)*0.15)
        
    def crop(self, data):
        if self.x_min is None:
            self.get_crop()
        return np.pad(data[self.x_min : self.x_max + 1, self.y_min : self.y_max + 1], self.margins)
            
            
    def get_density(self, crop : bool = True):
        """
        Get the reconstructed average density as numpy array.
        
        :param crop: If ``True``, crop the matrix to the support region.
        
        :returns: 2-D complex numpy array
        """
        if crop:
            return self.crop(self.density_avg)
        else:
            return np.copy(self.density_avg)
    
    def get_support(self, crop : bool =True):
        """
        Get the reconstructed average support as numpy array.
        
        :param crop: If ``True``, crop the matrix to the support region.
        
        :returns: 2-D numpy array
        """
        if crop:
            return self.crop(self.support_avg)
        else:
            return np.copy(self.support_avg)
    
    def get_density_best(self, crop : bool =True):
        """
        Get the reconstructed best density as numpy array.
        
        :param crop: If ``True``, crop the matrix to the support region.
        
        :returns: 2-D complex numpy array
        """
        if crop:
            return self.crop(self.density_best)
        else:
            return np.copy(self.density_best)
    
    def get_support_best(self, crop : bool =True):
        """
        Get the reconstructed best support as numpy array.
        
        :param crop: If ``True``, crop the matrix to the support region.
        
        :returns: 2-D numpy array
        """
        if crop:
            return self.crop(self.support_best)
        else:
            return np.copy(self.support_best)
        
        
    def get_stats(self):
        """
        Get the statistics of the reconstruction process as a nested dictionary.
        """
        return self.stats.get()
    
    def get_settings(self):
        """
        Get the settings with which the reconstruction was performed. Returns a `spring.Settings` object.
        """
        return self.settings
    

    def get_pattern(self):
        """
        Get the diffraction pattern used for the reconstruction as a 2D numpy array.
        """
        return self.pattern
    
    def get_residuals(self):
        """
        Get the residuals of the reconstruction, i.e. the difference between the experimental and reconstructed pattern.
        """
        recpattern=np.abs(np.fft.fftshift(np.fft.fft2(np.conj(self.get_density(crop=False))))[:,:])**2
        return np.abs(self.pattern-recpattern)

    def get_rec_pattern(self):
        """
        Get the retrieved pattern calculated from the reconstruction.
        """
        recpattern=np.abs(np.fft.fftshift(np.fft.fft2(np.conj(self.get_density(crop=False))))[:,:])**2
        return recpattern

        
    def plot_residuals(self, ax, cmap='inferno'):
        """
        Plot the residuals (difference between experimental and reconstructed pattern) on the given input axes (instance of `matplotlib.axes.Axes`).
        
        :param ax: `matplotlib.axes.Axes` object on which the data is plotted.
        :param cmap: a valid `matplotlib` colormap       
        """
                
        ax.clear()
        ax.axis("off")
        residuals = self.get_residuals()
        patternplot = np.copy(self.pattern)
        mask = patternplot<0
        patternplot[mask]=np.nan
        vmin = np.nanmean(patternplot[:patternplot.shape[0]//8,::patternplot.shape[0]//8])*0.5
        vmax = np.nanmax(patternplot)
        minresiduals = vmin
        residuals[mask] = minresiduals
        residuals[residuals<vmin] = minresiduals
        ax.imshow(residuals, norm=matplotlib.colors.LogNorm(vmin=minresiduals, vmax=vmax), cmap=cmap)
    
    def plot_pattern(self, ax, cmap='inferno'):
        """
        Plot the diffraction pattern on the given input axes (instance of `matplotlib.axes.Axes`).
        
        :param ax: `matplotlib.axes.Axes` object on which the data is plotted.
        :param cmap: a valid `matplotlib` colormap       
        """
                
        ax.clear()
        ax.axis("off")
        patternplot = np.copy(self.pattern)
        mask = patternplot<0
        patternplot[mask]=np.nan
        vmin = np.nanmean(patternplot[:patternplot.shape[0]//8,::patternplot.shape[0]//8])*0.5
        patternplot[mask]=vmin
        patternplot[patternplot<vmin]=vmin
        ax.imshow(patternplot, norm=matplotlib.colors.LogNorm(vmin=vmin), cmap=cmap)
        
        
    def plot_rec_pattern(self, ax, cmap='inferno'):
        """
        Plot the reconstructed diffraction pattern on the given input axes (instance of `matplotlib.axes.Axes`).
        
        :param ax: `matplotlib.axes.Axes` object on which the data is plotted.
        :param cmap: a valid `matplotlib` colormap       
        """
                
        ax.clear()
        ax.axis("off")
        patternplot = np.copy(self.pattern)
        mask = patternplot<0
        patternplot[mask]=np.nan
        vmin = np.nanmean(patternplot[:patternplot.shape[0]//8,::patternplot.shape[0]//8])*0.5
        vmax = np.amax(patternplot)
        recpattern = self.get_rec_pattern()
        #patternplot[mask]=vmin
        #patternplot[patternplot<vmin]=vmin
        ax.imshow(recpattern, norm=matplotlib.colors.LogNorm(vmin=vmin), cmap=cmap)
        
    def plot_density(self, ax, cmap='bone', energy=None, angle=None):
        """
        Plot the retrieved density module on the given input axes (instance of `matplotlib.axes.Axes`).
        
        :param ax: `matplotlib.axes.Axes` object on which the data is plotted.
        :param cmap: a valid `matplotlib` colormap       
        :param energy: photon energy in eV. If both energy and angle are given, a scale-bar is shown. 
        :param angle: maximum scattering angle at the edge of the detector, in degrees. If both energy and angle are given, a scale-bar is shown.     
        """
        ax.clear()
        ax.axis("off")
        densplot= self.get_density()
        ax.imshow(np.abs(densplot), cmap=cmap)
        
        if energy is not None:
            if angle is not None:

                theta = angle/180.*np.pi
                wavelength = 1240./energy
                kx = 2.*np.pi/wavelength * np.sin(theta)
                dx = np.pi/kx
                #print("dx:" , dx)
                
                totpixels = densplot.shape[0]
                totsize = totpixels/5.*dx
                tottensize = np.ceil(totsize/20.)*20.
                tottenpixels = tottensize/dx
                
                xstart = totpixels/26.
                xend = xstart+tottenpixels
                ystart = totpixels/10.
                yend = ystart
                
                textx = (xstart+xend)/2.
                texty = yend-totpixels/100
                
                textcolor = matplotlib.cm.get_cmap(cmap)(1.)
                
                ax.plot([xstart, xend], [ystart, yend], "-", color=textcolor, linewidth=4)
                ax.text(textx,texty, str(int(tottensize)) + "nm", color=textcolor, ha="center", va = "bottom", fontsize=14)
            
        

    def plot_phase(self, ax):
        """
        Plot the retrieved density phase on the given input axes (instance of `matplotlib.axes.Axes`).
        
        :param ax: `matplotlib.axes.Axes` object on which the data is plotted.
        """
        ax.clear()
        ax.axis("off")
        densplot= self.get_density()
        ax.imshow(colorize(densplot))
        
    def plot_support(self, ax, cmap='viridis'):
        """
        Plot the retrieved support function on the given input axes (instance of `matplotlib.axes.Axes`).
        
        :param ax: `matplotlib.axes.Axes` object on which the data is plotted.
        :param cmap: a valid `matplotlib` colormap       
        """
        ax.clear()
        ax.axis("off")
        suppplot= self.get_support()
        ax.imshow(np.abs(suppplot), cmap=cmap)

    
    
    def getaxes(self, stats):
        pltsize = 3
        ncols = 3
        sepsize = 0.2
        height_ratios=[1.]
        if stats:
            height_ratios+=[sepsize, 0.7]
            
        nrows = len(height_ratios)
        
        
        
        fig = plt.figure(figsize=(pltsize*ncols*height_ratios[0],np.sum(height_ratios)*pltsize))
        gs0 = fig.add_gridspec(nrows = len(height_ratios), ncols=1, height_ratios=height_ratios)
        
        gsplots = gs0[0].subgridspec(nrows=1, ncols=3)
        axplot = []
        axplot.append(fig.add_subplot(gsplots[0]))
        axplot.append(fig.add_subplot(gsplots[1]))
        axplot.append(fig.add_subplot(gsplots[2]))
        np.asarray(axplot)
        if stats:
            ns = 4
            width_ratios = [0.3,1]*ns
            
            gstats = gs0[2].subgridspec(nrows=1, ncols=len(width_ratios), width_ratios=width_ratios)
            axstats = [fig.add_subplot(gstats[1])]
            axstats.append(fig.add_subplot(gstats[3]))
            axstats.append(fig.add_subplot(gstats[5]))
            axstats.append(fig.add_subplot(gstats[7]))
            np.asarray(axstats)
        else:
            axstats = None
        
        #plt.tight_layout()
        plt.subplots_adjust(left=0., bottom=0., right=1., top=0.95, wspace=0.02, hspace=0.02)

        return fig, axplot, axstats
    
    #def getstataxes(self):
        #pltsize = 3
        #nrows = 1
        #ncols = 1
        #return plt.subplots(nrows = nrows,ncols = ncols, figsize=(ncols*pltsize,nrows*pltsize))
        
    
    def fillaxes(self, ax, cmap: str = 'bone', cmap_pattern: str ='inferno'):
        
        restitle = "Gen {:d}/{:d}".format(int(self.step), int(self.total), int(self.proc_id))
        titlestring=''
        if len(self.pid)>0:
            titlestring = "{:d}".format(self.pid[0])
            for i in range(1, len(self.pid)):
                titlestring+= " - {:d}".format(self.pid[i])
       
        self.plot_pattern(ax[0], cmap_pattern)
        self.plot_density(ax[1], cmap)
        self.plot_phase(ax[2])
        
        
        ax[0].set_title(titlestring)
        ax[1].set_title(restitle)
        
    
    #def plot(self, cmap: str = 'bone', cmap_pattern: str ='inferno'):
        #"""
        #Plot an overview of the reconstruction in a Matplotlib figure.
        
        #:param cmap: A valid matplotlib colormap to assign to the density module.
        #:param cmap_pattern: A valid matplotlib colormap to assign to the diffraction pattern.
        #"""

        #fig, ax = self.getaxes()
        #self.fillaxes(ax, cmap, cmap_pattern)
        #plt.show()

    def plot(self, stats: bool = False, cmap: str = 'bone', cmap_pattern: str ='inferno'):
        """
        Plot an overview of the reconstruction in a Matplotlib figure.

        :param stats: If true, plot also the behavior of relevant parameters and statistics during the reconstruction process.
        :param cmap: A valid matplotlib colormap to assign to the density module.
        :param cmap_pattern: A valid matplotlib colormap to assign to the diffraction pattern.
        """

        fig, ax, axstats = self.getaxes(stats)
        self.fillaxes(ax, cmap, cmap_pattern)
        if stats:
            self.fillstatsaxes(axstats)
        
        plt.show()
    

        
    def fillstatsaxes(self, ax):
        
        #fig, ax = self.getstataxes()
        
        colors = {
            "best": "#a33f3f",
            "mean": "#42a83b",
            "avg": "#6746a8",
            "all": "#b5e8b7",
            "OS" : "#d3c73d",
            "replace" : '#ea8975'
        }

        for aa in ax:
            aa.clear()
            

        locstats = self.stats.get()
                                
        error_best = np.array(locstats['best']['error'])
        error_avg = np.array(locstats['average']['error'])
        iters = np.array(locstats['global']['generation'])
        replace = np.array(locstats['global']['replacement'])*100
        OS=np.array(locstats['global']['OS'])
        error_all = np.array(locstats['all']['error'])
        
        error_worse = [np.amax(error_all[ii,:]) for ii in range(error_all.shape[0])]
        error_mean = np.mean(error_all, axis=1)

        
        ax[0].fill_between(iters, error_best, error_worse, color=colors["all"])
        ax[0].plot(iters, error_mean, color = colors["mean"])
        ax[1].plot(iters, error_avg, color = colors["avg"])
        ax[2].plot(iters, replace, color = colors["replace"])
        ax[3].plot(iters, OS, color = colors["OS"])
 
        
        ax[0].set_title("Errors")
        ax[1].set_title("Error of the Average")
        ax[2].set_title("Replacement factor")
        ax[3].set_title("Oversampling")

        for aa in ax:
            aa.spines.top.set_visible(False)
            aa.spines.right.set_visible(False)

        
        for aa in ax:
            aa.spines.bottom.set_position(('outward', 4))
            aa.spines.left.set_position(('outward', 4))
            
        for aa in ax:
            aa.set_xlim([0, np.amin([iters[-1]+10, self.total])])
            
            
            
    def get_prtf(self):
        """
        Calculate the Phase Retrieval Transfer Function on the current reconstructed density. 
        
        :returns: a tuple of two arrays. The first is the array of q values, normalized between 0 and 1. To get the actual corresponding q values, one should multiply this array by q_max. The second array contains the PRTF values.
        
        """
        
        recpattern=np.abs(np.fft.fftshift(np.fft.fft2(np.conj(self.get_density(crop=False))))[:,:])
        origpattern = np.copy(self.pattern)
        origpattern[origpattern<0] = 0
        origpattern = origpattern**0.5
        ratio = -1.*np.ones(origpattern.shape)
        ratio[origpattern>0] = recpattern[origpattern>0]/origpattern[origpattern>0]
        ratio[ratio>1] = 1
        
        #dtheta = 2.
        center = [self.pattern.shape[0]/2, self.pattern.shape[1]/2]
        nangles = self.pattern.shape[0]//2 
        anglemax = 1 
        dtheta = anglemax / nangles
        angles = np.arange(0, anglemax, dtheta)
        radprof = np.array(pattern.get_radial_profile(ratio, dtheta, center, angles, [0,360]))
        angles = angles[radprof>0]
        radprof = radprof[radprof>0]
        
        
        return angles, radprof
        


    def plot_prtf(self, ax=None):
        """
        Plot the Phase Retrieval Transfer Function, calculated on the reconstructed density.
        
        :param ax: `matplotlib.axes.Axes` object on which the PRTF is plotted. If `None`, a new figure and new axes are created and the data is plotted.
        """
        axesgiven=True
        if ax is None:
            fig, ax = plt.subplots(1,1)
            axesgiven=False
        
        angles, values = self.get_prtf()
        ax.clear()
        ax.plot([0,1],[0,0], ":", alpha=0, c='w')
        ax.plot([0,1],[1,1], ":", alpha=0, c='w')
        ax.plot(angles, vals)
        ax.plot([0,1],[np.exp(-1), np.exp(-1)], ":", c='black', alpha=0.3)
        ax.set_ylabel("PRTF")
        ax.set_xlabel(r"$q_{max}$")
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.spines.bottom.set_position(('outward', 4))
        ax.spines.left.set_position(('outward', 4))
        
        if not axesgiven:
            plt.show()
            
        
