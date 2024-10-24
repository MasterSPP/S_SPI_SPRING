import numpy as np
from spring.result import Result
import multiprocessing as mp
from IPython.display import display, clear_output
import threading
import time
import matplotlib.pyplot as plt
import fnmatch
from typing import Union

class LivePlot():
    def __init__(self, livequeue, newreceived, liverunning, result = None):
        self.livequeue = livequeue
        self.newreceived = newreceived
        self.liverunning = liverunning
        self.result = result
        self.fig = None
        self.jdisplay=None
    
    def getnext(self):
        if self.newreceived.wait(1):
            self.result = self.livequeue.get()
            self.newreceived.clear()

        
        
    def plot(self,  cmap: str = 'bone', cmap_pattern: str ='inferno'):

        if self.result is not None:
            if self.fig is None:
                self.fig, self.ax , self.axstats = self.result.getaxes(stats=True)
                #plt.ion()
            self.result.fillaxes(self.ax, cmap, cmap_pattern)
            self.result.fillstatsaxes(self.axstats)
            if self.jdisplay is None:
                self.jdisplay = display(self.fig, display_id=True) 
            self.jdisplay.update(self.fig)
        
        
    def run(self, cmap: str = 'bone', cmap_pattern: str ='inferno'):

        while self.liverunning.is_set():
            self.getnext()
            self.plot(cmap, cmap_pattern)
        self.jdisplay=None
        self.fig = None
        print("Live plotting stopped.")
    
    
    






def runplot(livequeue, newreceived, liverunning):
    
    try:
        print("Starting live plotting")
        lp = LivePlot(livequeue, newreceived, liverunning)
        lp.run()
        
    except KeyboardInterrupt:
        liverunning.clear()
        return
        
    return



def logger(livelogging, msgqueue, nlines = 10):
    while livelogging.is_set():
        msg = msgqueue.get()
        outmsg = '\n'.join(msg.splitlines()[-nlines:])
        clear_output(wait=True)
        print(outmsg)


def runproc(mpr, save_every, queue, logqueue, newevent, stopevent, doneeven):
    mpr.set_channels(queue, logqueue, newevent, stopevent,doneeven)
    return mpr.run(save_every)  

#class printer(str):
    #def __repr__(self):
       #return self

class _Hypervisor:
    """
    Object for handling asynchronous executions.

    .. warning:: 
        A new instance of this class shouldn't be created by the user. Use the global instance :data:`spring.hypervisor` to use the class methods.
    
    .. note:: 
        All class methods are non-blocking, apart those explicitly made for this scope like :attr:`wait`.
    
    """
    def __init__(self):        
        self.liveploton = False
        self.plotthread = None
        self.newevent = mp.Event()
        self.stopevent = mp.Event()
        self.doneevent = mp.Event()
        self.queue = mp.Queue()
        
        self.runpid = None
        self.runtag = None
        
        self.livequeue = mp.Queue()
        self.logqueue = mp.Queue()
        self.msgqueue = mp.Queue()
        
        self.newreceived = mp.Event()
        self.liverunning = mp.Event()
        self.livelogging = mp.Event()
        
        
        self.result = None
        self.process = None
        self.fig = None
        self.updatethread = threading.Thread(target=self.update)
        self.updatethread.setDaemon(True)
        self.updatethread.start()


        self.runnerthread = threading.Thread(target=self.runner)
        self.runnerthread.setDaemon(True)
        self.runnerthread.start()        
        self.runnerlock = threading.Lock()
        self.joblist = []
        
        self.loggerlock = threading.Lock()
        self.loglist = ["\n"]*50        
        
        self.loggerthread = threading.Thread(target=self.loggerdaemon)
        self.loggerthread.setDaemon(True)
        self.loggerthread.start()        
        self.loggerpublisherthread = threading.Thread(target=self.loggerpublisher)
        self.loggerpublisherthread.setDaemon(True)
        self.loggerpublisherthread.start()        

        
      
        
        
    def loggerdaemon(self):
        while True:
            newmsg = self.logqueue.get()
            self.loggerlock.acquire()
            self.loglist.pop(0)
            self.loglist.append(newmsg)
            self.loggerlock.release()          
            #time.sleep(0.1)
            
    def progressbar(self, fraction):
        
        perc = "{:3.0f}%".format(fraction*100)
        nsteps = 10
        ndone = int(np.round(fraction*nsteps))
        pstring = ''.join(["#" if i<ndone else "-" for i in range(nsteps)])
        pstring = "[" + pstring + "]" + perc
        return pstring
        
        
    def loggerpublisher(self):
        while True:
            
            nqueued = len(self.joblist)
            is_running = False
            if self.process is not None:
                is_running = self.process.is_alive()

            status = "- - - - - - - - - - - - - - - - - - - - -\n"
            
            if is_running:
                
                pidstr = self.get_pid_string(self.runpid, self.runtag)
                frac=0
                if self.result is not None:
                    if (np.array(self.result.pid) == np.array(self.runpid)).all():
                        frac=self.result.step/self.result.total
                
                status += "RUNNING "+ pidstr + " " + self.progressbar(frac)
            else:
                status += "IDLE"

            if nqueued>0:
                status+= " ({:d} queued)".format(nqueued)
            
            self.loggerlock.acquire()
            outlist = ''.join(self.loglist)
            self.loggerlock.release()    
            self.msgqueue.put(outlist + status)
            time.sleep(0.5)
            
    #def logger(self):
        ##while self.livelogging.is_set():
            ###clear_output(wait=True)
            ##self.loggerlock.acquire()
            ##outlist = ''.join(self.loglist)
            ##self.loggerlock.release()          
            ##outmsg = '\n'.join(outlist.splitlines()[-10:])
            ##if self.logdisplay is None:
                ##self.logdisplay = display('',display_id=True) 
            ##self.logdisplay.update()
            ##print(outmsg)
            ##time.sleep(1)
        ##self.logdisplay = None
        
        #print("Live logging stopped.")
    
    def runner(self):
        while True:
            time.sleep(0.5)
            if (self.process is None) or (not self.process.is_alive()):
                newargs=None
                self.runnerlock.acquire()
                if len(self.joblist)>0:
                    newargs = self.joblist[0]
                    self.joblist.pop(0)
                self.runnerlock.release()
                if newargs is not None:
                    self.reset()
                    self.run(*newargs)
                    self.process.join()
                    
    
    def get_pid_string(self, pid, tag):
        pidstr = "-".join(["{:d}".format(p) for p in pid])
        if len(tag)>0:
            pidstr += "-"+tag
        return pidstr
        
        
    
    def info(self):
        """
        Print information on the running process and the execution queue.
        Job names are formatted as ``"pid[0]-pid[1]- .. -pid[n]-tag"``.
        """
        
        is_running = False
        if self.process is not None:
            is_running = self.process.is_alive()
        
        if is_running:
            pidstr = self.get_pid_string(self.runpid, self.runtag)

            print("Running: "+pidstr)
        else:
            print("Idle.")

        self.runnerlock.acquire()
        nqueued = len(self.joblist)
        if nqueued>0:
            print("Execution queue ({:d} total):".format(nqueued))
            for ij in range(nqueued):
                pidstr = self.get_pid_string(self.joblist[ij][0].pid,self.joblist[ij][0].tag)
                print("  {:d}: ".format(ij) + pidstr)
        else:
            print("Empty queue.")
        self.runnerlock.release()

        
    def append(self,  mpr, save_every: int =-1):
        """
        Append a :class:`spring.MPR` at the back of the execution queue. It behaves like :meth:`spring.MPR.runasync`. If the execution queue is empty and no reconstruction process is currently running, the execution of the reconstruction is immediately started.
        
        :param mpr: an instance of :class:`spring.MPR` for which the execution via the :meth:`spring.MPR.run` has to be scheduled.
        :param save_every: The current status of the reconstruction is saved every *save_every* generations. If <0, only the final result at the last generation is saved.
        
        :returns: None
        
        """
        
        
        args = [mpr, save_every]
        self.runnerlock.acquire()
        self.joblist.append(args)
        self.runnerlock.release()
        
    def prepend(self,  mpr, save_every: int =-1):
        """
        Similar to :meth:`spring._Hypervisor.append`, but the reonstruction on the given :class:`spring.MPR` is put at the **front** of the execution queue (i.e. it is started at latest right after the one currently running).
        
        :param mpr: an instance of :class:`spring.MPR` for which the execution via the :meth:`spring.MPR.run` has to be scheduled.
        :param save_every: The current status of the reconstruction is saved every *save_every* generations. If <0, only the final result at the last generation is saved.
        
        :returns: None
        """
        
        
        args = [mpr, save_every]
        self.runnerlock.acquire()
        self.joblist.insert(0,args)
        self.runnerlock.release()
        
    def remove(self,  item: Union[int,str]):
        """
        Remove :class:`spring.MPR` objects submitted with :meth:`spring.MPR.runasync` from the execution queue. The given ``item`` parameter can be either:
        
            - An integer type: the element at the corresponding position in the execution queue (given by :meth:`spring._Hypervisor.info`) is removed.
            - A string: the element with the corresponding name (give by :meth:`spring._Hypervisor.info`) is removed from the list. Simple regular expressions allowed by `fnmatch <https://docs.python.org/3/library/fnmatch.html>`_ can be used (Unix shell-style wildcards).
        
        .. note::
            The operation may affect more than one queued job. For example ``item='190*'`` will remove all jobs whose pid starts with ``190``. The value ``item='*'`` will remove all queued jobs.
        
        .. warning::
            Using the position in the queue list is inherently unsafe, as the queue position may change in the meanwhile.
            
                    
        :param item: An integer, representing the current position in the execution queue (as given by :meth:`spring._Hypervisor.info`) or a string with the job name (as given by :meth:`spring._Hypervisor.info`). Simple regular expressions with the wildcards ``*``, ``?`` and ``[]`` are allowed to remove multiple items at once.
        
        :returns: None
        """
        
        if type(item) is str:            
            
            self.runnerlock.acquire()
            joblist_new = []
            for ij, job in enumerate(self.joblist):
                pidstr = self.get_pid_string(job[0].pid, job[0].tag)
                if not fnmatch.fnmatch(pidstr, item):
                    joblist_new.append(job)
                else:
                    print("Removing {:s} from queue".format(pidstr))
            self.joblist = joblist_new
            self.runnerlock.release()
        else:
            self.runnerlock.acquire()
            try:
                job = self.joblist[item]
                pidstr = self.get_pid_string(job[0].pid, job[0].tag)
                print("Removing {:s} from queue".format(pidstr))
                self.joblist.pop(item)
            except:
                print("Cannot remove item {:d}".format(item))
                
            self.runnerlock.release()
        return 
            
            
            
    
        
    def kill(self):
        """
        Empty the execution queue and interrupt the currently running one. Equivalent to the execution of :meth:`spring._Hypervisor.remove("*")` (which empties the whole execution queue) and :meth:`spring._Hypervisor.stop()` (which interrupts the currently running process).
                
        :returns: None
        """
    
        self.remove("*")
        self.stop()
    
    

    def reset(self):
        
        self.stop()
        
        #print("Resetting channels...",end='', flush=True)
        self.newevent.clear()
        self.stopevent.clear()
        self.doneevent.clear()
        self.newreceived.clear()
        #self.livequeue.clear()
        while not self.livequeue.empty():
            self.livequeue.get()        
        while not self.queue.empty():
            self.queue.get()
        self.result = None
        self.process = None
        self.fig = None
        #print("Done")
                
    def wait(self):
        """
        Block until the currently running reconstruction is complete.
        """
        if self.process is not None:
            self.process.join()
        return

    def run(self, mpr, save_every):
        
        self.runpid = mpr.pid     
        self.runtag = mpr.tag
        self.process = mp.Process(target=runproc, args=(mpr, save_every, self.queue, self.logqueue, self.newevent,self.stopevent, self.doneevent), daemon=False)
        self.process.start()


    def stop(self):
        """
        Stop the currently running MPR reconstruction. 
        """
        if self.process is not None:
            if self.process.is_alive():
                print("Terminating running MPR process... ", end='', flush=True)
                self.stopevent.set()

                self.process.join()
                print("Done")


    def update(self):
        while True:
            if self.newevent.wait(5):
                self.result = self.queue.get()
                self.newevent.clear()
                
                while not self.livequeue.empty():
                    self.livequeue.get()
                self.livequeue.put(self.result)
                self.newreceived.set()        

    def get(self): 
        """
        Get the reconstruction result at the current status.
        
        :returns: :class:`spring.Result`
        """
        return self.result


    def livelog(self):
        """
        Start/stop the live logging of the reconstruction status
        """
        
        
        if self.livelogging.is_set():
            self.livelogging.clear()
            self.livelogger.join()
        else:
            self.livelogging.set()
            self.livelogger = mp.Process(target=logger, args=(self.livelogging, self.msgqueue), daemon=False)
            self.livelogger.start()

    def liveplot(self, cmap: str = 'bone', cmap_pattern: str ='inferno'):
        """
        Start/stop the live plotting of the reconstruction status
        """
        
        if self.liverunning.is_set():
            print("Stopping live plotting... ", end='')
            self.liverunning.clear()
            self.plotprocess.join()
            print("Done")
        
        else:
            self.liverunning.set()
            self.plotprocess = mp.Process(target=runplot, args=(self.livequeue, self.newreceived, self.liverunning), daemon=False)
            self.plotprocess.start()
    
    




###### Hypervisor unique instance ############

hypervisor = _Hypervisor()
"""
Instance of :class:`spring.mpr_hypervisor.Hypervisor`.
"""






