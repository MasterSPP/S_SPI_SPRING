from spring_core.phase_retrieval import Solver
import numpy as np
from spring.alg_manager import *
from spring.result import Result
from spring.pattern_manager import Pattern
from spring.settings_manager import Settings
from spring.stats_manager import Stats
from spring_core import GPUtils
import copy 


#import threading
import os
import time
#import curses
from IPython.display import display, clear_output
from signal import signal, SIGINT
import queue
from spring.mpr_hypervisor import _Hypervisor, hypervisor


class StopRec(Exception):
    pass

class MPR:
    """
    Initialise the class to handle the reconstruction process.
    
    :param pattern: A :py:class:`spring.Pattern` object initialized with the diffraction pattern.
    :param settings: A :py:class:`spring.Settings` object initialized with the desired settings for the MPR parameters.
    :param tag:  If *tag* is given, the final part of the reconstruction file is formatted as ``...pid[n]_tag_rec.h5``

    
    """
    def __init__(self, pattern: Pattern, settings: Settings, tag: str=''):
        
        self.verbose=False
        self.fullsettings = settings
        self.settings = settings.get()
        self.settings_adv = settings.get_advanced()
        
        
        self.pattern = pattern.get()
        self.pid = pattern.pid
        self.tag = tag
        self.metadata=pattern.metadata
        self.gpulist = []

        #self.setIA()
        
        
        self.reconstructions = None
        self.refine = False
        
        self.thread = None
        self.result = None
        
        self.queue = None
        self.newevent = None
        self.stopevent= None
        self.doneevent = None
        
        self.logqueue = None
        
        
        
    def print(self, msg, end="\n", flush=True):
        if self.logqueue is None:
            print(msg, end=end, flush=flush)
        else:
            sendmsg = msg+end
            self.logqueue.put(sendmsg)
            
        
        
    def set_channels(self, queue, logqueue, newevent, stopevent, doneevent):
        self.queue = queue
        self.logqueue = logqueue
        self.newevent = newevent
        self.stopevent = stopevent
        self.doneevent = doneevent
        
    
    def InitPop(self):

        supportsize=self.settings['init']['supportsize']
        self.reconstructions = self.solver.InitializePop(popsize=self.settings['global']['popsize'], 
                                                    suppsize = supportsize, 
                                                    itemsize = [supportsize*self.settings['init']['itemsize_min'],supportsize*self.settings['init']['itemsize_max']], 
                                                    nitems = [self.settings['init']['itemnum_min'], self.settings['init']['itemnum_max']], 
                                                    exponent=[self.settings['init']['gamma'], self.settings['init']['gamma']], 
                                                    scaling=[self.settings_adv['scaling_min'], self.settings_adv['scaling_max']], 
                                                    phaserange=self.settings['init']['phaserange'] , 
                                                    suppcut=0)
        return self
    
        
    #def InitPopFromRec(self, popsize, density, support, gamma=0.5, itemsize=[0.2, 0.9], itemnum=[2,8], initphi=0):

        #coords = np.argwhere(support_start)
        #x_min, y_min = coords.min(axis=0)
        #x_max, y_max = coords.max(axis=0)
        #xdim = x_max-x_min
        #ydim = y_max-y_min
        #effdim = (xdim+ydim)//2
        #print("Detected support sizes - x: {:d}, y: {:d}, eff: {:d}". format(xdim, ydim,effdim))
        #starting_guess = spring.Reconstruction(density, support)
        #self.reconstructions = solver.InitializePopRef(popsize=nrec, reference=starting_guess, itemsize = [support_size*itemsize[0],support_size*itemsize[1]], nitems = itemnum , exponent=[gamma, gamma], scaling=[normfactor*0.5, normfactor*1.5], phaserange=initphi, suppcut=0)
        
        
        
    def InitIA(self):

        main_algs = []
        alg = self.settings['IA']['alg']
        it = self.settings['IA']['it']
        beta = self.settings['IA']['beta']
        if alg=='HIO':
            main_algs.append(HIO(it, 0, beta, decay_exp=self.settings_adv['decay_exp']))
        elif alg=='RAAR':
            main_algs.append(RAAR(it, 0, beta, decay_exp=self.settings_adv['decay_exp']))    
    
        main_algs.append(ER(self.settings['IA']['it_ER'], self.settings['IA']['it_ER']+it, decay_exp=self.settings_adv['decay_exp']))
        
        eval_algs=[ER(self.settings['IA']['it_eval'])]
        

        stab_algs = []
        if self.settings['IA']['it_stab']>0: stab_algs.append(ER(self.settings['IA']['it_stab']))  

        self.algman = AlgManager(main =  main_algs,
                            stabilization = stab_algs,
                            evaluation = eval_algs,
                            sw_sigma = self.settings['IA']['sigma'],
                            sw_threshold = self.settings['IA']['threshold'], 
                            repetitions = self.settings['IA']['repetitions'],
                            scalingstart = int(np.round(self.settings_adv['decay_start']*self.settings['global']['generations'])),
                            scalingend = int(np.round(self.settings_adv['decay_end']*self.settings['global']['generations']))
                            
                )
        return self
        
        
    def SelectGPUs(self):
        self.gpulist = []
        
        if isinstance(self.settings['global']['gpus'], int):
            ids = GPUtils.get_ids()
            info = GPUtils.get_info()
            usage = GPUtils.get_usage()
            if self.settings['global']['gpus']<=0 or self.settings['global']['gpus']>len(ids):
                self.gpulist = ids
            else:
                sind = np.argsort(usage)
                for id in sind[:self.settings['global']['gpus']]:
                    self.gpulist.append(ids[id])
            
            self.print("Selected GPUs:")
            if usage[0]<0:
                for gpu in self.gpulist:
                    self.print("  - {:s} (Id: {:d})".format(info[ids[gpu]], ids[gpu]))
            else:
                for gpu in self.gpulist:
                    self.print("  - {:s} (Id: {:d}, load: {:d}%)".format(info[ids[gpu]],ids[gpu],usage[ids[gpu]]))
            self.gpulist = np.array(self.gpulist)
        else:
            self.gpulist = np.array(self.settings['global']['gpus'])        
        
    def InitSolver(self):

        
        self.solver = Solver(np.copy(self.pattern), 
                                    reality=self.settings['IA']['reality'], 
                                    realpen = 0, 
                                    use_bounds = self.settings['global']['bounds'], 
                                    gpus=self.gpulist,
                                    nthreads = self.settings['global']['threads']*len(self.gpulist))
        
        self.solver.set_shift_param(shiftmin=self.settings_adv['shift_min'], shiftmax=self.settings_adv['shift_max'])
        
        


    def runasync(self, save_every: int =-1):
        """
        Start the reconstruction process asynchronously. 
        The reconstruction is saved in a file, that is updated along the reconstruction. The name of the file that contains the reconstruction is formed by using the *pid* parameter of the :py:class:`spring.Pattern`, with the form ``pid[0]_pid[1]_ ... pid[n]_rec.h5``.
        
        :param save_every: The current status of the reconstruction is saved every *save_every* generations. If <0, only the final result at the last generation is saved.
        
        :returns: None
        
        """
        
        
        global hypervisor
        hypervisor.append(self, save_every)
        return 
        
    def checkstop(self):
        if self.stopevent is not None:
            return self.stopevent.is_set()
        else:
            return False

    def run(self, save_every: int =-1):
        """
        Start the reconstruction process. 
        The reconstruction is saved in a file, that is updated along the reconstruction. The name of the file that contains the reconstruction is formed by using the *pid* parameter of the :py:class:`spring.Pattern`, with the form ``pid[0]_pid[1]_ ... pid[n]_rec.h5``.
        
        :param save_every: The current status of the reconstruction is saved every *save_every* generations. If <0, only the final result at the last generation is saved.
        
        :returns: :py:class:`spring.Result`
        
        """
        
        #self.fullsettings.print()
        try:
            
            self.stats = Stats()
            self.SelectGPUs()

            ##############################d
            self.print("Initializing solver... ", end='', flush=True)
            self.InitSolver()
            self.print("Done.")
            
            self.print("Initializing algorithms... ", end='', flush=True)
            self.InitIA()
            self.print("Done.")
            
            self.print("Initializing population... ", end='', flush=True)
            self.InitPop()              
            self.print("Done.")
            

            
            ##############################d
            self.print("... Running main loop ...")

            self.solver.Evaluate(self.reconstructions)
            self.sortedargs = np.argsort([rec.get_error() for rec in self.reconstructions])
            bestrec = self.reconstructions[self.sortedargs[0]]
            self.solver.set_sequence(self.algman.get_sequence(0, area = [np.sum(bestrec.get_support()), np.sum(bestrec.get_support())], update_support = not self.refine))
            self.reconstructions = self.solver.Sequence(self.reconstructions)  
            #self.sortedargs = np.argsort([rec.get_error() for rec in self.reconstructions])
            #self.reconstructions = [self.reconstructions[i] for i in self.sortedargs]

            self.generation = 0
            self.replcount=0
            
            
            self.sort()
            self.get_best()
            self.get_average()
            self.save_result()
            
            while self.generation<self.settings['global']['generations']:

                ttot = time.time()
            
                self.generation+=1
                
                ##########################
                tcrossprep = time.time()
                
                if not self.refine:
                    swparams = self.algman.get_support_params(self.generation)
                    if self.verbose: print("Algorithms sequence: ", *swparams, sep="\n\t")

                    self.reconstructions= self.solver.SW(self.reconstructions, *swparams)
                
                self.sort()
                self.get_best()
                self.get_average()

                tcrossprep = time.time() -tcrossprep
                
                    
                #################
                
                tsuppix = time.time()
                if not self.refine:
                    suppixbest = np.sum(self.bestrec.get_support())
                    suppixeval = np.sum(self.avgrec.get_support())
                    suppix=[suppixbest,suppixeval]
                            
                    
                else:
                    suppix = [0,0]
                tsuppix = time.time()-tsuppix
                    

                #################

                tcross = time.time()
                self.new_reconstructions = self.solver.Crossover(self.reconstructions,
                                                            self.bestrec, 
                                                            self.avgrec, 
                                                            self.settings['GA']['crossprob'], 
                                                            self.settings['GA']['crossweight'], 
                                                            self.settings_adv['crossexp'],  
                                                            self.settings_adv['crossft'],  
                                                            self.settings_adv['crosssym'], 
                                                            self.settings['GA']['crossaverage'],  
                                                            update_support = not self.refine)
                tcross = time.time()-tcross    
                
                ###################
                
                tprint = time.time()

                
                tprint = time.time()-tprint

 
                ###### IPR ##########
                tipr=time.time()
                algsequence = self.algman.get_sequence(self.generation, area = suppix, update_support = not self.refine)

                if self.verbose: print("Algorithms sequence: ", *algsequence, sep="\n\t")

                
                self.solver.set_sequence(algsequence)
                self.reconstructions = self.solver.Sequence(self.reconstructions)   
                self.new_reconstructions = self.solver.Sequence(self.new_reconstructions)   
                
                tipr=time.time()-tipr

                #############################d
                trepl=time.time()

                repl=np.zeros(len(self.reconstructions))
                self.replcount=0   

                for i in range(len(self.new_reconstructions)):
                    iold=i%len(self.reconstructions)
                    if self.new_reconstructions[i].get_error() < self.reconstructions[iold].get_error():
                        repl[iold]=1
                        self.reconstructions[iold] = self.new_reconstructions[i]

                self.replcount=np.sum(repl)
                
                trepl=time.time()-trepl

                
                ttot = time.time()-ttot
                
                self.save_result()
                
                
                if (save_every>0 and self.generation%save_every==0) or self.generation==self.settings['global']['generations']: 
                    self.result.save(self.settings['global']['workdir'], self.tag)   

                #if self.verbose: print("Timing (s) --> Print: {:.2f}s , CrossPrep: {:.2f}s , Cross: {:.2f}s , SuppPix: {:.2f}s , IPR: {:.2f}s , Repl: {:.2f}s , total: {:.2f}s ({:.2f}s)".format(tprint, tcrossprep, tcross, tsuppix, tipr, trepl, ttot, (tprint+tcrossprep+tcross+tsuppix+tipr+trepl)))
                
                if self.checkstop():
                    raise StopRec             
            
            self.print("Reconstruction completed")
            self.setdone()
            return self.result
        
        except StopRec:
            self.print("Reconstruction interrupted.")
            self.print("Saving reconstruction... ", end='')
            self.result.save(self.settings['global']['workdir'], self.tag)   
            self.print('Done')                
            self.print("Empyting the communication queue... ", end='')
            try:
                while 1:
                    self.queue.get(timeout=0.5)
            except queue.Empty:
                pass
            
            self.print('Done')                
            self.setdone()
            return self.result
        except KeyboardInterrupt:
            pass
    
    def save_result(self):
        
            self.stats.add(self.reconstructions, self.bestrec, self.avgrec, self.replcount, self.generation, tstamp=time.time())
            #self.stats.print(self.settings['global']['generations'])
            
            res = Result()
            res.set(self.pattern, 
                    self.bestrec.get_density(), self.bestrec.get_support(), 
                    self.avgrec.get_density(), self.avgrec.get_support(),
                    self.pid,
                    self.stats,
                    self.fullsettings,
                    self.metadata,
                    step = self.generation,
                    total = self.settings['global']['generations'])
                                        
            if self.queue is not None:
                while not self.queue.empty():
                    self.queue.get()
                self.queue.put(res)
                self.newevent.set()
            else: 
                res.print_status()
            self.result=res               
    
        
        
    
    
    def setdone(self):
        if self.doneevent is not None:
            self.doneevent.set()
        
    def sort(self):
        self.sortedargs = np.argsort([rec.get_error() for rec in self.reconstructions])

        
    def get_average(self):

        self.solver.set_shift_param(shiftmin=0, shiftmax=2)
        self.reconstructions = self.solver.Reshift(self.reconstructions, self.bestrec)

        #t = time.time()
        self.avgrec = self.solver.GetAverage(self.reconstructions, fraction=self.settings_adv['avg_frac'])
        #print("Avg time ", time.time()-t)

        self.solver.set_shift_param(shiftmin=self.settings_adv['shift_min'], shiftmax=self.settings_adv['shift_max'])
        self.solver.Evaluate(self.avgrec)



    def get_best(self):
        self.bestrec = self.reconstructions[self.sortedargs[0]]
        

#############################################d
#############################################d
#############################################d



        
            
