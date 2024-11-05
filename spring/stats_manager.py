import numpy as np


class Stats:
    def __init__(self):
        
        self.scr = None
        
        self.entries = [
            'error',
            'complexity',
            'symmetry',
            'module_sum',
            'complex_sum'
            ]
        
        self.stats={
            'best' : {ee : [] for ee in self.entries},
            'average' : {ee : [] for ee in self.entries},
            'mean' : {ee : [] for ee in self.entries},
            'all' : {ee : [] for ee in self.entries},
            'global':{
                'OS':[],
                'generation':[],
                'replacement':[],
                'tstamp': [],
                }
            }
        
        
        self.printinfo = {'error': 'Err.',
                          'complexity': 'Compl.',
                          'symmetry': 'Sym.',
                          'module_sum': 'Sum.'
                          }
        
        self.printlines = {'best':      'Best\t',
                           'mean':      'Mean\t',
                           'average':   'Avg \t'
                           }
        
    def add(self, reconstructions, bestrec, avgrec, replcount, generation, tstamp, save_OS=False):
        
        errors = [rec.get_error() for rec in reconstructions]
        compelxities = [rec.get_reality() for rec in reconstructions]
        symmetries = [rec.get_symmetry() for rec in reconstructions]
        module_sums = [rec.get_module_sum() for rec in reconstructions] 
        complex_sums = [rec.get_complex_sum() for rec in reconstructions] 
        
        
        bestind = np.argmin(errors)
        
        self.stats['best']['error'].append(errors[bestind])
        self.stats['best']['complexity'].append(compelxities[bestind])
        self.stats['best']['symmetry'].append(symmetries[bestind])
        self.stats['best']['module_sum'].append(module_sums[bestind])
        self.stats['best']['complex_sum'].append(complex_sums[bestind])
        
        self.stats['average']['error'].append(avgrec.get_error())
        self.stats['average']['complexity'].append(avgrec.get_reality())
        self.stats['average']['symmetry'].append(avgrec.get_symmetry())
        self.stats['average']['module_sum'].append(avgrec.get_module_sum())
        self.stats['average']['complex_sum'].append(avgrec.get_complex_sum()) 

        self.stats['mean']['error'].append(np.mean(errors))
        self.stats['mean']['complexity'].append(np.mean(compelxities))
        self.stats['mean']['symmetry'].append(np.mean(symmetries))
        self.stats['mean']['module_sum'].append(np.mean(module_sums))
        self.stats['mean']['complex_sum'].append(np.mean(complex_sums))
        
        bestsupport = bestrec.get_support()
        matrixsize = bestsupport.shape[0]*bestsupport.shape[1]
        self.stats['global']['generation'].append(generation)
        self.stats['global']['replacement'].append(replcount/len(reconstructions))
        self.stats['global']['tstamp'].append(tstamp)
    
        self.stats['all']['error'].append(errors)
        self.stats['all']['complexity'].append(compelxities)
        self.stats['all']['symmetry'].append(symmetries)
        self.stats['all']['module_sum'].append(module_sums)
        self.stats['all']['complex_sum'].append(complex_sums)
        
        
        if save_OS:
            OS_vals = [matrixsize/np.sum(rec.get_support())  for rec in reconstructions]
            self.stats['global']['OS'].append(OS_vals)
        else:    
            self.stats['global']['OS'].append(matrixsize/np.sum(bestsupport))
        

    def get(self):
        return self.stats


    def set(self, statsdict):
        self.stats = statsdict
        return


    def print(self):
        
        outstring = ""
        
        head = "\tOS: {:.2f}  Repl: {:.3f}".format(self.stats['global']['OS'][-1], self.stats['global']['replacement'][-1])
        
        outstring+= head+"\n"
        
        
        for pl in self.printlines.keys():
            linestring = '\t'
            linestring+= self.printlines[pl]
            for pi in self.printinfo.keys():
                linestring+=' '+self.printinfo[pi] + " {:.3e} ;".format(self.stats[pl][pi][-1])
            outstring+= linestring +"\n"
        
        #clear_output(wait=True)
        print(outstring)
