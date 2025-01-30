import yaml
import json
from prettytable import PrettyTable, MARKDOWN
from textwrap import fill

class Settings:
    """
    Class to handle the settings of MPR. The settings are initialized with their default value. If *fname* is specified, then settings are loaded from the indicated file previously saved via the :py:meth:`~Settings.save` method.
    """
    def __init__(self, fname: str =None):
        glob = {
                'generations': {
                    'value': 100,
                    'range': [1,None],
                    'type': 'int',
                    'help': 'Total number of MPR generations.'},
                'popsize':{
                    'value': 128,
                    'range': [8,None],
                    'type': 'int',
                    'help': 'Population size. Total number of reconstructions that form the population.'},
                'gpus': {
                    'value': [0],      
                    'range': [0,None],         
                    'type': 'list/int',
                    'help': 'Number of GPUs to use or, alternatively, list of specific GPU ids. If a number N is given, the current usage of the available GPUs is inspected and the N ones with lower computing load are selected. If N<=0, all available GPUs are selected.'},
                'threads':{
                    'value': 8,      
                    'range': [1,None],         
                    'type': 'int',
                    'help': 'Number of CPU threads per GPU. The total number of threads used for calculations is this value times the number of GPU in use'},
                'bounds':{
                    'value': 0,      
                    'range': [0,1],         
                    'type': 'int',
                    'help': 'If set to 1, use the upper bound for the missing intensities in the diffraction data'},
                'workdir':{
                    'value': './',      
                    'range': [None,None],         
                    'type': 'string',
                    'help': 'Path where the results are saved'},
            }
        
        init={
                'gamma' : 
                    {'value': 0.5,      
                     'range': [0,2],         
                     'type': 'float',
                    'help': 'Gamma value for the initizalization of the densities'},
                'supportsize':{
                    'value': 60,
                    'range': [1,None],
                    'type': 'int',
                    'help': 'Initial linear dimension of the support function.'},
                'itemsize_min':{
                    'value': 0.2,
                    'range': [0,1],
                    'type': 'float',
                    'help': 'Minimum relative diameter of the spherical density profiles used to define the starting densities. The actual minimum size in pixels is calcualted as itemsize_min*supportsize'},
                'itemsize_max':{
                    'value': 0.9,
                    'range': [0,1],
                    'type': 'float',
                    'help': 'Maximum relative diameter of the spherical density profiles used to define the starting densities. The actual maximum size in pixels is calcualted as itemsize_max*supportsize'},
                'itemnum_min': {
                    'value': 2,
                    'range': [1,None],
                    'type': 'int',
                    'help': 'Minimum number of spherical density profiles used to create the starting densities'},
                'itemnum_max': {
                    'value': 8,
                    'range': [1,None],
                    'type': 'int',
                    'help': 'Maximum number of spherical density profiles used to create the starting densities'},
                'phaserange':{
                    'value': 0,
                    'range': [0,1],
                    'type': 'float',
                    'help': 'Range of phase values in real space for the initialized densities. If 0, densities are completely real-valued. In general, the phase values are initialized in the range [-phaserange*pi, phaserange*pi]'},               
            }
                
        IA = {
                'alg': {
                    'value': 'HIO',
                    'range': ['HIO','RAAR'],
                    'type': 'string',
                    'help': 'Iterative algorithm used for local optimization.'},
                'it': {
                    'value': 40,
                    'range': [0,None],
                    'type': 'int',
                    'help': 'Starting number of iterations of the iterative algorithm.'},
                'beta':{
                    'value': 0.95,
                    'range': [0,1],
                    'type': 'float',
                    'help': 'Beta parameter of the iterative algorithm'},
                'it_ER':{ 
                    'value': 40,
                    'range': [0,None],
                    'type': 'int',
                    'help': 'Number of Error Reduction iterations executed after RAAR or HIO.'},
                'it_eval':{ 
                    'value': 40,
                    'range': [0,None],
                    'type': 'int',
                    'help': 'Number of Error Reduction iterations used for the evaluation of the reconstruction.'},
                'it_stab':{
                    'value': 0,
                    'range': [0,None],
                    'type': 'int',
                    'help': 'Number of Error Reduction iterations used for the stabilization of the reconstruction.'},
                'sigma':{
                    'value': 2.5,
                    'range': [0,None],
                    'type': 'float',
                    'help': 'Starting value of the smoothing for the Shrink-wrap algorithm. This value is then reduced to 0.5 at the end of the reconstruction.'},
                'threshold':{
                    'value': 0.03,
                    'range': [0,None],
                    'type': 'float',
                    'help': 'Starting threshold for the Shrink-wrap algorithm. This value is then reduced to 2/3 of its starting value at the end of the reconstruction.'},
                'reality':{
                    'value': 0.5,
                    'range': [0,1],
                    'type': 'float',
                    'help': 'Reality constraint of the densities. Phases are constrained between -pi*(1-reality) and pi*(1-reality) during the execution of the IAs iterations'},
                'repetitions':{
                    'value': 3,
                    'range': [3,None],
                    'type': 'int',
                    'help': 'Number of repetitions of the IA algorithm sequence per iteration for each density.'},
            }
        
        GA = {
                'crossprob': {
                    'value': 0.6,
                    'range': [0,1],
                    'type': 'float',
                    'help': 'Probability of crossover'},
                'crossweight': {
                    'value': 0.4,
                    'range': [0,1],
                    'type': 'float',
                    'help': 'Weight of the differential crossover'},
                'crossaverage': {
                    'value': 0,
                    'range': [0,2],
                    'type': 'float',
                    'help': 'Weight assigned to the average intensities in the masked regions of the diffraction pattern during the crossover. If 0, average intensities are not used. If 1, the intensities of the new densities are completely replaced with the ones of the average.'},
                }
                
        self.settings = {
            'global': glob,
            'init': init,
            'IA':IA,
            'GA': GA            
            }
        
        self.advanced = {
            'decay_exp': 1.5,
            'decay_start': 0.1,
            'decay_end': 0.95,
            'shift_min': 0,   
            'shift_max': 2,
            'avg_frac': 0.8,
            'avg_supp': 1.,
            'crosssym' : False,
            'crossexp' : 1,
            'crossft' : 1,
            'scaling_min': 0.5,
            'scaling_max': 1.5,
            'seed' : 42
            }
        
        if fname is not None:
            self.load(fname)
         
    
    def check(self, section, name, value):
        try:
            setval = self.settings[section][name]
            return True
        except Exception as e:
            print(e)
            return False
        
        
       
    def get(self, section: str = None, name: str = None):
        """
        Get the value of the parameters, returned as a dictionary.
        If both ``section`` and ``name`` are specified, the parameter is directly returned as a value.
        
        :param section: Section to which the parameter belongs.
        :param name: Name of the parameter
        """
        setvals = {key:{ key2 : self.settings[key][key2]['value'] for key2 in   self.settings[key].keys()} for key in self.settings.keys()}
        if section is None:
            return setvals
        else:
            if name is None:
                try:
                    return setvals[section]
                except Exception as e:
                    print(e)
                    return None
            else:
                try:
                    return setvals[section][name]
                except Exception as e:
                    print(e)
                    return None
                
    def get_advanced(self):
        return self.advanced
    
    def set_advanced(self, name: str , value):
        """
        Set the value of a parameter with name ``name`` belonging to the section ``section``.
        
        :param section: Section to which the parameter belongs.
        :param name: Name of the parameter
        :param value: Value to assign to the parameter      
        """
        self.advanced[name]=value
            
        return self
    
    def set(self, section : str , name: str , value):
        """
        Set the value of a parameter with name ``name`` belonging to the section ``section``.
        
        :param section: Section to which the parameter belongs.
        :param name: Name of the parameter
        :param value: Value to assign to the parameter      
        """
        if self.check(section, name, value):
            self.settings[section][name]['value']=value
        else:
            print("Wrong format")
            
        return self
    
    def info(self, section: str = None, name: str = None):
        """
        Print information about the current settings. 
        
        :param section: If given, it only prints information on parameters belonging to the given section.
        :param name: If given along with ``section``, it only prints information on that parameter. 
        
        """

        table = PrettyTable(["Section", "Name", "Value", "Type","Range", "Description"]) 
        rows = []
        outset=self.settings

        for secname in outset.keys():
            if section is None or secname==section:
                for  iname, parname in enumerate(outset[secname].keys()):
                    if name is None or parname==name:
                        desc = outset[secname][parname]
                        rangestr=''
                        #print(desc['type'])
                        if desc['type']=='string':
                            if desc['range'][0] is not None:
                                for v in desc['range']:
                                    rangestr+=v+' '
                        
                        if desc['type']=='int':
                            if desc['range'][0] is not None:
                                rangestr+='>{:d} '.format(desc['range'][0])
                            if desc['range'][1] is not None:
                                rangestr+='<{:d}'.format(desc['range'][1])
                                
                        if desc['type']=='float':
                            if desc['range'][0] is not None:
                                rangestr+='>{:.1f} '.format(desc['range'][0])
                            if desc['range'][1] is not None:
                                rangestr+='<{:.1f}'.format(desc['range'][1])
                        
                        secstring=''
                        if iname==0:
                            secstring=secname
                        
                        row = [secstring,parname,desc['value'],desc['type'], rangestr , fill(desc['help'], width=50)]
                        rows.append(row)
        
        for irow,row in enumerate(rows):
            div=True
            if irow<len(rows)-1:
                if rows[irow+1][0]=='':
                    div=False
                
            table.add_row(row, divider=div)
        table.set_style(MARKDOWN)
        print(table)
                
        
    def print(self, section: str = None, name: str = None):
        """
        Print the current values of the parameters.
        
        :param section: If given, it only prints the parameter values belonging to the given section.
        :param name: If given along with ``section``, it only prints the value of that parameter. 
        
        """
        outset=self.get()
        
        subdict={}
        if section is None:
            subdict = outset
        else:
            if name is None:
                subdict = outset[section]
            else:
                subdict = outset[section][name]
                
        print(yaml.dump(subdict, sort_keys=False,default_flow_style=False))
        
    
    def save(self, fname:str):
        """
        Save the current settings into the file ``fname``. The file is in YAML format.
        
        :param fname: name of the file where the settings are saved
        """
        outset=self.get()
                
        with open(fname, "w") as f:  
            yaml.safe_dump(outset, f, sort_keys=False,default_flow_style=False)
            
    
    def setall(self, inset):
        for section in self.settings.keys():
            for name in self.settings[section].keys():
                if name in inset[section].keys():
                    value = inset[section][name]
                    if self.check(section, name, value):
                        #print(value)
                        self.settings[section][name]['value']=value
                    else:
                        print("WARNING: value {} for setting {}/{} not in range ({},{}). Keeping the current value {}".format(value, section, name,self.settings[section][name]['range'][0], self.settings[section][name]['range'][1],self.settings[section][name]['value']))
                else:
                        print("WARNING: setting {}/{} not found in file. Keeping the current value {}".format(section, name,self.settings[section][name]['value']))        
    
    def load(self, fname: str):
        """
        Load settings from the YAML file ``fname``.
        
        :param fname: name of the file from which the settings are loaded
        
        """
        inset = {}
        
        with open(fname) as stream:
            try:
                inset = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                
        self.setall(inset)

        
    def getjson(self):
        outset=self.get()
        return json.dumps(outset)
    
    def setjson(self, jsonstr):
        inset = json.loads(jsonstr)
        self.setall(inset)
        return 

    
