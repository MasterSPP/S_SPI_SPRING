import numpy as np

class Algorithm:
    def __init__(self, name: str, it_start: int, it_end: int, parameters: list, decay_exp: float):
        self.it_start = it_start
        self.it_end = it_end
        self.decay_exp = decay_exp
        self.parameters = parameters
        self.name = name
        
    def get(self, gen_it, gen_start, gen_end):
        
        scaling = np.clip(np.real((1.-((gen_it-gen_start)/(gen_end-gen_start)))**self.decay_exp), 0,1)
        
        alg_it = int(np.round(scaling*self.it_start + (1-scaling)*self.it_end))
        return self.name, [alg_it, *self.parameters]        
        

class HIO(Algorithm):
    def __init__(self, it_start: int, it_end: int = -1,  beta = 0.95, decay_exp: float = 1,):
        if it_end<0: it_end=it_start
        super().__init__("HIO", it_start, it_end, [beta], decay_exp)

class RAAR(Algorithm):
    def __init__(self, it_start: int, it_end: int = -1, beta = 0.95, decay_exp: float = 1, ):
        if it_end<0: it_end=it_start
        super().__init__("RAAR", it_start, it_end, [beta], decay_exp)

class ER(Algorithm):
    def __init__(self, it_start: int, it_end: int = -1, decay_exp: float = 1):
        if it_end<0: it_end=it_start
        super().__init__("ER", it_start, it_end, [], decay_exp)
        
class SW:
    def __init__(self, sigma: float = 1.5, threshold: float = 0.07, sigma_end: float = -1, threshold_end: float = -1, decay_exp: float = 1):
        self.sigma = sigma
        self.threshold = threshold
        self.sigma_end = (sigma_end if sigma_end>0 else 0.66)
        self.threshold_end = (threshold_end if threshold_end>0 else threshold*0.66)        
        self.decay_exp = decay_exp
        
    def get(self, gen_it, gen_start, gen_end, area = None, area2=None):
        scaling = np.clip(np.real((1.-((gen_it-gen_start)/(gen_end-gen_start)))**self.decay_exp), 0,1)
        locsigma = scaling*self.sigma + (1-scaling)*self.sigma_end
        locthreshold = scaling*self.threshold + (1-scaling)*self.threshold_end
        
        if area is None:
            return "SW", [locsigma, locthreshold]
        else:
            if area2 is None:
                return "SWA", [locsigma, area]
            else:
                locarea = scaling*area2 + (1-scaling)*area
                return "SWA", [locsigma, locarea]
    
                
            
            
        
class AlgManager:
    def __init__(self, main: list[Algorithm], stabilization: list[Algorithm] = [], evaluation: list[Algorithm] = [],  sw_sigma: float = 1.7, sw_threshold: float = 0.07, repetitions: int = 3, scalingstart: int = 0, scalingend: int = 1000):
        self.main_sequence = main
        self.stabilization_sequence = stabilization
        self.evaluation_sequence = evaluation
        self.scalingstart = scalingstart
        self.scalingend = scalingend

        self.support_update = SW(sigma = sw_sigma, threshold=sw_threshold)
        self.repetitions= (repetitions if repetitions>2 else 2)
        
         
    def get_main_sequence(self, gen_it):
        alg_sequence = []
        for alg in self.main_sequence:
            alg_sequence.append(alg.get(gen_it, self.scalingstart, self.scalingend))
        
        return alg_sequence
        
    def get_support_sequence(self, gen_it, area=None, area2=None):
        return [self.support_update.get(gen_it, self.scalingstart, self.scalingend, area, area2)]
    
    def get_support_params(self, gen_it, area=None):
        name, params = self.support_update.get(gen_it, self.scalingstart, self.scalingend, area)
        return params


    def get_stabilization_sequence(self, gen_it):
        alg_sequence = []
        
        for alg in self.stabilization_sequence:
            alg_sequence.append(alg.get(gen_it, self.scalingstart, self.scalingend))
        
        return alg_sequence
    
    
    def get_evaluation_sequence(self, gen_it):
        alg_sequence = []
        
        for alg in self.evaluation_sequence:
            alg_sequence.append(alg.get(gen_it, self.scalingstart, self.scalingend))
        
        return alg_sequence

    
    def get_sequence(self, gen_it, area = None, update_support=True):
        
        stab_seq = self.get_stabilization_sequence(gen_it)
        main_seq = self.get_main_sequence(gen_it)
        eval_seq = self.get_evaluation_sequence(gen_it)
        full_seq = stab_seq + main_seq

        if update_support:
            sw_seq = self.get_support_sequence(gen_it)
            swa_seq = self.get_support_sequence(gen_it, area[0])
            swa_seq_eval = self.get_support_sequence(gen_it, area[0], area[1])
        else:
            sw_seq = []
            swa_seq = []
            swa_seq_eval = []
        
        alg_sequence = []
        
        #for irep in range(self.repetitions-1):
            #alg_sequence += full_seq
            #if irep==self.repetitions-2:
                #alg_sequence += sw_seq
        
        #alg_sequence += swa_seq + full_seq + swa_seq_eval + eval_seq

        for irep in range(self.repetitions-2):
            alg_sequence += full_seq + sw_seq
            #if irep==self.repetitions-2:
                #alg_sequence += sw_seq
        
        alg_sequence += full_seq + swa_seq
        
        alg_sequence += full_seq + swa_seq_eval + eval_seq
       
        return alg_sequence
    
    

    def get_sequence_ipr(self, gen_it, area = None, update_support=True):
        
        stab_seq = self.get_stabilization_sequence(gen_it)
        main_seq = self.get_main_sequence(gen_it)
        eval_seq = self.get_evaluation_sequence(gen_it)
        full_seq = stab_seq + main_seq

        if update_support:
            sw_seq = self.get_support_sequence(gen_it)
        else:
            sw_seq = []
            
        alg_sequence = []
        

        for irep in range(self.repetitions):
            alg_sequence += full_seq + sw_seq
        
        
        alg_sequence += eval_seq
       
        return alg_sequence
        
        
        
        
        
        
        
        
        
        
        
        
        
        
