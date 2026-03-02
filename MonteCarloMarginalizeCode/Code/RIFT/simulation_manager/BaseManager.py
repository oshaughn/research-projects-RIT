

import numpy as np



class SimulationArchive(Object):
    """
    SimulationArchive: tool to allow (a) storing list of 'simulations', (b) querying if present, (c) generating if needed,from a generation function.
    """
    def __init__(self,name=None, **kwargs):
        self.name=name
        self.simulations = {} # dictionary of simulations.  More complex return 

    def simulation_exists_q(self, sim_params):
        if sim_params in self.simulations:  # generaly such a simple test is impossible! 
            return True
        else:
            return False

    def generate_simulation(self, sim_params, generator=None):
        self.simulations[sim_params] = generator(sim_params)

    def retrieve_simulation)(self, sim_params):
        return self.simulation[sim_params]


class SimulationArchiveOnLocalDisk(SimulationArchive):
    def __init__(self, name):
        return None
        
