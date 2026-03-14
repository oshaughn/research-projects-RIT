

import numpy as np
import os

def easy_write(fname,content_lines):
    with open(fname, 'w') as f:
        f.write(str(content_lines) +"\n")

class SimulationArchive:
    """
    SimulationArchive: tool to allow (a) storing list of 'simulations', (b) querying if present, (c) generating if needed,from a generation function.
         -simulation_exists_q : return simulation
         -retrieve_simulation : 
    """
    def __init__(self,name=None, **kwargs):
        self.name=name
        self.simulations = {} # dictionary of simulations.  More complex return
        self.generator = None
        # Generic assignment, if present
        for x in kwargs:
            if hasattr(self, x):
                setattr(self, x, kwargs[x])

    def simulation_exists_q(self, sim_params):
        if sim_params in self.simulations:  # generaly such a simple test is impossible! 
            return True
        else:
            return False

    def generate_simulation(self, sim_params, generator=None, generator_kwargs={}):
        if generator is None and self.generator is None:
            raise Exception(" No generator for  simulation ")
        if generator:
            self.simulations[sim_params] = generator(sim_params, **generator_kwargs)
        else:
            self.simulations[sim_params] = self.generator(sim_params,**generator_kwargs)

    def retrieve_simulation(self, sim_params):
        return self.simulations[sim_params]


class SimulationArchiveOnLocalDisk(SimulationArchive):
    """
        User can specify a 'base location'.  Simulations are associated with files on disk.
         - The generator function is passed the base location as a keyword argument.
         - simulation output is saved to disk.  User is responsible for managing the save_command. File names are identical to the key value.
        No setup services are provided about the base location, aside from that.
    """
    def __init__(self, name=None,storage_method=None,**kwargs):
        self.base_location = None
        self.save_command = np.savetxt
        self.save_params_command = easy_write
        self.load_simulation = np.loadtxt
        self.load_metadata = np.loadtxt
        self.valid_simulation_file = lambda x: os.path.exists(x) and not(x.startswith('metadata_'))
        self.valid_metadata_file = lambda x: os.path.exists(x) and (x.startswith('metadata_'))
        self
        # default method
        if storage_method == 'integers':
            self.internal_index = {}   # integers mapping to actual sim_params
            self.key_to_params = lambda x: self.internal_index[x]
        else:
            self.key_to_params = lambda x: eval(x) # evaluate the string
            
        super().__init__(name,**kwargs)

        if os.path.exists(self.base_location):
            # Initialize base location
            print(" Path exists. Attempting to initialize from saved data ")
            # WRITE PROCEDURE HERE : metadata to initialize the archive, etc
            # DEFAULT IS TO ASSUME ALL FILES IN DIRECTORY ARE VALID, keys are names
            for fname in os.listdir(self.base_location):
#                print(fname)
                full_fname = os.path.join(self.base_location, fname)
                if os.path.isfile(full_fname) and self.valid_simulation_file(full_fname):
                    meta_file = os.path.join(self.base_location, 'metadata_'+fname)  # HARDCODED DEFAULT, FIX THIS
                    full_meta_name = os.path.join(self.base_location, meta_file)
                    if self.valid_metadata_file(full_meta_name):
                        print( " Valid sim ", fname, " with metadata ", meta_file)
                        sim_params = self.load_metadata(full_meta_name)
                        self.simulations[fname] = [sim_params, full_fname,full_meta_name]  # provides access to both simulation and 
        else:
            os.mkdir(self.base_location)
        return None


        
    def generate_simulation(self, sim_params, sim_name=None,meta_name=None,generator=None, generator_kwargs={}):
        if self.simulation_exists_q(sim_params):
            return True # no need
        if generator is None and self.generator is None:
            raise Exception(" No generator for  simulation ")
        generator_here = generator
        if generator_here is None:
            generator_here = self.generator
        sim_here = generator_here(sim_params, base_location=self.base_location, **generator_kwargs)
        new_fname = sim_name
        if sim_name is None:
            sim_name = str(len(self.simulations)+1) # integer
        if meta_name is None:
            meta_name = 'metadata_' + sim_name
        # Save metadata
        full_meta_name = os.path.join(self.base_location, meta_name)
        self.save_params_command(full_meta_name, sim_params)
        # Save simulation data
        full_fname = os.path.join(self.base_location, sim_name)
        self.save_command(full_fname, sim_here)
        # store data
        self.simulations[sim_name] = [sim_params, full_fname, full_meta_name]

    def retrieve_simulation(self, sim_params):
        return self.load_command(self.simulations[sim_params])


if __name__ == "__main__":
    import BaseManager

    def my_generator(k,**kwargs):
        x = np.linspace(0,1,30)
        y = np.sin(k*x)
        return np.c_[x,y]

    archive = SimulationArchiveOnLocalDisk("test", base_location="foo")
    archive.generator=my_generator
    sim_param = 0.3
    archive.generate_simulation(sim_param)
    if archive.simulation_exists_q(sim_param):
        val = archive.retrieve_simulation(sim_param)
        print(" Archive retrieved for ", sim_param,  len(val))
    
