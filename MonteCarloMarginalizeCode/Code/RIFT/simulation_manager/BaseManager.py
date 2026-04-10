

import numpy as np
import os

def easy_write(fname,content_lines):
    with open(fname, 'w') as f:
        f.write(str(content_lines) +"\n")

class SimulationArchive:
    """
    SimulationArchive: tool to allow (a) storing list of 'simulations', (b) querying if present, (c) generating if needed,from a generation function
    
         -simulation_exists_q : check if simulation present
         -retrieve_simulation :
    INTERNAL DATA STRUCTURES:
         - simulations : key/value pairs, keys are simulation params, values are simulation results. NOT LIKE OTHER REALIZATIONS.
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
        INTERNAL DATA STRUCTURES
            - simulations: key/value pairs, value is [sim_params, file_name, meta_name]
    """
    def __init__(self, name=None,storage_method=None,**kwargs):
        self.base_location = None
        self.save_command = np.savetxt
        self.save_params_command = easy_write
        self.load_simulation = np.loadtxt
        self.load_metadata = np.loadtxt
        self.valid_simulation_file = lambda x: os.path.exists(x) and not(x.startswith('metadata_'))
        self.valid_metadata_file = lambda x: os.path.exists(x) and (x.startswith('metadata_'))
        self.params_same_q = lambda x,y : str(x)==str(y)   # check if parameters are IDENTICAL
        if not(hasattr(self, '_internal_simulations_have_sub_directories')):
            self._internal_simulations_have_sub_directories = False
        if not(hasattr(self, '_internal_annotator')):
            self._internal_annotator = None  # internal function to return lists, to append to the structure.
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
        full_meta_fname = os.path.join(self.base_location, meta_name)
        self.save_params_command(full_meta_fname, sim_params)
        # Save simulation data
        if self._internal_simulations_have_sub_directories:
            # make directory
            full_fname = os.path.join(self.base_location, sim_name)
            if not os.path.exists(full_fname):
                os.mkdir(full_fname)
            # make full filename. Note full filename saved, so we can access the sved simulation file right now
            full_fname = os.path.join(self.base_location, sim_name,sim_name)
        else:
            full_fname = os.path.join(self.base_location, sim_name)
        self.save_command(full_fname, sim_here)
        # Add extra internal annotations if any to end
        extra_stuff = []
        if self._internal_annotator:
            extra_stuff += self._internal_annotator(params=sim_params, sim_path=full_fname, sim_meta_path=full_meta_fname)
        # store data
        self.simulations[sim_name] = [sim_params, full_fname, full_meta_fname] + extra_stuff

    def index_simulation(self, sim_params):
        # finds internal index for param.
        for indx, name in enumerate(self.simulations):
            if self.params_same_q( self.simulations[name][0], sim_params):
                return name   # internal index
        raise Exception(" No such simulation  for ", sim_params)

    def simulation_exists_q(self, sim_params):
        for indx, name in enumerate(self.simulations):
            if self.params_same_q(self.simulations[name][0], sim_params):
                return True
        return False
            

    def retrieve_simulation(self, sim_params):
        sim_idx  = self.index_simulation(sim_params)
        print(sim_idx)
        return self.load_simulation(self.simulations[sim_idx][1])
    
    def retrieve_simulation_by_name(self, internal_name):
        # Index to find 
        return self.load_simulation(self.simulations[sim_params][1]) # load in file


def append_queue_default(params=None,sim_path=None, sim_meta_path=None, sim_annotation=None):
    if sim_annotation is None:
        return [{'queue_status': 'ready'}]   # idea :'ready', 'running', 'complete', 'stuck'
    else:
        return [{'queue_status': 'complete'}]   # idea :'ready', 'running', 'complete', 'stuck'
def dummy_true(**kwargs):
    return True
    
class SimulationArchiveOnLocalDiskExternalQueue(SimulationArchiveOnLocalDisk):
    """
        User can specify a 'base location'.  Simulations are associated with files on disk.
        Simulations are *performed* by an external queue manager, and *ready* at some point TBD.
        Instead of *generating* simulations, we *queue* simulations -- so we have  a simulations_queue
       INTERNAL DATA STRUCTURES
         self.simulations = [sim_params, fname, fname_meta, annotations] # annotator can do whatever it wants, in element 3
    """
    def __init__(self, **kwargs):
        self._internal_annotator = append_queue_default   # also responsible for status updates ! 
        self._internal_check_complete = dummy_true
        super().__init__(**kwargs)

    def simulation_status_q(self, sim_params):
        # Check if simulation complete
        sim_idx = self.index_simulation(sim_params)
        sim_here =self.simulations[sim_idx]
        print(sim_here)
        return self._internal_check_complete(params=sim_here[0], sim_path=sim_here[1], sim_meta_path=sim_here[2], sim_annotation=sim_here[3:])  # the annotator default, note user has own burden to interpret depending on annotator
    def simulation_status_update(self, sim_params):
        # update simulation status. Will update for all
        sim_idx = self.index_simulation(sim_params)        
        sim_here = self.simulations[sim_idx]
        new_annotation  = self._internal_annotator(params=sim_here[0], sim_path=sim_here[1], sim_meta_path=sim_here[2], sim_annotation=sim_here[3])
        sim_here[3:] = new_annotation # update
        self.simulations[sim_idx] = sim_here  # update
        

    
if __name__ == "__main__":
#    import BaseManager

    def my_generator(k,**kwargs):
        x = np.linspace(0,1,30)
        y = np.sin(k*x)
        return np.c_[x,y]

    print(" Test 0: OnLocalDisk ")
    archive = SimulationArchiveOnLocalDisk("test", base_location="foo", _internal_annotator=append_queue_default)
    archive.generator=my_generator
    sim_param = 0.3
    archive.generate_simulation(sim_param)
    archive.generate_simulation(sim_param+1)
    archive.generate_simulation(sim_param+2)
    print(" Internal simulation archive ", archive.simulations)
    if archive.simulation_exists_q(sim_param):
        val = archive.retrieve_simulation(sim_param)
        print(" Archive retrieved for ", sim_param,  len(val))

    print(" Test 1: OnLocalDiskExternalQueue ")
    archive = SimulationArchiveOnLocalDiskExternalQueue(name="test", base_location="bar")
    archive.generator=my_generator
    sim_param = 0.5
    archive.generate_simulation(sim_param)
    archive.generate_simulation(sim_param+1)
    print("Internal simulation archive ", archive.simulations)
    my_status = archive.simulation_status_q(0.5)
    if my_status:
        archive.simulation_status_update(0.5)
        print(archive.simulations)   # test we can flag simulations as complete
