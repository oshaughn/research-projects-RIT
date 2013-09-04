

class Distribution:
    def __init__(self,variables,bounds,pdf,sampler):
        self.variables =variables
        self.bounds = bounds
        self.pdf = pdf
        self.sampler = sampler

    def random(self):
        self.sampler()

    def pdf(self,x):
        self.pdf(x)

