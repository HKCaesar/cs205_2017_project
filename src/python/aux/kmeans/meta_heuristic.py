import abc
import pandas as pd

class kmeans(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 n_clusters=2,
                 n_init=100,
                 max_iter = 100):

        self.n_clusters = 2
        self.n_init = n_init
        self.max_iter = max_iter

        self.data = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.timing_ = None
        self.engine_ = None

    def load_data(self, filename):
        self.data = pd.read_csv(filename).values

    @abc.abstractmethod
    def engine(self):
        pass



    def run(self):
        for k in range(n_init):


    def time(self):
        #save json with important variables
        pass



def distortion(data,labels, centers):
    