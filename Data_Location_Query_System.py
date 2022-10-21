import numpy as np
from tqdm import tqdm
from Data_Utils import serialize, un_serialize
from sklearn.neighbors import BallTree


class LocationQuerySystem:
    def __init__(self):
        self.gps = []
        self.tree = None
        self.n_neighbor = None
        self.ngb_locs = None

    def build_tree(self, dataset):
        self.gps = np.zeros((len(dataset.idx2gps) - 1, 2), dtype=np.float64)
        for idx, (lat, lon) in dataset.idx2gps.items():
            if idx != 0:
                self.gps[idx - 1] = [lat, lon]
        self.tree = BallTree(
            self.gps,
            leaf_size=1,
            metric='haversine'
        )

    def prefetch(self, n_neighbor):
        self.n_neighbor = n_neighbor
        self.ngb_locs = np.zeros((self.gps.shape[0], n_neighbor), dtype=np.int32)
        for idx, gps in tqdm(enumerate(self.gps), total=len(self.gps), leave=True):
            gps = gps.reshape(1, -1)
            _, ngb_locs = self.tree.query(gps, n_neighbor + 1)
            ngb_locs = ngb_locs[0, 1:]
            ngb_locs += 1
            self.ngb_locs[idx] = ngb_locs

    def get(self, loc_idx, k):
        if k <= self.n_neighbor:
            knn_loc = self.ngb_locs[loc_idx - 1][:k]
            return knn_loc
        else:
            gps = self.gps[loc_idx].reshape(1, -1)
            _, knn_loc = self.tree.query(gps, k + 1)
            knn_loc = knn_loc[0, 1:]
            knn_loc += 1
            return knn_loc

    def save(self, path):
        data = {
            "gps": self.gps,
            "tree": self.tree,
            "n_neighbor": self.n_neighbor,
            "neighbor_locs": self.ngb_locs
        }
        serialize(data, path)

    def load(self, path):
        data = un_serialize(path)
        self.gps = data["gps"]
        self.tree = data["tree"]
        self.n_neighbor = data["n_neighbor"]
        self.ngb_locs = data["neighbor_locs"]