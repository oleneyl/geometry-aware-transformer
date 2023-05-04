import numpy as np
import scipy.sparse as sp
from scipy.special import sph_harm

class DataContainer:
    def __init__(self, filename, cutoff, target_keys, chemical_sequence_length=32, atomic_length=32, angular=False,
                is_contain_force=False, cg_size=1, basis_distance=30):
        self.chemical_sequence_length = 32
        self.atomic_length = 16
        self.angular = angular
        self.cg_size = cg_size
        self.is_contain_force = is_contain_force
        self.basis_distance = basis_distance

        data_dict = np.load(filename, allow_pickle=True)
        self.cutoff = cutoff
        self.target_keys = target_keys
        for key in ['id', 'N', 'Z', 'R']:
            if key in data_dict:
                setattr(self, key, data_dict[key])
            else:
                setattr(self, key, None)
        self.targets = np.stack([data_dict[key] for key in self.target_keys], axis=1)

        if self.N is None:
            self.N = np.zeros(len(self.targets), dtype=np.int32)
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])

        assert self.R is not None

    def __len__(self):
        return self.targets.shape[0]

    def _get_distance_matrix(self):
        pass

    def __getitem__(self, idx):
        if type(idx) is int or type(idx) is np.int64:
            idx = [idx]

        data = [self.__get_item_of_single_index(i) for i in idx]
        keys = data[0].keys()

        output = {k : np.stack([datum[k] for datum in data]) for k in keys}
        if self.angular:
            output['pad_mask'] = output['pad_mask'][:, np.newaxis, :]
        else:
            output['pad_mask'] = output['pad_mask'][:, np.newaxis, np.newaxis, :]
        return output

    def __get_item_of_single_index(self, idx):
        assert type(idx) is int or type(idx) is np.int64

        data = {}
        data['targets'] = self.targets[idx]
        data['id'] = self.id[idx]
        data['N'] = self.N[idx]
        adj_matrices = []

        data['Z'] = np.zeros(self.chemical_sequence_length, dtype=np.int32)  ## Atom type
        data['R'] = np.zeros([self.chemical_sequence_length, 3], dtype=np.float32)  ## Atom position
        if self.is_contain_force:
            data['F'] = np.zeros([self.chemical_sequence_length, 3], dtype=np.float32)

        nend = 0
        
        n = data['N']
        nstart = nend
        nend = nstart + n

        if self.Z is not None:
            data['Z'][nstart:nend] = self.Z[self.N_cumsum[idx]:self.N_cumsum[idx + 1]]

        R = self.R[self.N_cumsum[idx]:self.N_cumsum[idx + 1]]
        data['R'][nstart:nend] = R

        if self.is_contain_force:
            F = self.F[self.N_cumsum[idx]:self.N_cumsum[idx + 1]]
            data['F'][nstart:nend] = F


        Dij = np.linalg.norm(data['R'][:, None, :] - data['R'][None, :, :], axis=-1)

        pad_mask = np.ones([self.chemical_sequence_length])
        pad_mask[nstart:nend] = 0.0

        data['atom_type'] = data['Z']
        data['orbit_coeff'] = np.eye(data['Z'].shape[-1])
        
        data['distance'] = Dij
        data['pad_mask'] = pad_mask
        data['output_mask'] = 1.0 - pad_mask

        data['extract_matrix'] = np.diag(np.ones([self.chemical_sequence_length]))

        return data


def spherical_coord(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:, :, 0] ** 2 + xyz[:, :, 1] ** 2 + np.finfo(float).eps
    ptsnew[:, :, 0] = np.sqrt(xy + xyz[:, :, 2] ** 2)
    ptsnew[:, :, 1] = np.pi / 2 - np.abs(
        np.pi / 2 - np.arctan2(np.sqrt(xy), xyz[:, :, 2]))  # for elevation angle defined from Z-axis down

    ptsnew[:, :, 2] = np.arctan2(xyz[:, :, 1], xyz[:, :, 0])
    return ptsnew
