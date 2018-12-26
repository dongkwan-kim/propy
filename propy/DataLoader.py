import os

import pickle
import math

from propy.prop import *


def dump_batch(instance, path, name):
    with open(os.path.join(path, name), 'wb') as f:
        pickle.dump(instance, f)


def assign_or_concat(base_sequence, extra_sequence):

    if base_sequence is None:
        return extra_sequence

    if isinstance(base_sequence, list):
        return base_sequence + extra_sequence
    elif isinstance(base_sequence, np.ndarray):
        return np.concatenate((base_sequence, extra_sequence))
    else:
        raise TypeError


class ActionMatrixLoader:

    __slots__ = ["path", "actions", "matrices", "selected_node_indices", "x_features", "ys", "adj"]

    def __init__(self, path: str, actions: list, path_exist_ok=True):

        self.path: str = path
        os.makedirs(self.path, exist_ok=path_exist_ok)

        self.actions: list = actions

        # (num_info, num_actions, num_selected_nodes, num_selected_nodes)
        self.matrices: list = None

        # (num_info, num_selected_nodes)
        self.selected_node_indices: list = None

        # (num_nodes, num_features)
        self.x_features: np.ndarray = None

        # (num_info, num_classes)
        self.ys: np.ndarray = None

    def __len__(self):
        assert len(self.matrices) == len(self.ys)
        return len(self.matrices)

    def __getitem__(self, item) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        :param item: id (int) of info
        :return: shape of 0: (num_actions, num_selected_nodes, num_selected_nodes),
                 shape of 1: (num_selected_nodes, num_features),
                 shape of 2: (num_classes,)
        """
        return self.matrices[item], self.x_features[self.selected_node_indices[item]], self.ys[item]

    def update_matrices_and_indices(self, matrices, selected_node_indices):
        self.matrices = assign_or_concat(self.matrices, matrices)
        self.selected_node_indices = assign_or_concat(self.selected_node_indices, selected_node_indices)

    def update_x_features(self, x_features):
        self.x_features = assign_or_concat(self.x_features, x_features)

    def update_ys(self, ys):
        self.ys = assign_or_concat(self.ys, ys)

    def dump(self, name_prefix, num_dist=1):

        assert self.matrices is not None
        assert len(self.matrices) == len(self.ys)

        # Dump xs, ys
        info_batch_size = int(math.ceil(len(self)/num_dist))
        x_batch_size = int(math.ceil(len(self.x_features)/num_dist))

        for i in range(num_dist):

            info_start, info_end = (i*info_batch_size, (i+1)*info_batch_size)
            x_start, x_end = (i*x_batch_size, (i+1)*x_batch_size)

            instance_to_dump = ActionMatrixLoader(path=self.path, actions=self.actions)
            instance_to_dump.update_matrices_and_indices(
                matrices=self.matrices[info_start:info_end],
                selected_node_indices=self.selected_node_indices[info_start:info_end],
            )
            instance_to_dump.update_x_features(self.x_features[x_start:x_end])
            instance_to_dump.update_ys(self.ys[info_start:info_end])
            dump_batch(instance=instance_to_dump, path=self.path, name="{}_{}.pkl".format(name_prefix, i))

        cprint("Dump: {} with dist {}".format(name_prefix, num_dist), "blue")

    def load(self, name_prefix):
        # Load xs, ys
        file_names_of_prefix = [f for f in os.listdir(self.path) if f.startswith(name_prefix) and f.endswith(".pkl")]

        if not file_names_of_prefix:
            return False

        for file_name in file_names_of_prefix:
            if not self._load_batch(path=self.path, name=file_name):
                cprint("Load Failed in Loading {}".format(file_names_of_prefix), "red")
                return False

        cprint("Loaded: {}".format(file_names_of_prefix), "green")
        return True

    def _load_batch(self, path, name):
        try:
            with open(os.path.join(path, name), 'rb') as f:
                loaded: ActionMatrixLoader = pickle.load(f)
                self.matrices = assign_or_concat(self.matrices, loaded.matrices)
                self.selected_node_indices = assign_or_concat(self.selected_node_indices, loaded.selected_node_indices)
                self.x_features = assign_or_concat(self.x_features, loaded.x_features)
                self.ys = assign_or_concat(self.ys, loaded.ys)
            return True
        except Exception as e:
            cprint('Load Failed: {} \n\t{}.\n'.format(os.path.join(path, name), e), "red")
            return False
