import os

import pickle
import math

from propy.prop import *


def dump_batch(instance, path, name):
    with open(os.path.join(path, name), 'wb') as f:
        pickle.dump(instance, f)


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

    def set_xy(self, matrices, selected_node_indices, x_features, ys):
        assert len(matrices) == len(selected_node_indices) == len(ys)
        self.matrices = matrices
        self.selected_node_indices = selected_node_indices
        self.x_features = x_features
        self.ys = ys

    def dump(self, name_prefix, num_dist=1):

        assert self.matrices is not None
        assert len(self.matrices) == len(self.ys)

        # Dump adjacency
        meta_instance = ActionMatrixLoader(path=self.path, actions=self.actions)
        dump_batch(instance=meta_instance, path=self.path, name="{}_meta.pkl".format(name_prefix))

        # Dump xs, ys
        info_batch_size = int(math.ceil(len(self)/num_dist))
        x_batch_size = int(math.ceil(len(self.x_features)/num_dist))

        for i in range(num_dist):

            info_start, inf_end = (i*info_batch_size, (i+1)*info_batch_size)
            x_start, x_end = (i*x_batch_size, (i+1)*x_batch_size)

            instance_to_dump = ActionMatrixLoader(path=self.path, actions=self.actions)
            instance_to_dump.set_xy(
                matrices=self.matrices[info_start:inf_end],
                selected_node_indices=self.selected_node_indices[info_start:inf_end],
                x_features=self.x_features[x_start:x_end],
                ys=self.ys[info_start:inf_end],
            )
            dump_batch(instance=instance_to_dump, path=self.path, name="{}_{}.pkl".format(name_prefix, i))

        cprint("Dump: {} with dist {}".format(name_prefix, num_dist), "blue")

    def load(self, name_prefix):

        # Load adjacency
        try:
            with open(os.path.join(self.path, "{}_meta.pkl".format(name_prefix)), 'rb') as f:
                loaded: ActionMatrixLoader = pickle.load(f)
                assert loaded.actions == self.actions
                assert loaded.path == self.path
        except Exception as e:
            cprint('Load Failed: {} \n\t{}.\n'.format(name_prefix, e), "red")
            return False

        # Load xs, ys
        file_names_of_prefix = [f for f in os.listdir(self.path)
                                if f.startswith(name_prefix) and f.endswith(".pkl") and "meta" not in f]
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
                if self.matrices is None:
                    self.matrices = loaded.matrices
                    self.selected_node_indices = loaded.selected_node_indices
                    self.x_features = loaded.x_features
                    self.ys = loaded.ys
                else:
                    self.matrices += loaded.matrices
                    self.selected_node_indices += loaded.selected_node_indices
                    self.x_features = np.concatenate((self.x_features, loaded.x_features))
                    self.ys = np.concatenate((self.ys, loaded.ys))
            return True
        except Exception as e:
            cprint('Load Failed: {} \n\t{}.\n'.format(os.path.join(path, name), e), "red")
            return False
