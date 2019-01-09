import os

import pickle
import math

from propy.prop import *
from propy.DataUtil import *


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

    __slots__ = ["path", "actions", "matrices_in_list_form", "selected_node_indices", "x_features", "ys",
                 "num_features", "num_classes", "is_coo_repr", "is_binary_repr"]

    def __init__(self, path: str, actions: list, is_coo_repr=True, path_exist_ok=True):

        self.path: str = path
        os.makedirs(self.path, exist_ok=path_exist_ok)

        self.actions: list = actions

        # Meta information
        self.num_features = None
        self.num_classes = None
        self.is_coo_repr = is_coo_repr

        # (num_info, num_actions, 3), 3 = [i, j, val]
        self.matrices_in_list_form: list = None

        # (num_info, num_selected_nodes)
        self.selected_node_indices: list = None

        # (num_nodes, num_features)
        self.x_features: np.ndarray = None

        # (num_info, num_classes)
        self.ys: np.ndarray = None

    def __len__(self):
        assert len(self.matrices_in_list_form) == len(self.ys)
        return len(self.matrices_in_list_form)

    def __getitem__(self, item) -> Tuple:
        """
        :param item: id (int) of info
        :return: tuple of length 3
        if is_coo_repr:
            shape of 0: (num_actions, 2, num_selected_edges),
            shape of 1: (num_selected_nodes, num_features),
            shape of 2: (num_classes,)
        else:
            shape of 0: (num_actions, num_selected_nodes, num_selected_nodes),
            shape of 1: (num_selected_nodes, num_features),
            shape of 2: (num_classes,)
        TODO: Support slice as an item.
        TODO: Support not is_binary_repr for is_coo_repr & is_binary_repr for not is_coo_repr
        """
        indices = self.selected_node_indices[item]
        if self.is_coo_repr:
            matrices_in_list_form = self.matrices_in_list_form[item]
            matrices_coo = [list_to_coo(lst) for lst in matrices_in_list_form]
            return matrices_coo, self.x_features[indices], self.ys[item]
        else:
            matrices = np.asarray([list_to_matrix(lst, size=len(indices)) for lst in self.matrices_in_list_form[item]])
            return matrices, self.x_features[indices], self.ys[item]

    def get_batch_generator(self, batch_size=None,
                            shuffle=False, seed=None,
                            is_train=None, train_ratio=0.8) -> Generator:

        data_m, data_x, data_y = [], [], []

        indexes = np.asarray(range(len(self)))

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indexes)

        if is_train is not None:
            train_idx = int(len(self) * train_ratio)
            if is_train:
                indexes = indexes[:train_idx]
            else:
                indexes = indexes[train_idx:]

        for i, idx in enumerate(indexes):

            m, x, y = self[idx]

            data_m.append(m)
            data_x.append(x)
            data_y.append(y)

            if batch_size and (i + 1) % batch_size == 0:
                yield data_m, data_x, data_y
                data_m, data_x, data_y = [], [], []

        yield data_m, data_x, data_y

    def update_matrices_and_indices(self, matrices_sequence, selected_node_indices, convert_to_list=True):

        if convert_to_list:
            matrices_sequence_in_list_form = []
            for matrices in matrices_sequence:
                matrices_sequence_in_list_form.append([matrix_to_list(mat) for mat in matrices])
            self.matrices_in_list_form = assign_or_concat(self.matrices_in_list_form, matrices_sequence_in_list_form)
        else:
            self.matrices_in_list_form = assign_or_concat(self.matrices_in_list_form, matrices_sequence)

        self.selected_node_indices = assign_or_concat(self.selected_node_indices, selected_node_indices)

    def update_x_features(self, x_features):
        if self.x_features is None:
            self.num_features = x_features[0].shape[0]
        self.x_features = assign_or_concat(self.x_features, x_features)

    def dynamic_update_x_features(self, update_func: Callable, **kwargs):
        """
        :param update_func: function that takes
                            *("matrices_in_list_form", "selected_node_indices", "x_features") & **kwargs
        """
        prev_shape = self.x_features.shape
        self.x_features = update_func(
            matrices_in_list_form=self.matrices_in_list_form,
            selected_node_indices=self.selected_node_indices,
            x_features=self.x_features,
            **kwargs,
        )
        assert prev_shape == self.x_features.shape

    def update_ys(self, ys):
        if self.ys is None:
            self.num_classes = ys[0].shape[0]
        self.ys = assign_or_concat(self.ys, ys)

    def dump(self, name_prefix, num_subfiles=1):

        assert self.matrices_in_list_form is not None
        assert len(self.matrices_in_list_form) == len(self.ys)

        # Dump xs, ys
        info_batch_size = int(math.ceil(len(self) / num_subfiles))
        x_batch_size = int(math.ceil(len(self.x_features) / num_subfiles))

        for i in range(num_subfiles):

            info_start, info_end = (i*info_batch_size, (i+1)*info_batch_size)
            x_start, x_end = (i*x_batch_size, (i+1)*x_batch_size)

            instance_to_dump = ActionMatrixLoader(path=self.path, actions=self.actions)
            instance_to_dump.update_matrices_and_indices(
                matrices_sequence=self.matrices_in_list_form[info_start:info_end],
                selected_node_indices=self.selected_node_indices[info_start:info_end],
                convert_to_list=False,
            )
            instance_to_dump.update_x_features(self.x_features[x_start:x_end])
            instance_to_dump.update_ys(self.ys[info_start:info_end])
            dump_batch(instance=instance_to_dump, path=self.path, name="{}_{}.pkl".format(name_prefix, i))

        cprint("Dump: {} with num_subfiles {}".format(name_prefix, num_subfiles), "blue")

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
                self.matrices_in_list_form = assign_or_concat(self.matrices_in_list_form, loaded.matrices_in_list_form)
                self.selected_node_indices = assign_or_concat(self.selected_node_indices, loaded.selected_node_indices)
                self.x_features = assign_or_concat(self.x_features, loaded.x_features)
                self.ys = assign_or_concat(self.ys, loaded.ys)
            return True
        except Exception as e:
            cprint('Load Failed: {} \n\t{}.\n'.format(os.path.join(path, name), e), "red")
            return False
