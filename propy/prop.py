from collections import defaultdict
from copy import deepcopy

import networkx as nx
import numpy as np
from termcolor import cprint
import propy.NetworkUtil as nu

from typing import List, Tuple, Dict, Sequence, Callable, Generator
from pprint import pprint
import os


class NetworkPropagation(nx.DiGraph):

    def __init__(self,
                 nodes: Sequence,
                 edges: List[Tuple],
                 num_info: int,
                 propagation: Dict[int, List[Tuple]] or float,
                 propagation_kwargs: Dict = None,
                 user_actions: List[str] = None,
                 is_verbose: bool = True,
                 seed: int = 42,
                 **attr):

        self.seed = seed
        super().__init__(**attr)

        self.add_nodes_from(nodes)
        self.add_edges_from(edges, follow=1)

        self.num_info = num_info
        self.is_verbose = is_verbose

        self.info_to_propagation: Dict[int, List[Tuple]] = self._get_info_to_propagation(num_info, propagation,
                                                                                         **(propagation_kwargs or {}))
        self.info_to_attributes: Dict[int, Dict] = {info: {} for info in self.info_to_propagation.keys()}

        self.user_actions, user_actions = [], user_actions if user_actions else []
        self.user_actions.append("follow")
        self._append_user_actions_with_info("propagate")
        for action_key in user_actions:
            self._append_user_actions_with_info(action_key)

        self.event_listeners = defaultdict(list)
        self.add_event_listener(event_type="propagate", callback_func=propagate_default_listener)

    def _append_user_actions_with_info(self, action_key):
        for info in self.info_to_propagation.keys():
            self.user_actions.append("{}_{}".format(action_key, info))

    def _get_info_to_propagation(self,
                                 num_info: int,
                                 propagation: Dict[int, List[Tuple]] or float,
                                 min_path_length: int = 1,
                                 max_iter: int = None,
                                 decay_rate: float = 1.0) -> Dict[int, List[Tuple]]:

        propagation_dict = dict()

        # Generate propagation with probability p if propagation is probability (float)
        if isinstance(propagation, float):
            root_idx = 0
            while len(propagation_dict) < num_info:
                roots = nu.sample_propagation_roots(self, num_info, seed=self.seed + root_idx)
                for root in roots:
                    events = nu.get_propagation_events(
                        self, root,
                        propagation_prob=propagation,
                        max_iter=(max_iter or len(self.nodes)),
                        decay_rate=decay_rate,
                        seed=(self.seed + root_idx)
                    )

                    if len(events) >= min_path_length:
                        propagation_dict[root_idx] = events
                        root_idx += 1

                    if len(propagation_dict) >= num_info:
                        break

        return propagation_dict or propagation

    # Magic Methods

    def __repr__(self):
        return self.get_title()

    def __str__(self):
        return self.get_title()

    def __getitem__(self, item):
        return self.info_to_propagation[item]

    # Data Methods

    def get_action_matrix(self, action_key: str,
                          time_stamp: int or float = None, is_binary_repr=False, nodelist: list = None) -> np.ndarray:
        """
        :param action_key: str in self.user_actions
        :param time_stamp: Remove actions time of which is greater than time_stamp
        :param is_binary_repr: convert all positive integers to one
        :param nodelist: The rows and columns are ordered according to the nodes in nodelist.
        :return:
        """
        assert action_key in self.user_actions

        action_matrix = nu.to_numpy_matrix(self, nodelist=nodelist, weight=action_key, value_for_non_weight_exist=0)

        if time_stamp is not None:
            action_matrix = np.multiply(action_matrix, action_matrix <= time_stamp)

        if is_binary_repr:
            return nu.to_binary_repr(action_matrix)
        else:
            return action_matrix

    def get_action_matrices_and_indices(self, concerned_action_keys: List[str],
                                        base_action_keys: List[str],
                                        time_stamp: int or float = None,
                                        is_binary_repr=False) -> (np.ndarray, np.ndarray):
        """
        :param concerned_action_keys: list of str in self.user_actions
        :param base_action_keys: list of str in self.user_actions, not be considered in indices calculation
        :param time_stamp: Remove actions time of which is greater than time_stamp
        :param is_binary_repr: convert all positive integers to one
        :return tuple of matrices and indices

        if indices = [0, 1],

        [[+ + - -]   ->   [[+ +]
         [+ + - -]         [+ +]]
         [| |    ]
         [| |    ]]
        """
        ordered_full_nodes = np.asarray(self.nodes())
        concerned_nodes = set()
        for concerned_action_key in concerned_action_keys:
            for u, v in self.get_edges_of_attr(concerned_action_key):
                concerned_nodes.update((u, v))
        concerned_nodes = sorted(concerned_nodes)

        concerned_indices = [np.where(ordered_full_nodes == node)[0][0] for node in concerned_nodes]

        action_matrices = []
        for base_action_key in base_action_keys:
            action_matrix = self.get_action_matrix(base_action_key, time_stamp, is_binary_repr, concerned_nodes)
            action_matrices.append(action_matrix)

        for concerned_action_key in concerned_action_keys:
            action_matrix = self.get_action_matrix(concerned_action_key, time_stamp, is_binary_repr, concerned_nodes)
            action_matrices.append(action_matrix)

        return np.asarray(action_matrices), np.asarray(concerned_indices)

    def get_action_matrices_and_indices_of_all_info(self,
                                                    concerned_action_prefixes: List[str],
                                                    base_action_keys: List[str] = None,
                                                    time_stamp: int or float = None,
                                                    is_binary_repr=False,
                                                    is_concerned=True) -> (Sequence, Sequence):
        """
        See self.get_generator_of_action_matrices_and_indices_of_all_info
        """
        seq_of_action_matrices, seq_of_concerned_indices = [], []
        generator_of_action_matrices_and_indices = self.get_generator_of_action_matrices_and_indices_of_all_info(
            concerned_action_prefixes, base_action_keys, time_stamp, is_binary_repr, is_concerned
        )
        for matrices, indices in generator_of_action_matrices_and_indices:
            seq_of_action_matrices.append(matrices)
            seq_of_concerned_indices.append(indices)

        return seq_of_action_matrices, seq_of_concerned_indices

    def get_generator_of_action_matrices_and_indices_of_all_info(self,
                                                                 concerned_action_prefixes: List[str],
                                                                 base_action_keys: List[str] = None,
                                                                 time_stamp: int or float = None,
                                                                 is_binary_repr=False,
                                                                 is_concerned=True) -> Generator:
        """
        :param concerned_action_prefixes: list of prefixes of actions in self.user_actions
        :param base_action_keys: list of actions that are not prefixes, not be considered in indices calculation
        :param time_stamp: Remove actions time of which is greater than time_stamp
        :param is_binary_repr: convert all positive integers to one
        :param is_concerned: Boolean flag to only consider participated nodes
        :return: tuple of sequence of matrices and sequence of indices, shape of which is
                 (num_info, num_actions, num_selected_nodes, num_selected_nodes). Note that
                 num_selected_nodes can be different for each element.
        """
        base_action_keys = base_action_keys or []
        for info in self.info_to_propagation.keys():
            concerned_action_keys = ["{}_{}".format(action_prefix, info) for action_prefix in concerned_action_prefixes]
            if is_concerned:
                matrices, indices = self.get_action_matrices_and_indices(
                    concerned_action_keys=concerned_action_keys,
                    base_action_keys=base_action_keys,
                    time_stamp=time_stamp,
                    is_binary_repr=is_binary_repr
                )
            else:
                raise NotImplementedError

            yield matrices, indices

    def get_sequence_of_info_attr(self, attr, encode_func: Callable = None) -> np.ndarray:
        """
        :param attr: attribute of info
        :param encode_func: function to converts attribute value
        :return: this is for y sequence for data pair (x, y)
        """
        encode_func = encode_func or (lambda x: x)
        sequence = [encode_func(self.get_info_attr(info, attr)) for info in self.info_to_propagation]
        return np.asarray(sequence)

    def dump(self, file_prefix, path=None):
        path = path or "."
        file_path_and_name = os.path.join(path, "{}_{}.pkl".format(file_prefix, self.get_title()))
        nx.write_gpickle(self, file_path_and_name)
        cprint("Dump: {}".format(file_path_and_name), "blue")

    @classmethod
    def load(cls, file_name_or_prefix, path=None):
        path = path or "."
        try:
            file_path_and_name = os.path.join(path, file_name_or_prefix)
            loaded: NetworkPropagation = nx.read_gpickle(file_path_and_name)
        except Exception as e:
            file_names_starts_with_prefix = [f for f in os.listdir(path)
                                             if f.startswith(file_name_or_prefix) and f.endswith(".pkl")]
            if file_names_starts_with_prefix:
                file_name = file_names_starts_with_prefix[-1]
                file_path_and_name = os.path.join(path, file_name)
                loaded: NetworkPropagation = nx.read_gpickle(file_path_and_name)
            else:
                file_path_and_name = "NOT FOUND / {}".format(file_name_or_prefix)
                loaded = False
        cprint("Load: {}".format(file_path_and_name), "green")
        return loaded

    # Propagation Methods

    def get_last_time_of_propagation(self) -> int or float:
        last_times = []
        for _, propagation in self.info_to_propagation.items():
            last_times.append(propagation[-1][0])
        return max(last_times)

    def simulate_propagation(self):
        for info, propagation in self.info_to_propagation.items():
            for t, parent_id, node_id in propagation:
                self._run_event_listener("propagate", (t, parent_id, node_id), info=info)

    # Attributes Manipulation Methods

    def add_action(self, u, v, action_key, value):
        """
        :param u: main node (subject of the action, e.g. "u follows v", "u shares v's", "u flags v's")
        :param v: sub node
        :param action_key: attribute key of edges
        :param value: attribute value of edges
        """
        self.add_edge(u, v, **{action_key: value})

    def get_info_attr(self, info, attr=None):
        if attr is None:
            return self.info_to_attributes[info]
        else:
            return self.info_to_attributes[info][attr]

    def set_info_attr(self, info, attr, val):
        self.info_to_attributes[info][attr] = val

    def get_edges_of_attr(self, attr) -> Dict:
        return nx.get_edge_attributes(self, attr)

    def get_nodes_of_attr(self, attr) -> Dict:
        return nx.get_node_attributes(self, attr)

    def set_attr_of_edge(self, edge: Tuple, attr, val):
        nx.set_edge_attributes(self, {edge: val}, name=attr)

    def set_attr_of_node(self, node, attr, val):
        nx.set_node_attributes(self, {node: val}, name=attr)

    def set_attr_of_edges(self, edge_to_val: Dict, attr):
        nx.set_edge_attributes(self, edge_to_val, name=attr)

    def set_attr_of_nodes(self, node_to_val: Dict, attr):
        nx.set_node_attributes(self, node_to_val, name=attr)

    # Event Listener Methods

    def add_event_listener(self, event_type: str, callback_func: Callable, **kwargs):
        """
        :param event_type: str from ["propagate",]
        :param callback_func: some_func(network_propagation: NetworkPropagation, event: Tuple, info: int, **kwargs)
        :param kwargs: kwargs for callback_func
        :return:
        """
        self.event_listeners[event_type].append((callback_func, kwargs))

    def _run_event_listener(self, given_event_type: str, event: Tuple, info: int):
        for callback_func, kwargs in self.event_listeners[given_event_type]:
            callback_func(
                network_propagation=self,
                event=event,
                info=info,
                **kwargs
            )

    # NetworkX Overrides

    def predecessors(self, n, feature=None):
        if feature is None:
            return super().predecessors(n)
        else:
            return [p for p, features in self._pred[n].items() if feature in features]

    def copy(self, as_view=False):
        if as_view:
            raise NotImplementedError
        return deepcopy(self)

    # Util Methods

    def get_roots(self) -> List[int]:
        roots = []
        for info, history in self.info_to_propagation.items():
            roots.append(history[0][-1])  # (t, p, n)
        return roots

    def draw_graph(self, color_type: str = "root", **kwargs):

        if color_type is "root":
            roots = self.get_roots()
            node_color = nu.get_highlight_node_color(self.nodes, roots)
        elif color_type is "real_value_attr":
            node_to_attr = self.get_nodes_of_attr(kwargs["attr"])
            low_color = (255, 0, 0) if "low_color" not in kwargs else kwargs["low_color"]
            high_color = (0, 0, 255) if "high_color" not in kwargs else kwargs["high_color"]
            node_color = nu.get_node_color_of_real_value_attr(node_to_attr, low_color, high_color)
        else:
            raise ValueError

        nu.draw_graph(self, node_color=node_color, **kwargs)

    def get_title(self):
        key_attributes = {
            "num_info": self.num_info,
            "nodes": self.number_of_nodes(),
            "edges": self.number_of_edges(),
            "seed": self.seed,
        }
        return "_".join(["{}_{}".format(k, v) for k, v in key_attributes.items()])

    def get_num_info(self):
        return len(self.info_to_propagation)

    def pprint_propagation(self):
        pprint(self.info_to_propagation)


def propagate_default_listener(network_propagation: NetworkPropagation, event: Tuple, info: int, **kwargs):
    current_time, parent_id, node_id = event
    propagate_key = f"propagate_{info}"

    # Mark propagate attribute to node
    network_propagation.set_attr_of_node(node_id, attr=propagate_key, val=current_time)

    # Note that we consider that 'propagate' is a consequence of {node_id}'s action.
    if parent_id != "ROOT":
        network_propagation.add_action(node_id, parent_id, action_key=propagate_key, value=current_time)
