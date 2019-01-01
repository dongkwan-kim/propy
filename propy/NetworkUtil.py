import os
from typing import Callable, Any, List, Tuple

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def get_scale_free_graph(n, seed, path=None, force_save=True, **kwargs) -> nx.DiGraph:
    path = path or "."
    path_and_file = os.path.join(path, "scale_free_n{}_{}.gpickle".format(n, kwargs))
    if force_save:
        g = nx.DiGraph(nx.scale_free_graph(n=n, seed=seed, **kwargs))
        g.remove_edges_from(nx.selfloop_edges(g))
        setattr(g, "path", path_and_file)
        nx.write_gpickle(g, path_and_file)
        return g
    try:
        g = nx.read_gpickle(os.path.join(".", path_and_file))
        print("Load: {}".format(path_and_file))
    except FileNotFoundError:
        g = nx.DiGraph(nx.scale_free_graph(n=n, seed=seed, **kwargs))
        g.remove_edges_from(nx.selfloop_edges(g))
        setattr(g, "path", path_and_file)
        nx.write_gpickle(g, os.path.join(".", path_and_file))
        print("Dump: {}".format(path_and_file))
    return g


def sample_propagation_roots(g: nx.Graph,
                             num_info: int,
                             root_selection_metric: Callable = nx.closeness_centrality,
                             seed: int = None) -> np.ndarray:
    """
    :param g: Graph class of Network
    :param num_info: Number of propagation
    :param root_selection_metric: default is nx.closeness_centrality
    :param seed: Indicator of random number generation state.
    :return:
    """
    np.random.seed(seed)

    sorted_node_and_value = sorted(root_selection_metric(g).items(), key=lambda kv: -kv[1])
    sum_value = sum([v for n, v in sorted_node_and_value])
    normalized_probs = [float(v) / sum_value for n, v in sorted_node_and_value]
    chosen_nodes = np.random.choice(
        [n for n, _ in sorted_node_and_value],
        size=num_info,
        p=normalized_probs,
    )
    return chosen_nodes


def get_propagation_events(g: nx.DiGraph,
                           root: Any,
                           propagation_prob: float,
                           max_iter: int,
                           decay_rate: float = 1,
                           seed: int = None) -> List[Tuple]:
    np.random.seed(seed)

    exposed_and_not_propagated_node_to_parent = dict()
    propagated_events = list()

    propagated_events.append((1, "ROOT", root))
    exposed_and_not_propagated_node_to_parent.update({p: root for p in g.predecessors(root) if p != root})

    for t in range(2, max_iter + 2):

        # list(...keys()): To prevent "dictionary changed size during iteration"
        for node in list(exposed_and_not_propagated_node_to_parent.keys()):

            is_propagated = np.random.choice([True, False], p=[propagation_prob, 1 - propagation_prob])
            if is_propagated:

                parent = exposed_and_not_propagated_node_to_parent[node]
                del exposed_and_not_propagated_node_to_parent[node]

                propagated_events.append((t, parent, node))

                exposed_and_not_propagated_node_to_parent.update({
                    follower: node for follower in g.predecessors(node)
                    if follower not in [e[-1] for e in propagated_events]
                })

        if len(propagated_events) == len(g.nodes):
            break

        propagation_prob *= decay_rate

    return propagated_events


# Drawing

def get_highlight_node_color(all_nodes, highlight_nodes, base_color="grey", highlight_color="red"):
    node_color = []
    for n in all_nodes:
        if n in highlight_nodes:
            node_color.append(highlight_color)
        else:
            node_color.append(base_color)
    return node_color


def get_hex_color(rgb_tuple):
    return "#%02x%02x%02x" % tuple([max(0, min(x, 255)) for x in rgb_tuple])


def get_func_two_points(x1, y1, x2, y2) -> Callable:
    slope = (y2 - y1)/(x2 - x1)
    return lambda x: (y2 + slope * (x - x2))


def get_node_color_of_real_value_attr(node_to_value, low_color, high_color):
    values = node_to_value.values()
    min_val, max_val = min(values), max(values)
    red = get_func_two_points(min_val, low_color[0], max_val, high_color[0])
    green = get_func_two_points(min_val, low_color[1], max_val, high_color[1])
    blue = get_func_two_points(min_val, low_color[2], max_val, high_color[2])
    node_color = []
    for n, v in node_to_value.items():
        node_color.append(get_hex_color((int(red(v)), int(green(v)), int(blue(v)))))
    return node_color


def draw_graph(g: nx.DiGraph, **kwargs):
    """
    :param g:
    :param node_color = get_highlight_node_color(g.nodes, highlight_nodes)
    :return:
    """
    settings = {"node_size": 10, "with_labels": True, "edge_color": "#777777", "width": 0.3, "font_size": 6}
    settings.update(kwargs)
    drawing_method = kwargs["drawing_method"] if "drawing_method" in kwargs else nx.draw_kamada_kawai

    copied_g = nx.DiGraph()
    copied_g.add_nodes_from(g.nodes)
    copied_g.add_edges_from(nx.get_edge_attributes(g, "follow").keys())
    drawing_method(copied_g, **settings)
    plt.show()
    plt.clf()


# Supplement network methods

def get_matrix_of_selected_nodes(matrix, selected_indices):
    return matrix[selected_indices][:, selected_indices]


# Overrides NetworkX methods

def to_numpy_matrix(G, nodelist=None, dtype=None, order=None,
                    multigraph_weight=sum, weight='weight', nonedge=0.0,
                    value_for_non_weight_exist=1):
    A = to_numpy_array(G, nodelist=nodelist, dtype=dtype, order=order,
                       multigraph_weight=multigraph_weight, weight=weight,
                       nonedge=nonedge, value_for_non_weight_exist=value_for_non_weight_exist)
    M = np.asmatrix(A, dtype=dtype)
    return M


def to_numpy_array(G, nodelist=None, dtype=None, order=None,
                   multigraph_weight=sum, weight='weight', nonedge=0.0,
                   value_for_non_weight_exist=1):

    if nodelist is None:
        nodelist = list(G)
    nodeset = set(nodelist)
    if len(nodelist) != len(nodeset):
        msg = "Ambiguous ordering: `nodelist` contained duplicates."
        raise nx.NetworkXError(msg)

    nlen = len(nodelist)
    undirected = not G.is_directed()
    index = dict(zip(nodelist, range(nlen)))

    if G.is_multigraph():
        # Handle MultiGraphs and MultiDiGraphs
        A = np.full((nlen, nlen), np.nan, order=order)
        # use numpy nan-aware operations
        operator = {sum: np.nansum, min: np.nanmin, max: np.nanmax}
        try:
            op = operator[multigraph_weight]
        except:
            raise ValueError('multigraph_weight must be sum, min, or max')

        for u, v, attrs in G.edges(data=True):
            if (u in nodeset) and (v in nodeset):
                i, j = index[u], index[v]
                e_weight = attrs.get(weight, value_for_non_weight_exist)
                A[i, j] = op([e_weight, A[i, j]])
                if undirected:
                    A[j, i] = A[i, j]
    else:
        # Graph or DiGraph, this is much faster than above
        A = np.full((nlen, nlen), np.nan, order=order)
        for u, nbrdict in G.adjacency():
            for v, d in nbrdict.items():
                try:
                    A[index[u], index[v]] = d.get(weight, value_for_non_weight_exist)
                except KeyError:
                    # This occurs when there are fewer desired nodes than
                    # there are nodes in the graph: len(nodelist) < len(G)
                    pass

    A[np.isnan(A)] = nonedge
    A = np.asarray(A, dtype=dtype)
    return A
