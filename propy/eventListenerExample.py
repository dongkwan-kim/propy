from typing import Tuple

import numpy as np

from propy.prop import NetworkPropagation


def randomly_flag(network_propagation: NetworkPropagation, event: Tuple, info: int, **kwargs):

    current_time, parent_id, node_id = event
    np.random.seed(kwargs["seed"] + current_time)

    followers = network_propagation.predecessors(node_id, "follow")
    exposed, flagged = 0, 0
    flag_users = []
    for follower_id in followers:
        exposed += 1
        if np.random.random() < kwargs["flag_prob"]:
            flagged += 1
            network_propagation.add_action(node_id, follower_id, f"flag_{info}", current_time)
            flag_users.append(follower_id)

    print("Info: {} at time {}, Exposed: {}, Flagged: {} by {}".format(
        info, current_time, exposed, flagged, flag_users
    ))
