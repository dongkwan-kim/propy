import random

from propy.prop import *


# Callback functions
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


if __name__ == '__main__':
    g = nu.get_scale_free_graph(n=5, seed=42)
    prop = NetworkPropagation(g.nodes, g.edges(),
                              user_actions=["flag"], num_info=1, propagation=0.3, seed=42)
    prop.add_event_listener(
        event_type="propagate",
        callback_func=randomly_flag,
        flag_prob=0.6,
        seed=42,
    )
    prop.simulate_propagation()

    print(prop.get_roots())
    print(prop.get_action_matrix("follow"))
    print(prop.get_action_matrix("propagate_0"))
    print(prop.get_action_matrix("flag_0", time_stamp=2))
    print(prop.get_action_matrix("flag_0"))
    print(prop.get_edge_of_attr("propagate_0"))
    prop.pprint_propagation()
    prop.draw_graph()
