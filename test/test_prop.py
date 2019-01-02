import unittest
from propy.prop import *
from propy.EventListenerExample import randomly_flag
from propy.DataUtil import *
from propy.DataLoader import *


class TestProp(unittest.TestCase):

    def test_prop_basic(self):
        g = nu.get_scale_free_graph(n=5, seed=42)
        prop = NetworkPropagation(g.nodes, g.edges(),
                                  user_actions=["flag"], num_info=1, propagation=0.3,
                                  is_verbose=False, seed=42)
        prop.add_event_listener(
            event_type="propagate",
            callback_func=randomly_flag,
            flag_prob=0.6,
            seed=42,
        )
        prop.simulate_propagation()

        self.assertEqual(len(prop), 5)
        self.assertTrue(np.array_equal(
            prop.get_action_matrix("follow"),
            np.asarray([[0.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0]])
        ))
        self.assertTrue(np.array_equal(
            prop.get_action_matrix("propagate_0"),
            np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 4.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0]])
        ))
        self.assertTrue(np.array_equal(
            prop.get_action_matrix("flag_0"),
            np.asarray([[0.0, 1.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 3.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0]])
        ))
        self.assertTrue(np.array_equal(
            prop.get_action_matrix("flag_0", time_stamp=2),
            np.asarray([[0.0, 1.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0]])
        ))

    def test_prop_dump_and_load(self):
        g = nu.get_scale_free_graph(n=20, seed=42)
        prop = NetworkPropagation(g.nodes, g.edges(),
                                  user_actions=["flag"], num_info=2, propagation=0.3,
                                  is_verbose=False, seed=42)
        prop.add_event_listener(
            event_type="propagate",
            callback_func=randomly_flag,
            flag_prob=0.6,
            seed=42,
        )
        prop.simulate_propagation()
        prop.dump("test")

        loaded_prop = NetworkPropagation.load("test")
        self.assertTrue(np.array_equal(
            loaded_prop.get_action_matrix("follow"),
            prop.get_action_matrix("follow"),
        ))
        self.assertTrue(np.array_equal(
            loaded_prop.get_action_matrix("propagate_0"),
            prop.get_action_matrix("propagate_0"),
        ))
        self.assertTrue(np.array_equal(
            loaded_prop.get_action_matrix("flag_0"),
            prop.get_action_matrix("flag_0"),
        ))
        self.assertTrue(np.array_equal(
            loaded_prop.get_action_matrix("flag_0", time_stamp=2),
            loaded_prop.get_action_matrix("flag_0", time_stamp=2),
        ))

    def test_prop_kwargs(self):
        g = nu.get_scale_free_graph(n=100, seed=42)
        prop = NetworkPropagation(g.nodes, g.edges(),
                                  user_actions=["flag"], num_info=1, propagation=1.0,
                                  propagation_kwargs={"decay_rate": 0},
                                  is_verbose=False, seed=42)
        prop.add_event_listener(
            event_type="propagate",
            callback_func=randomly_flag,
            flag_prob=0.6,
            seed=42,
        )
        prop.simulate_propagation()

        for x, (u, _) in zip(sorted(prop.predecessors(prop.get_roots()[0], feature="follow")),
                             sorted(prop.get_edges_of_attr("propagate_0").keys(), key=lambda t: t[0])):
            self.assertEqual(x, u)

        prop = NetworkPropagation(g.nodes, g.edges(),
                                  user_actions=["flag"], num_info=1, propagation=1.0,
                                  propagation_kwargs={"max_iter": 1},
                                  is_verbose=False, seed=42)
        prop.add_event_listener(
            event_type="propagate",
            callback_func=randomly_flag,
            flag_prob=0.6,
            seed=42,
        )
        prop.simulate_propagation()

        for x, (u, _) in zip(sorted(prop.predecessors(prop.get_roots()[0], feature="follow")),
                             sorted(prop.get_edges_of_attr("propagate_0").keys(), key=lambda t: t[0])):
            self.assertEqual(x, u)

    def test_prop_default_event_listener(self):
        g = nu.get_scale_free_graph(n=100, seed=42)
        prop = NetworkPropagation(g.nodes, g.edges(),
                                  user_actions=["flag"], num_info=1, propagation=1.0,
                                  propagation_kwargs={"decay_rate": 0},
                                  is_verbose=False, seed=42)
        prop.simulate_propagation()
        root = prop.get_roots()[0]
        for x, (k, t) in zip(sorted(prop.predecessors(root, feature="follow") + [root]),
                             sorted(prop.get_nodes_of_attr("propagate_0").items())):
            self.assertEqual(x, k)
            if x != root:
                self.assertEqual(t, 2)
            else:
                self.assertEqual(t, 1)

    def test_data_loader(self):
        concerned_action_prefixes = ["flag", "propagate"]
        base_action_keys = ["follow"]
        g = nu.get_scale_free_graph(n=20, seed=42)
        prop = NetworkPropagation(g.nodes, g.edges(),
                                  user_actions=["flag"], num_info=2, propagation=0.3,
                                  is_verbose=False, seed=42)
        prop.simulate_propagation()

        prop.set_info_attr(info=0, attr="is_fake", val=True)
        prop.set_info_attr(info=1, attr="is_fake", val=False)

        matrices, indices = prop.get_action_matrices_and_indices_of_all_info(
            concerned_action_prefixes, base_action_keys,
        )

        data_loader = ActionMatrixLoader(path=".", actions=base_action_keys + concerned_action_prefixes)
        if not data_loader.load("test_loader"):
            data_loader.update_matrices_and_indices(
                matrices_sequence=matrices,
                selected_node_indices=indices,
            )
            data_loader.update_x_features(ones_feature(prop.number_of_nodes(), 5))
            data_loader.update_ys(prop.get_sequence_of_info_attr("is_fake",
                                                                 encode_func=lambda x: np.eye(2)[int(not x)]))
            data_loader.dump(name_prefix="test_loader")

        mats_1, xs_1, ys_1 = data_loader[1]
        self.assertTrue(np.array_equal(
            mats_1[0],
            np.asarray([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
        ))
        self.assertEqual(data_loader.actions, ['follow', 'flag', 'propagate'])
        self.assertEqual(xs_1.shape, (17, 5))
        self.assertTrue(np.array_equal(ys_1, np.asarray([0, 1])))


if __name__ == '__main__':
    unittest.main()
