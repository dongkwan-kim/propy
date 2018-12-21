import unittest
from propy.prop import *
from propy.EventListenerExample import randomly_flag


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
            np.asarray([[0.0, 0.0, 3.0, 3.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 4.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
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

        for x, (_, v) in zip(sorted(prop.predecessors(prop.get_roots()[0], feature="follow")),
                             sorted(prop.get_edges_of_attr("propagate_0").keys(), key=lambda t: t[1])):
            self.assertEqual(x, v)

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

        for x, (_, v) in zip(sorted(prop.predecessors(prop.get_roots()[0], feature="follow")),
                             sorted(prop.get_edges_of_attr("propagate_0").keys(), key=lambda t: t[1])):
            self.assertEqual(x, v)


if __name__ == '__main__':
    unittest.main()
