import unittest
from propy.prop import *
from propy.eventListenerExample import randomly_flag


class TestProp(unittest.TestCase):

    def test_prop(self):
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

        prop.dump("test")

        loaded_prop = NetworkPropagation.load("test")

        self.assertTrue(np.array_equal(
            loaded_prop.get_action_matrix("follow"),
            np.asarray([[0.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0]])
        ))
        self.assertTrue(np.array_equal(
            loaded_prop.get_action_matrix("propagate_0"),
            np.asarray([[0.0, 0.0, 3.0, 3.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 4.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0]])
        ))
        self.assertTrue(np.array_equal(
            loaded_prop.get_action_matrix("flag_0"),
            np.asarray([[0.0, 1.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 3.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0]])
        ))
        self.assertTrue(np.array_equal(
            loaded_prop.get_action_matrix("flag_0", time_stamp=2),
            np.asarray([[0.0, 1.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0]])
        ))


if __name__ == '__main__':
    unittest.main()
