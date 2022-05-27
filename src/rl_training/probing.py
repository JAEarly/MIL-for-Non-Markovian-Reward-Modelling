from numpy import array


probe_actions = {
    "key_treasure": {
        "Optimal Left": [(0.15, 0.45), ("D", 2), ("R", 3), ("U", 5), ("N", 89)],
        "Optimal Right": [(0.85, 0.45), ("D", 2), ("L", 3), ("U", 5), ("N", 89)],
        "Sub Optimal": [(0.15, 0.55), ("U", 3), ("R", 7), ("D", 7), ("L", 3), ("U", 6), ("N", 73)],
        "Challenging": [(0.15, 0.45), ("D", 4), ("R", 2), ("U", 3), ("R", 3), ("D", 3), ("R", 2), ("U", 7), ("L", 3),
                        ("N", 72)],
        "Failure Case": [(0.85, 0.45), ("L", 3), ("U", 3), ("N", 93)],
    },
    "charger_treasure": {
        "Optimal": [(0.85, 0.45), ("D", 2), ("L", 3), ("N", 42), ("U", 5), ("N", 47)],
        "Under charged": [(0.85, 0.45), ("D", 2), ("L", 3), ("N", 20), ("U", 5), ("N", 69)],
        "Over charged": [(0.85, 0.45), ("D", 2), ("L", 3), ("N", 60), ("U", 5), ("N", 29)],
        "Challenging": [(0.15, 0.55), ("L", 1), ("D", 3), ("N", 20), ("R", 1), ("U", 1), ("R", 1), ("D", 1), ("R", 1),
                        ("U", 1), ("R", 1), ("D", 1), ("R", 1), ("U", 1), ("R", 1), ("D", 1), ("R", 1), ("U", 1),
                        ("R", 1), ("D", 1), ("R", 1), ("U", 5), ("L", 4), ("N", 49)],
    },
    "moving_treasure": {
        "Optimal Left": [(0.15, 0.55), ("U", 1), ("R", 3, 0.75), ("U", 1), ("L", 15, 0.2), ("U", 2, 0.5),
                         ("R", 41, 0.2), ("D", 1, 0.5), ("L", 35, 0.2)],
        "Optimal Right": [(0.85, 0.55), ("U", 1), ("L", 5), ("U", 1), ("L", 13, 0.2), ("U", 2, 0.5),
                          ("R", 41, 0.2), ("D", 1, 0.5), ("L", 35, 0.2)],
        "Static": [(0.15, 0.55), ("U", 1), ("R", 2), ("U", 1), ("N", 95)],
        "Challenging": [(0.85, 0.45), ("R", 1), ("D", 4), ("L", 9), ("U", 9),
                        ("R", 1), ("D", 3), ("R", 1), ("U", 3), ("R", 1), ("D", 3), ("R", 1), ("U", 3),
                        ("R", 1), ("D", 3), ("R", 1), ("U", 3), ("R", 1), ("D", 3), ("R", 1), ("U", 3),
                        ("R", 1), ("D", 4), ("L", 6), ("N", 33)],
    },
    "timer_treasure": {
        "Optimal": [(0.15, 0.45), ("R", 3), ("N", 44), ("U", 3), ("N", 49)],
        "Too Late": [(0.85, 0.45), ("L", 3), ("N", 53), ("U", 3), ("N", 40)],
        "Too Early": [(0.15, 0.45), ("R", 3), ("N", 28), ("U", 3), ("N", 65)],
        "Challenging": [(0.85, 0.45), ("L", 3), ("N", 44), ("U", 3), ("R", 1), ("U", 1),
                        ("L", 1), ("U", 1), ("L", 1), ("D", 1), ("L", 1), ("D", 1), ("R", 1), ("N", 40)],
    },
}

V = 0.1
action_map = {
    "U": array([0, +V]),
    "D": array([0, -V]),
    "L": array([-V,  0]),
    "R": array([+V,  0]),
    "N": array([0,  0])
}


def probe(oracle_name, probe_name):
    actions = probe_actions[oracle_name][probe_name]
    xy = [array(actions[0])]
    for action in actions[1:]:
        if len(action) == 2:
            direction, num = action
            mul = 1
        else:
            direction, num, mul = action
            assert mul <= 1
        for _ in range(num):
            xy.append(xy[-1] + action_map[direction] * mul)
    return array(xy)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    for tr in [probe("key_treasure", 0), probe("key_treasure", 1)]:
        print(len(tr))
        ax.plot(*tr.T, lw=5, alpha=0.5)

    plt.show()
