maps = {
    "keytreasure_A": {
        "shape": [1,1], "max_speed": .1,
        "boxes": {
            "land":        {"coords": [[0,0],[1,1]], "face_colour": "darkolivegreen"},
            "spawn_left":  {"coords": [[0.1,0.4],[0.2,0.6]], "face_colour": "purple", "edge_colour": "k", "init_weight": 0.5},
            "spawn_right": {"coords": [[0.8,0.4],[0.9,0.6]], "face_colour": "purple", "edge_colour": "k", "init_weight": 0.5},
            "key":         {"coords": [[0.4,0.1],[0.6,0.3]], "face_colour": "darkgoldenrod", "edge_colour": "k"},
            "treasure":    {"coords": [[0.4,0.7],[0.6,0.9]], "face_colour": "olivedrab", "edge_colour": "k", "default_activation": 0},
        }
    },
    "keytreasure_B": {
        "shape": [1,1], "max_speed": .1,
        "boxes": {
            "land":        {"coords": [[0,0],[1,1]], "face_colour": "darkolivegreen"},
            "spawn_left":  {"coords": [[0.05,0.05],[0.15,0.95]], "face_colour": "purple", "edge_colour": "k", "init_weight": 0.5},
            "spawn_right": {"coords": [[0.85,0.05],[0.95,0.95]], "face_colour": "purple", "edge_colour": "k", "init_weight": 0.5},
            "key":         {"coords": [[0.4,0.1],[0.6,0.3]], "face_colour": "darkgoldenrod", "edge_colour": "k"},
            "treasure":    {"coords": [[0.4,0.7],[0.6,0.9]], "face_colour": "olivedrab", "edge_colour": "k", "default_activation": 0},
        }
    },
    "timertreasure": {
        "shape": [1,1], "max_speed": .1,
        "boxes": {
            "land":        {"coords": [[0,0],[1,1]], "face_colour": "darkolivegreen"},
            "spawn_left":  {"coords": [[0.1,0.4],[0.2,0.6]], "face_colour": "purple", "edge_colour": "k", "init_weight": 0.5},
            "spawn_right": {"coords": [[0.8,0.4],[0.9,0.6]], "face_colour": "purple", "edge_colour": "k", "init_weight": 0.5},
            "treasure":    {"coords": [[0.4,0.7],[0.6,0.9]], "face_colour": "olivedrab", "edge_colour": "k", "default_activation": 0},
        }
    },
    "movingtreasure": {
        "shape": [1,1], "max_speed": .1,
        "boxes": {
            "land":        {"coords": [[0,0],[1,1]], "face_colour": "darkolivegreen"},
            "spawn_left":  {"coords": [[0.1,0.4],[0.2,0.6]], "face_colour": "purple", "edge_colour": "k", "init_weight": 0.5},
            "spawn_right": {"coords": [[0.8,0.4],[0.9,0.6]], "face_colour": "purple", "edge_colour": "k", "init_weight": 0.5},
            "treasure":    {"coords": [[0.4,0.7],[0.6,0.9]], "face_colour": "olivedrab", "edge_colour": "k"},
        }
    },
    "chargertreasure": {
        "shape": [1,1], "max_speed": .1,
        "boxes": {
            "land":        {"coords": [[0,0],[1,1]], "face_colour": "darkolivegreen"},
            "spawn_left":  {"coords": [[0.1,0.4],[0.2,0.6]], "face_colour": "purple", "edge_colour": "k", "init_weight": 0.5},
            "spawn_right": {"coords": [[0.8,0.4],[0.9,0.6]], "face_colour": "purple", "edge_colour": "k", "init_weight": 0.5},
            "charge_zone": {"coords": [[0.0,0.0],[1.0,0.3]], "face_colour": "lightgray", "edge_colour": "k"},
            "charge_bar":  {"coords": [[0.0,0.0],[0.0,0.3]], "face_colour": "gray"},
            "treasure":    {"coords": [[0.4,0.7],[0.6,0.9]], "face_colour": "olivedrab", "edge_colour": "k"},
        }
    },
}
