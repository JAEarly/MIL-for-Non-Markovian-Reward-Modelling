import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DISCRETE_ACTION_MAP = np.array([
    [ 0., 0.], # 0: no-op
    [-1., 0.], # 1: left
    [ 1., 0.], # 2: right
    [ 0.,-1.], # 3: down
    [ 0., 1.]  # 4: up
]) 

class HoloNav(gym.Env):
    """
    2D holonomic navigation task with either continuous or discrete actions.
    """
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, map="default", continuous=True, action_noise=(None,), render_mode=False):
        self.continuous, self.action_noise = continuous, action_noise
        self._build_map(map)
        # Set up rendering.
        self.render_mode = render_mode
        if self.render_mode: 
            assert self.render_mode in self.metadata["render.modes"]
            if self.render_mode == "rgb_array": plt.switch_backend("agg")
            self.render_map()
        else: self.ax = None
        self.obs = None
        self.seed()

    def seed(self, seed=None): 
        self.rng = np.random.default_rng(seed)
        self.action_space.np_random.seed(seed)

    def reset(self): 
        ok = False
        while not ok:
            # Choose a box to initialise in and sample a random uniform location inside.
            init_box  = self.rng.choice(list(self.map["boxes"].keys()), p=self._init_box_probs)
            xb, yb = zip(*self.map["boxes"][init_box]["coords"])
            self.obs, ok = [self.rng.uniform(*xb), self.rng.uniform(*yb)], True
            # Allow boxes with init_weight = 0 to block this location.
            if "boxes" in self.map:
                for b in self.map["boxes"].values():
                    if "init_weight" in b and b["init_weight"] == 0 and pt_in_box(self.obs, b["coords"]): ok = False; break
        # Reset box activation to the defaults.
        for n, b in self.map["boxes"].items(): self._set_activation(n, b["default_activation"])
        # Collect activation for trigger targets and add to observation vector.
        for target in self.trigger_targets: self.obs.append(float(self.map["boxes"][target]["active"]))
        self.obs = np.array(self.obs)
        return self.obs.copy()

    def step(self, action):
        assert action in self.action_space, f"Invalid action (space = {self.action_space})"
        # With epsilon action noise, have a nonzero probability of resampling a random action.
        if self.action_noise[0] == "epsilon" and self.action_noise[1] > 0 and self.rng.rand() < self.action_noise[1]: 
            action = self.action_space.sample()
        if not self.continuous: action = DISCRETE_ACTION_MAP[action].copy()
        # With Gaussian action noise, a noise vector is added to the *unscaled* action vector.
        # NOTE: This currently means max_speed can be exceeded.
        if self.action_noise[0] == "gaussian": 
            action += self.rng.normal(scale=self.action_noise[1], size=action.shape)
        action *= self.map["max_speed"] # NOTE: Action scaling applied here.
        xy = self.obs[:2]
        # Update x,y position, clipping within bounds.
        xy_new = np.clip(xy + action, [0,0], self.map["shape"])
        # Calculate reward.
        reward, reward_components, p_continue, intersect_wall = self.reward(xy, xy_new)
        # Update observation.
        self.obs[:2] = xy if intersect_wall else xy_new
        # Terminate according to continuation probability.
        done = self.rng.random() >= p_continue
        # Update activations.
        for i, target in enumerate(self.trigger_targets): self.obs[2+i] = self.map["boxes"][target]["active"]
        return self.obs.copy(), reward, done, reward_components 

    def reward(self, xy, xy_new):
        """
        Reward function.
        NOTE: Reward is based on current and next position.
        """
        reward_components, p_continue = [], 1
        # Get reward from attractors.
        if "point_attractors" in self.map:
            for n, p in self.map["point_attractors"].items():
                reward_components.append(p["reward"] * np.linalg.norm(xy - p["coords"]))
        if "line_attractors" in self.map:
            for n, l in self.map["line_attractors"].items():
                reward_components.append(l["reward"] * pt_to_line_dist(xy, l["coords"]))
        # Get reward and continuation probability from boxes.
        if "boxes" in self.map:
            for n, b in self.map["boxes"].items():
                if pt_in_box(xy, b["coords"]) and b["active"]:
                    if "reward" in b: reward_components.append(b["reward"])
                    if "continuation_prob" in b: p_continue *= b["continuation_prob"]  
                    if "trigger" in b:
                        for target, active in b["trigger"]: self._set_activation(target, active)
                elif "reward" in b: reward_components.append(0.)
        # Get reward from walls.
        intersect_wall = False
        if not np.all(np.isclose(xy, xy_new)) and "walls" in self.map:
            for n, w in self.map["walls"].items():
                if do_intersect(xy, xy_new, w["coords"][0], w["coords"][1]): 
                    if "reward" in w: reward_components[n] = w["reward"] 
                    intersect_wall = True
                elif "reward" in w: reward_components.append(0.)
        return sum(reward_components), {"reward_components": reward_components}, p_continue, intersect_wall

    def render(self, mode="human", pause=1e-6): 
        assert mode == self.render_mode, f"Render mode is {self.render_mode}, so cannot use {mode}"
        if self.obs is not None: self._render_agent()
        if self.render_mode == "human": plt.pause(pause)
        elif self.render_mode == "rgb_array": # From https://stackoverflow.com/a/7821917.
            self.fig.canvas.draw()
            data = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            return data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        
    # ===================================================================

    def _build_map(self, map): 
        self.map = map; self.map_elements = {}        
        # Convert coords to NumPy arrays and prevent illegal name repetition.
        for typ in {"boxes","walls","point_attractors","line_attractors"}:
            if typ in self.map:
                for n, x in self.map[typ].items(): 
                    assert n not in self.map_elements, f"Repeated map element name: {n}"
                    self.map_elements[n] = None # Gets replaced with matplotlib element if self.render_mode != False.
                    x["coords"] = np.array(x["coords"])
                    assert (x["coords"] >= 0).all() and (x["coords"] <= self.map["shape"]).all(), f"Out-of-bounds element: {n}"
                    if typ == "boxes" and "default_activation" not in x: x["default_activation"] = True
        # Check for trigger targets, which add boolean dimensions to the observation space. List them in alphanumeric order.
        self.trigger_targets = sorted(list({t[0] for b in self.map["boxes"].values() if "trigger" in b for t in b["trigger"]}))
        # Define observation and action spaces.
        self.observation_space = gym.spaces.Box(
            low=np.float32([0.,0.]+[0. for _ in self.trigger_targets]), 
            high=np.float32(self.map["shape"]+[1. for _ in self.trigger_targets])
            )
        if self.continuous: self.action_space = gym.spaces.Box(np.float32(-1.), np.float32(1.), shape=(2,)) 
        else: self.action_space = gym.spaces.Discrete(5) 
        # Probability distribution for initialisation box.
        w = np.array([(b["init_weight"] if "init_weight" in b else 0) for b in self.map["boxes"].values()])
        s = w.sum()
        assert s != 0, "Must specify at least one initialisation box."
        self._init_box_probs = w / w.sum()

    def render_map(self):
        try: self.ax
        except: 
            S = 2
            ratio = self.map["shape"][1] / self.map["shape"][0]
            self.fig, self.ax = plt.subplots(figsize=(S, ratio * S) if ratio > 1 else (S / ratio, S))
        self.ax.set_xticks([]); self.ax.set_yticks([]); plt.ion(); self.ax.set_aspect("equal", "box")
        if "boxes" in self.map:
            for n, b in self.map["boxes"].items():
                self.map_elements[n] = Rectangle(
                    xy=[b["coords"][0,0], b["coords"][0,1]],
                    width=b["coords"][1,0] - b["coords"][0,0],
                    height=b["coords"][1,1] - b["coords"][0,1],
                    facecolor=(b["face_colour"] if "face_colour" in b else "none"),
                    edgecolor=(b["edge_colour"] if "edge_colour" in b else None),
                )
                self.ax.add_patch(self.map_elements[n])
        if "walls" in self.map:
            for n, w in self.map["walls"].items():
                self.map_elements[n] = self.ax.plot(*zip(*w["coords"]), c="k")
        if "point_attractors" in self.map:
            for n, p in self.map["point_attractors"].items():
                self.map_elements[n] = self.ax.scatter(*p["coords"], c=p["colour"], zorder=3)
        if "line_attractors" in self.map:
            for n, l in self.map["line_attractors"].items():
                self.map_elements[n] = self.ax.plot(*zip(*l["coords"]), c=l["colour"], ls="--")
        self.ax.set_xlim([0,self.map["shape"][0]])
        self.ax.set_ylim([0,self.map["shape"][1]])

    def _set_activation(self, target, active): 
        try: prev = self.map["boxes"][target]["active"]
        except: prev = None 
        if active == "flip": active = ~prev
        active = bool(active)
        self.map["boxes"][target]["active"] = active
        if active != prev and self.render_mode: 
            self.map_elements[target].set_alpha((1 if active else .25)) 

    def _render_agent(self):
        try: self._scatter_point.remove()
        except: pass
        self._scatter_point = self.ax.scatter(
            *self.obs[:2], 
            s=70, c="k", lw=1, edgecolor="w", zorder=4
            )

# ===================================================================
# SOME GEOMETRY FUNCTIONS

"""First three of these are from https://stackoverflow.com/a/18524383."""

def side(a, b, c):
    """Returns a position of the point c relative to the line going through a and b.
    Points a, b are expected to be different."""
    d = (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
    return 1 if d > 0 else (-1 if d < 0 else 0)

def is_point_in_closed_segment(a, b, c):
    """Returns True if c is inside closed segment, False otherwise. a, b, c are expected to be collinear."""
    if a[0] < b[0]:
        return a[0] <= c[0] and c[0] <= b[0]
    if b[0] < a[0]:
        return b[0] <= c[0] and c[0] <= a[0]
    if a[1] < b[1]:
        return a[1] <= c[1] and c[1] <= b[1]
    if b[1] < a[1]:
        return b[1] <= c[1] and c[1] <= a[1]
    return a[0] == c[0] and a[1] == c[1]

def do_intersect(a, b, c, d):
    """Check if line segments [a, b], [c, d] intersect."""
    s1 = side(a,b,c)
    s2 = side(a,b,d)
    # All points are colinear
    if s1 == 0 and s2 == 0:
        return \
            is_point_in_closed_segment(a, b, c) or is_point_in_closed_segment(a, b, d) or \
            is_point_in_closed_segment(c, d, a) or is_point_in_closed_segment(c, d, b)
    # No touching and on the same side
    if s1 and s1 == s2:
        return False
    s1 = side(c,d,a)
    s2 = side(c,d,b)
    # No touching and on the same side
    if s1 and s1 == s2:
        return False
    return True

def pt_to_line_dist(pt, ln):
    """Distance from a point to a line."""
    return np.linalg.norm(np.cross(ln[1]-ln[0], ln[0]-pt))/np.linalg.norm(ln[1]-ln[0])

def pt_in_box(pt, bx):
    """Check if a point lies inside a box."""
    return np.all(pt - bx[0] >= [0,0]) and np.all(pt - bx[1] <= [0,0]) 