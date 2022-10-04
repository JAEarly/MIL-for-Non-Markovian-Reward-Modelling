import torch

from dataset.rl.rl_dataset import RLDataset
from rl_training.maps import maps


class MovingTreasureDataset(RLDataset):

    @classmethod
    def generate_metadata(cls, bags):
        # Generate treasure position and direction here as it's consistent between bags
        treasure_width = 0.2
        treasure_speed = 0.02
        treasure_min_x = torch.zeros(100)
        treasure_movement = torch.zeros(100)
        x = 0.5 - (treasure_width / 2)  # Current min x position of treasure
        m = - treasure_speed  # Current movement of treasure
        for idx in range(100):
            treasure_min_x[idx] = x
            treasure_movement[idx] = m
            x += m
            if not (1e-4 < x < (1 - (treasure_width + 1e-4))):
                m *= -1

        # Get treasure y positions
        tre_bounds = maps['movingtreasure']['boxes']['treasure']['coords']
        tre_y0, tre_y1 = tre_bounds[0][1], tre_bounds[1][1]

        # Gather the rest of the metadata
        bags_metadata = []
        for bag in bags:
            assert len(treasure_min_x) == len(treasure_movement) == len(bag)

            # Generate bag metadata
            in_treasure = torch.zeros(len(bag))
            for idx, instance in enumerate(bag):
                x, y = instance
                if treasure_min_x[idx] <= x <= treasure_min_x[idx] + treasure_width and tre_y0 <= y <= tre_y1:
                    in_treasure[idx] = 1

            bag_metadata = {
                'true_pos_x': bag[:, 0],
                'true_pos_y': bag[:, 1],
                'treasure_min_x': treasure_min_x,
                'treasure_movement': treasure_movement,
                'in_treasure': in_treasure,
            }
            bags_metadata.append(bag_metadata)
        return bags_metadata
