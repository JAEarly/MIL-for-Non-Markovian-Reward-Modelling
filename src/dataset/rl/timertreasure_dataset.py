from dataset.rl.rl_dataset import RLDataset
from rl_training.maps import maps

import torch

class TimerTreasureDataset(RLDataset):

    @classmethod
    def generate_metadata(cls, bags):
        print('Dataset metadata not found, generating now...')

        # Get env info used for bag metadata
        tre_bounds = maps['timertreasure']['boxes']['treasure']['coords']
        tre_x0, tre_y0, tre_x1, tre_y1 = tre_bounds[0][0], tre_bounds[0][1], tre_bounds[1][0], tre_bounds[1][1]

        bags_metadata = []
        for bag in bags:
            in_treasure = torch.zeros(len(bag))
            for idx, instance in enumerate(bag):
                x, y = instance
                if tre_x0 <= x <= tre_x1 and tre_y0 <= y <= tre_y1:
                    in_treasure[idx] = 1
            bag_metadata = {
                'true_pos_x': bag[:, 0],
                'true_pos_y': bag[:, 1],
                'in_treasure': in_treasure,
            }
            bags_metadata.append(bag_metadata)
        return bags_metadata
