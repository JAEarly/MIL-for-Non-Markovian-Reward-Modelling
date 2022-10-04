import torch

from dataset.rl.rl_dataset import RLDataset
from rl_training.maps import maps


class KeyTreasureDataset(RLDataset):

    @classmethod
    def generate_metadata(cls, bags):
        print('Dataset metadata not found, generating now...')

        # Get env info used for bag metadata
        key_bounds = maps['keytreasure_A']['boxes']['key']['coords']
        tre_bounds = maps['keytreasure_A']['boxes']['treasure']['coords']
        key_x0, key_y0, key_x1, key_y1 = key_bounds[0][0], key_bounds[0][1], key_bounds[1][0], key_bounds[1][1]
        tre_x0, tre_y0, tre_x1, tre_y1 = tre_bounds[0][0], tre_bounds[0][1], tre_bounds[1][0], tre_bounds[1][1]

        # Create metadata for each bag
        bags_metadata = []
        for bag in bags:
            # Generate bag metadata
            has_key = torch.zeros(len(bag))
            in_treasure = torch.zeros(len(bag))
            found_key = False
            for idx, instance in enumerate(bag):
                x, y = instance
                if not found_key and key_x0 <= x <= key_x1 and key_y0 <= y <= key_y1:
                    found_key = True
                    has_key[idx:] = 1
                if tre_x0 <= x <= tre_x1 and tre_y0 <= y <= tre_y1:
                    in_treasure[idx] = 1
            bag_metadata = {
                'true_pos_x': bag[:, 0],
                'true_pos_y': bag[:, 1],
                'has_key': has_key,
                'in_treasure': in_treasure,
            }
            bags_metadata.append(bag_metadata)
        return bags_metadata
