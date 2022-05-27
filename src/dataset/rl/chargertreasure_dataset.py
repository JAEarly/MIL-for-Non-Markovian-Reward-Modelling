from dataset.rl.rl_dataset import RLDataset
from rl_training.maps import maps
import torch


class ChargerTreasureDataset(RLDataset):

    @classmethod
    def generate_metadata(cls, bags):
        bags_metadata = []

        # Get env info used for bag metadata
        chg_bounds = maps['chargertreasure']['boxes']['charge_zone']['coords']
        tre_bounds = maps['chargertreasure']['boxes']['treasure']['coords']
        chg_x0, chg_y0, chg_x1, chg_y1 = chg_bounds[0][0], chg_bounds[0][1], chg_bounds[1][0], chg_bounds[1][1]
        tre_x0, tre_y0, tre_x1, tre_y1 = tre_bounds[0][0], tre_bounds[0][1], tre_bounds[1][0], tre_bounds[1][1]

        for bag in bags:
            # Generate bag metadata
            charge = torch.zeros(len(bag))
            in_charge = torch.zeros(len(bag))
            in_treasure = torch.zeros(len(bag))
            cur_charge = 0
            for idx, instance in enumerate(bag):
                x, y = instance
                if chg_x0 <= x <= chg_x1 and chg_y0 <= y <= chg_y1:
                    in_charge[idx] = 1
                    cur_charge += 0.02
                if tre_x0 <= x <= tre_x1 and tre_y0 <= y <= tre_y1:
                    in_treasure[idx] = 1
                charge[idx] = cur_charge

            bag_metadata = {
                'true_pos_x': bag[:, 0],
                'true_pos_y': bag[:, 1],
                'charge': charge,
                'in_charge': in_charge,
                'in_treasure': in_treasure,
            }
            bags_metadata.append(bag_metadata)
        return bags_metadata
