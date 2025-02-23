import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.q_loss = nn.MSELoss()
        self.coord_loss = nn.MSELoss(reduction='none')

    def forward(self, q_pred, q_target, coord_pred, coord_target, reward_weights):
        q_loss_value = self.q_loss(q_pred, q_target)
        coord_loss_raw = self.coord_loss(coord_pred, coord_target)
        coord_loss_sample = coord_loss_raw.mean(dim=1)
        coord_loss_value = (reward_weights * coord_loss_sample).mean()
        
        # Increase the weight of the coordinate loss
        coord_loss_weight = 50.0  # Increased weight
        total_loss = q_loss_value + coord_loss_weight * coord_loss_value

        # Logging for debugging
        # print(f"Q Loss: {q_loss_value.item()}, Coord Loss: {coord_loss_value.item()}")
        # print(f"Coord Pred: {coord_pred}, Coord Target: {coord_target}")
        # print(f"Reward Weights: {reward_weights}")

        return total_loss, q_loss_value, coord_loss_value