import torch
import torch.nn as nn
import torch.nn.functional as F

class A2CNetwork(nn.Module):
    """
    Simple A2C-style network using the same feature extraction as the original DuelingNetwork.
    Produces policy logits and value predictions instead of Q-values.
    """
    def __init__(self, num_actions, structured_size, action_coord_sizes=None):
        super(A2CNetwork, self).__init__()
        
        # Screen branch for 3-channel RGB input
        self.screen_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4),
        )
        self.screen_pool = nn.AdaptiveAvgPool2d((32, 32))

        # Minimap branch (assuming 64x64 input)
        self.minimap_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(8),
        )
        self.minimap_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc_structured = nn.Sequential(
            nn.Linear(structured_size, 64),
            nn.ReLU(),
        )

        # Compute combined feature size
        with torch.no_grad():
            dummy_screen = torch.zeros(1, 3, 192, 256)         # (channels, height, width)
            dummy_minimap = torch.zeros(1, 3, 128, 128)          # assume minimap remains 128x128
            dummy_struct = torch.zeros(1, structured_size)
            s = self.screen_pool(self.screen_conv(dummy_screen)).view(1, -1)
            m = self.minimap_pool(self.minimap_conv(dummy_minimap)).view(1, -1)
            d = self.fc_structured(dummy_struct)
            combined_input_size = s.size(1) + m.size(1) + d.size(1)

        self.combined_fc = nn.Sequential(
            nn.Linear(combined_input_size, 256),
            nn.ReLU(),
        )

        # Policy (actor) and value heads
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1)

        # Optional coordinate heads (if needed)
        self.action_coord_heads = nn.ModuleDict({})
        if action_coord_sizes:
            for action, output_dim in action_coord_sizes.items():
                self.action_coord_heads[str(action)] = nn.Linear(256, output_dim)

    def forward(self, screen, minimap, structured):
        if screen.dim() == 5:
            screen = screen.squeeze(1)
        if minimap.dim() == 5:
            minimap = minimap.squeeze(1)
        # Ensure RGB
        if screen.size(1) != 3:
            screen = screen.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        if minimap.size(1) != 3:
            minimap = minimap.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        x_screen = self.screen_pool(self.screen_conv(screen)).view(screen.size(0), -1)
        x_minimap = self.minimap_pool(self.minimap_conv(minimap)).view(minimap.size(0), -1)
        x_struct = self.fc_structured(structured)
        combined = torch.cat([x_screen, x_minimap, x_struct], dim=1)
        combined = self.combined_fc(combined)

        # Actor: policy logits
        policy_logits = self.policy_head(combined)
        # Critic: value estimate
        value = self.value_head(combined)

        # Coord heads if needed
        coord_outputs = {}
        for action, head in self.action_coord_heads.items():
            coord_outputs[action] = head(combined)

        return policy_logits, value, coord_outputs
    
def create_a2c_model(num_actions, structured_size=31, action_coord_sizes=None):
    if action_coord_sizes is None:
        action_coord_sizes = {i: 2 for i in range(num_actions)}
    return A2CNetwork(num_actions, structured_size, action_coord_sizes)