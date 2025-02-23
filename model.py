import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingNetwork(nn.Module):
    def __init__(self, num_actions, structured_size, action_coord_sizes):
        super(DuelingNetwork, self).__init__()
        
        # Screen branch for 3-channel RGB input.
        self.screen_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4)  # for 128x128 -> approx. 32x32 output
        )
        self.screen_pool = nn.AdaptiveAvgPool2d((32, 32))
        
        # Minimap branch (assuming 64x64 input).
        self.minimap_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(8)  # for 64x64 -> approx. 8x8 output
        )
        self.minimap_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        self.fc_structured = nn.Sequential(
            nn.Linear(structured_size, 64),
            nn.ReLU(),
        )
        
        # Dynamically compute the combined feature size.
        with torch.no_grad():
            dummy_screen = torch.zeros(1, 3, 128, 128)
            dummy_minimap = torch.zeros(1, 3, 64, 64)
            dummy_structured = torch.zeros(1, structured_size)
            x_screen = self.screen_conv(dummy_screen)
            x_screen = self.screen_pool(x_screen).view(1, -1)
            x_minimap = self.minimap_conv(dummy_minimap)
            x_minimap = self.minimap_pool(x_minimap).view(1, -1)
            x_struct = self.fc_structured(dummy_structured)
            combined_input_size = x_screen.size(1) + x_minimap.size(1) + x_struct.size(1)
        
        self.combined_fc = nn.Sequential(
            nn.Linear(combined_input_size, 256),
            nn.ReLU(),
        )
        
        # Dueling outputs.
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, num_actions)
        
        # Coordinate heads for each action.
        self.action_coord_heads = nn.ModuleDict({
            str(action): nn.Linear(256, output_dim)
            for action, output_dim in action_coord_sizes.items()
        })

    def forward(self, screen, minimap, structured, chosen_action=None):
        # Remove extra singleton dim.
        if screen.dim() == 5:
            screen = screen.squeeze(1)
        if minimap.dim() == 5:
            minimap = minimap.squeeze(1)
            
        # Ensure screen is RGB.
        if screen.size(1) != 3:
            screen = screen.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        # Ensure minimap is RGB.
        if minimap.size(1) != 3:
            minimap = minimap.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        
        # Process screen branch.
        x_screen = self.screen_conv(screen)
        x_screen = self.screen_pool(x_screen)
        x_screen = x_screen.view(x_screen.size(0), -1)

        # Process minimap branch.
        x_minimap = self.minimap_conv(minimap)
        x_minimap = self.minimap_pool(x_minimap)
        x_minimap = x_minimap.view(x_minimap.size(0), -1)

        # Process structured input.
        x_struct = self.fc_structured(structured)
        
        # Combine features.
        combined = torch.cat([x_screen, x_minimap, x_struct], dim=1)
        combined = self.combined_fc(combined)
        
        # Q-value computations.
        val = self.value(combined)
        adv = self.advantage(combined)
        q_values = val + adv - adv.mean(dim=1, keepdim=True)
        
        if chosen_action is not None:
            coord_out = self.action_coord_heads[str(chosen_action)](combined)
            return q_values, {"coord": coord_out}
        else:
            coord_all = {action: head(combined) for action, head in self.action_coord_heads.items()}
            return q_values, {"coord_all": coord_all}

def create_dueling_model(num_actions, structured_size=31, action_coord_sizes=None):
    if action_coord_sizes is None:
        action_coord_sizes = {i: 2 for i in range(num_actions)}
    return DuelingNetwork(num_actions, structured_size, action_coord_sizes)