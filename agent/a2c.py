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
        
        # Update screen_conv to accept the actual number of input channels
        self.screen_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),  # Larger stride reduces computation
            nn.ReLU(inplace=True),  # inplace=True saves memory
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),  # FIXED: Changed from 16 to 32 to match previous conv output
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # FIXED: Changed input from 16 to 32
            nn.ReLU(inplace=True),
            # Let adaptive pooling handle downsampling
        )
        self.screen_pool = nn.AdaptiveAvgPool2d((32, 32))  # Smaller output size
        
        # Similar update for minimap_conv if needed
        self.minimap_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
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

def a2c_train_step(model, optimizer, rollout, gamma=0.99, ent_coef=0.01, vf_coef=0.5, batch_size=64, writer=None):
    """
    Optimized A2C update step using batched processing.
    """
    if len(rollout) < batch_size:
        return 0.0, 0.0, 0.0, 0.0
    
    # Pre-allocate tensors when possible and avoid multiple list iterations
    num_items = len(rollout)
    # Only process once to extract dimensions
    sample_item = rollout[0]
    screens_shape = sample_item[0].shape
    minimaps_shape = sample_item[1].shape
    structured_shape = sample_item[2].shape
    
    # Handle device consistently
    device = sample_item[0].device
    
    # Pre-allocate tensors when size is known
    try:
        screens = torch.cat([item[0] for item in rollout])
        minimaps = torch.cat([item[1] for item in rollout])
        structureds = torch.cat([item[2] for item in rollout])
        actions = torch.tensor([item[3] for item in rollout], dtype=torch.long, device=device)
        rewards = torch.tensor([float(item[4]) for item in rollout], dtype=torch.float32, device=device)
        dones = torch.tensor([float(item[5]) if len(item) > 5 else 0.0 for item in rollout], 
                              dtype=torch.float32, device=device)
    except Exception as e:
        print(f"Error in tensor preparation: {e}")
        return 0.0, 0.0, 0.0, 0.0
    
    # Process in larger batches when possible
    total_loss = 0.0
    policy_loss_total = 0.0
    value_loss_total = 0.0
    entropy_total = 0.0
    
    num_batches = 0
    batch_indices = torch.randperm(len(screens))  # Shuffle for better training
    
    for i in range(0, len(screens), batch_size):
        batch_idx = batch_indices[i:i+batch_size]
        # Use indexing to avoid unnecessary copies when possible
        batch_screens = screens.index_select(0, batch_idx)
        batch_minimaps = minimaps.index_select(0, batch_idx) 
        batch_structureds = structureds.index_select(0, batch_idx)
        batch_actions = actions.index_select(0, batch_idx)
        batch_rewards = rewards.index_select(0, batch_idx)
        batch_dones = dones.index_select(0, batch_idx)
        
        # Rest of processing remains similar...

        try:
            # Get all values first
            policy_logits, values, _ = model(batch_screens, batch_minimaps, batch_structureds)
            values = values.squeeze()
            
            if values.dim() == 0:  # Handle scalar tensor
                values = values.unsqueeze(0)
            
            with torch.no_grad():
                _, last_value, _ = model(batch_screens[-1:], batch_minimaps[-1:], batch_structureds[-1:])
                last_value = last_value.squeeze()
            
            # Compute advantages for the batch
            advantages = []
            gae = 0
            for t in reversed(range(len(batch_rewards))):
                next_value = last_value if t == len(batch_rewards) - 1 else values[t]
                delta = batch_rewards[t] + gamma * next_value * (1 - batch_dones[t]) - values[t]
                gae = delta + gamma * gae * (1 - batch_dones[t])
                advantages.insert(0, gae)
            advantages = torch.stack(advantages)
            
            # Compute losses
            log_probs = F.log_softmax(policy_logits, dim=1)
            chosen_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze()
            
            policy_loss = -(chosen_log_probs * advantages.detach()).mean()
            value_loss = F.mse_loss(values, advantages + values.detach())
            entropy = -(log_probs * torch.exp(log_probs)).sum(dim=1).mean()
            
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item() if loss is not None else 0.0
            policy_loss_total += policy_loss.item() if policy_loss is not None else 0.0
            value_loss_total += value_loss.item() if value_loss is not None else 0.0
            entropy_total += entropy.item() if entropy is not None else 0.0
            num_batches += 1
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            continue
    
    if num_batches == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    if writer:
        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Total", total_loss / num_batches, num_batches)
        writer.add_scalar("Loss/Policy", policy_loss_total / num_batches, num_batches)
        writer.add_scalar("Loss/Value", value_loss_total / num_batches, num_batches)
        writer.add_scalar("Loss/Entropy", entropy_total / num_batches, num_batches)
    
    return (
        total_loss / num_batches,
        policy_loss_total / num_batches,
        value_loss_total / num_batches,
        entropy_total / num_batches
    )
    
def create_a2c_model(num_actions, structured_size=31, action_coord_sizes=None):
    if action_coord_sizes is None:
        action_coord_sizes = {i: 2 for i in range(num_actions)}
    model = A2CNetwork(num_actions=num_actions, 
                    structured_size=structured_size)
    # Make the forward pass faster with TorchScript
    scripted_model = torch.jit.script(model)
    return scripted_model