import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# --- HELPER: CNN Module ---
class CNNBase(nn.Module):
    """
    Simple CNN to extract features from CarRacing (96x96x3) images.
    Output dimension depends on the architecture.
    """
    def __init__(self):
        super(CNNBase, self).__init__()
        # Input: (3, 96, 96)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)  # -> (32, 23, 23)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # -> (64, 10, 10)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) # -> (64, 8, 8)
        self.fc = nn.Linear(64 * 8 * 8, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) # Flatten
        x = F.relu(self.fc(x))
        return x

class Policy(nn.Module):
    continuous = True 

    def __init__(self, action_dim=3, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Policy, self).__init__()
        self.device = device
        self.action_dim = action_dim

        # 1. ARCHITECTURE
        # Actor Network (Policy)
        self.actor_cnn = CNNBase()
        self.actor_mean = nn.Linear(512, action_dim)
        # Learnable log_std (State-independent for stability)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic Network (Value)
        self.critic_cnn = CNNBase()
        self.critic_value = nn.Linear(512, 1)

        self.to(device)

    def forward(self, x):
        """
        Forward pass for the Actor. Returns Mean and Std.
        x: Input image tensor (Batch, C, H, W)
        """
        # Normalize pixel values if not already done
        if x.max() > 1.0: 
            x = x / 255.0
            
        features = self.actor_cnn(x)
        mu = self.actor_mean(features)
        
        # Exponentiate log_std to get positive std
        # Expand log_std to match batch size
        std = torch.exp(self.actor_log_std).expand_as(mu)
        
        return mu, std

    def get_value(self, x):
        """Forward pass for the Critic."""
        if x.max() > 1.0: x = x / 255.0
        features = self.critic_cnn(x)
        return self.critic_value(features)

    def act(self, state):
        """
        Selects an action for the environment during EVALUATION (Inference only).
        Compatible with main.py which expects just the action.
        """
        # Preprocess: (H, W, C) -> (1, C, H, W) + Tensor conversion
        state_tensor = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mu, std = self.forward(state_tensor)
            dist = Normal(mu, std)
            action = dist.sample()
            
            # Post-processing for CarRacing:
            # Action 0: Steering [-1, 1]
            # Action 1: Gas [0, 1]
            # Action 2: Brake [0, 1]
            # The network outputs unbounded Gaussians, so we clip for the env.
            action_np = action.cpu().numpy()[0]
            action_np[0] = np.clip(action_np[0], -1, 1)
            action_np[1] = np.clip(action_np[1], 0, 1)
            action_np[2] = np.clip(action_np[2], 0, 1)

        return action_np

    def _select_action_train(self, state_tensor):
        """
        Internal helper: Returns action, log_prob, and value for TRAINING.
        """
        with torch.no_grad():
            mu, std = self.forward(state_tensor)
            value = self.get_value(state_tensor)
            
            dist = Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Return scalars/flat arrays
        # Use .item() for value to ensure it's a simple float, avoiding shape (1,) issues later
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.item()

    # --- TRPO MATHEMATICAL KERNELS ---

    def fisher_vector_product(self, states, p, damping=0.01):
        """
        Computes F * p efficiently using the double backprop trick.
        states: batch of observations
        p: vector to multiply with Fisher Matrix
        """
        p.detach()
        mu, std = self.forward(states)
        dist = Normal(mu, std)
        
        # KL Divergence against ITSELF (fixed detachment)
        # We use a trick: KL(pi_old || pi_new) where pi_old is detached pi_new
        mu_old = mu.detach()
        std_old = std.detach()
        dist_old = Normal(mu_old, std_old)
        
        # Analytic KL for Gaussians
        kl = torch.distributions.kl_divergence(dist_old, dist).mean()
        
        # First derivative: Gradient of KL w.r.t parameters
        kl_grad = torch.autograd.grad(kl, self.actor_parameters(), create_graph=True)
        kl_grad_flat = torch.cat([grad.reshape(-1) for grad in kl_grad])
        
        # Dot product: (grad_KL * p)
        kl_grad_p = (kl_grad_flat * p).sum()
        
        # Second derivative: Gradient of (grad_KL * p) w.r.t parameters -> F * p
        kl_hessian_p = torch.autograd.grad(kl_grad_p, self.actor_parameters())
        kl_hessian_p_flat = torch.cat([grad.reshape(-1) for grad in kl_hessian_p])
        
        # Add damping to ensure positive definiteness (Standard TRPO trick)
        return kl_hessian_p_flat + damping * p

    def conjugate_gradient(self, states, b, n_steps=10, residual_tol=1e-10, damping=0.01):
        """
        Solves Fx = b for x using Conjugate Gradient.
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)

        for _ in range(n_steps):
            # Compute F * p
            Ap = self.fisher_vector_product(states, p, damping=damping)
            alpha = rdotr / torch.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def actor_parameters(self):
        """Helper to get only actor params (exclude critic)"""
        return list(self.actor_cnn.parameters()) + \
               list(self.actor_mean.parameters()) + \
               [self.actor_log_std]

    # --- MAIN TRAINING LOOP AND UPDATES ---

    def _trpo_step(self, batch, max_kl=0.01, damping=0.001):
        """
        Performs ONE TRPO update using the provided batch of data.
        """
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        # Ensure advantages and returns are 1D [N] and not [N, 1] to avoid broadcasting issues
        advantages = batch['advantages'].to(self.device).reshape(-1)
        returns = batch['returns'].to(self.device).reshape(-1)
        old_log_probs = batch['old_log_probs'].to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 1. Compute Policy Gradient (L_surrogate)
        mu, std = self.forward(states)
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        ratio = torch.exp(log_probs - old_log_probs)
        # Note: ratio is [N], advantages is [N]. Element-wise product.
        surr_loss = (ratio * advantages).mean()

        # Get gradient (g)
        grads = torch.autograd.grad(surr_loss, self.actor_parameters())
        g = torch.cat([grad.reshape(-1) for grad in grads]).detach()

        # 2. Use Conjugate Gradient to find search direction (x = F^-1 g)
        x = self.conjugate_gradient(states, g, damping=damping)

        # 3. Compute Max Step Size (beta)
        Fx = self.fisher_vector_product(states, x, damping=damping)
        xFx = torch.dot(x, Fx)
        beta = torch.sqrt(2 * max_kl / (xFx + 1e-8))
        full_step = beta * x

        # 4. Line Search
        old_params = parameters_to_vector(self.actor_parameters())
        success = False
        step_frac = 1.0
        
        for j in range(10): 
            new_step = step_frac * full_step
            new_params = old_params + new_step
            vector_to_parameters(new_params, self.actor_parameters())
            
            with torch.no_grad():
                mu_new, std_new = self.forward(states)
                dist_new = Normal(mu_new, std_new)
                log_probs_new = dist_new.log_prob(actions).sum(dim=-1)
                
                kl = (old_log_probs - log_probs_new).mean()
                ratio_new = torch.exp(log_probs_new - old_log_probs)
                surr_loss_new = (ratio_new * advantages).mean()
                
            if kl <= max_kl * 1.5 and surr_loss_new > surr_loss:
                success = True
                break
            step_frac *= 0.5
        
        if not success:
            vector_to_parameters(old_params, self.actor_parameters())
            # print("Line search failed.")

        # 5. Update Critic
        optimizer_critic = torch.optim.Adam(self.critic_cnn.parameters(), lr=1e-3)
        optimizer_critic.add_param_group({'params': self.critic_value.parameters()})
        
        for _ in range(5): 
            # squeeze() makes it [N], returns is [N]. Matching shapes.
            values_pred = self.get_value(states).squeeze()
            value_loss = F.mse_loss(values_pred, returns)
            optimizer_critic.zero_grad()
            value_loss.backward()
            optimizer_critic.step()

    def train(self, render=False):
        """
        The Main Loop called by main.py.
        Handles Environment interaction -> GAE -> Update.
        """
        # Hyperparameters
        total_timesteps = 2000000 # Adjust as needed
        timesteps_per_batch = 4096  # Better stability
        gamma = 0.99        # Crucial for long-term planning
        lam = 0.95      # GAE lambda: controls the "Bias-Variance Trade-off" when calculating rewards.
        damping = 0.01  # Standard speed
        max_kl = 0.01   # Allows curvature to work

        if render:
            env = gym.make('CarRacing-v2', continuous=self.continuous, render_mode='human')
        else:
            env = gym.make('CarRacing-v2', continuous=self.continuous)

        state, _ = env.reset()
        
        # Loop over batches
        num_updates = total_timesteps // timesteps_per_batch
        
        print(f"Starting Training: {num_updates} updates expected.")
        
        for update in range(num_updates):
            # 1. Collect Data
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_values = []
            batch_log_probs = []
            batch_dones = []
            
            for t in range(timesteps_per_batch):
                # Prepare state
                state_tensor = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                # Get action from policy
                action, log_prob, value = self._select_action_train(state_tensor)
                
                # Step env (Clip action for environment safety, but store raw action for math)
                clipped_action = action.copy()
                clipped_action[0] = np.clip(clipped_action[0], -1, 1) # Steer
                clipped_action[1] = np.clip(clipped_action[1], 0, 1)  # Gas
                clipped_action[2] = np.clip(clipped_action[2], 0, 1)  # Brake
                
                next_state, reward, terminated, truncated, _ = env.step(clipped_action)
                done = terminated or truncated
                
                # Store
                batch_states.append(state)
                batch_actions.append(action)
                batch_rewards.append(reward)
                batch_values.append(value)
                batch_log_probs.append(log_prob)
                batch_dones.append(done)
                
                state = next_state
                if done:
                    state, _ = env.reset()
            
            # 2. Compute GAE and Returns
            # Bootstrap value of next state if not done
            next_state_tensor = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.get_value(next_state_tensor).cpu().item()
            
            batch_advantages = []
            batch_returns = []
            last_gae_lam = 0
            
            for t in reversed(range(timesteps_per_batch)):
                if t == timesteps_per_batch - 1:
                    next_non_terminal = 1.0 - batch_dones[t] # rough approx if batch cuts mid-episode
                    next_val = next_value
                else:
                    next_non_terminal = 1.0 - batch_dones[t]
                    next_val = batch_values[t+1]
                
                delta = batch_rewards[t] + gamma * next_val * next_non_terminal - batch_values[t]
                last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
                batch_advantages.insert(0, last_gae_lam)
                
                # Return = Advantage + Value
                batch_returns.insert(0, last_gae_lam + batch_values[t])
            
            # 3. Package data for TRPO step
            batch = {
                'states': torch.from_numpy(np.array(batch_states)).float().permute(0, 3, 1, 2), # (B, C, H, W)
                'actions': torch.tensor(np.array(batch_actions)).float(),
                'returns': torch.tensor(np.array(batch_returns)).float(),
                'advantages': torch.tensor(np.array(batch_advantages)).float(),
                'old_log_probs': torch.tensor(np.array(batch_log_probs)).float()
            }
            
            # 4. Perform Update
            self._trpo_step(batch, max_kl=max_kl, damping=damping)
            
            if (update + 1) % 1 == 0:
                print(f"Update {update+1}/{num_updates} complete. Mean Reward: {np.mean(batch_rewards) * timesteps_per_batch / max(1, sum(batch_dones)):.2f}")
                
        env.close()

    def save(self, filename='model.pt'):
        torch.save(self.state_dict(), filename)

    def load(self, filename='model.pt'):
        self.load_state_dict(torch.load(filename, map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret