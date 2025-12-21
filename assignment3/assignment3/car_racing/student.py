import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# ==============================================================================
# 1. MODEL ARCHITECTURE
# ==============================================================================

class CNNBase(nn.Module):
    """
    Feature Extractor for the CarRacing environment.
    Input:  (Batch, 3, 96, 96) RGB Images
    Output: (Batch, 512) Feature Vector
    """
    def __init__(self):
        super(CNNBase, self).__init__()
        # Standard 3-layer CNN often used in DQN/PPO papers for Atari/CarRacing
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)  # Reduces dim to 23x23
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # Reduces dim to 10x10
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) # Reduces dim to 8x8
        self.fc = nn.Linear(64 * 8 * 8, 512) # Flatten and map to hidden features

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) # Flatten
        x = F.relu(self.fc(x))
        return x

class Policy(nn.Module):
    """
    TRPO Policy Network.
    Contains both Actor (Policy) and Critic (Value) networks.
    """
    continuous = True 

    def __init__(self, action_dim=3, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Policy, self).__init__()
        self.device = device
        self.action_dim = action_dim

        # --- ACTOR (The Driver) ---
        self.actor_cnn = CNNBase()
        self.actor_mean = nn.Linear(512, action_dim)
        # Learnable log_std. 
        # Note: Independent of state (Parameter, not Linear layer). 
        # This simplifies the Fisher Matrix calculation as covariance is diagonal.
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # --- CRITIC (The Coach) ---
        self.critic_cnn = CNNBase()
        self.critic_value = nn.Linear(512, 1)

        self.to(device)

    def forward(self, x):
        """
        Actor Forward Pass.
        Returns: Mean (mu) and Standard Deviation (std) for the Gaussian distribution.
        """
        # Pixel normalization (0-255 -> 0.0-1.0)
        if x.max() > 1.0: 
            x = x / 255.0
            
        features = self.actor_cnn(x)
        mu = self.actor_mean(features)

        # --- IMPORTANT: ACTION BOUNDING ---
        # Tanh squashes the output to [-1, 1].
        # This is crucial for CarRacing where steering is [-1, 1].
        # It also prevents the "exploding action" problem in early training.
        mu = torch.tanh(mu) 
        
        # Exponentiate log_std to get positive standard deviation
        std = torch.exp(self.actor_log_std).expand_as(mu)
        
        return mu, std

    def get_value(self, x):
        """Critic Forward Pass: Estimates V(s)"""
        if x.max() > 1.0: x = x / 255.0
        features = self.critic_cnn(x)
        return self.critic_value(features)

    def act(self, state):
        """Inference only (for testing/rendering)"""
        state_tensor = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, std = self.forward(state_tensor)
            dist = Normal(mu, std)
            action = dist.sample()
            
            # Clip for environment (Steering -1,1 | Gas 0,1 | Brake 0,1)
            action_np = action.cpu().numpy()[0]
            action_np[0] = np.clip(action_np[0], -1, 1)
            action_np[1] = np.clip(action_np[1], 0, 1)
            action_np[2] = np.clip(action_np[2], 0, 1)
        return action_np

    def _select_action_train(self, state_tensor):
        """Training step: Returns extra info (log_prob, value) needed for buffers."""
        with torch.no_grad():
            mu, std = self.forward(state_tensor)
            value = self.get_value(state_tensor)
            dist = Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.item()

    # ==============================================================================
    # 2. TRPO CORE MATH (Conjugate Gradient & Fisher Product)
    # ==============================================================================

    def fisher_vector_product(self, states, p, damping=0.01):
        """
        COMPUTES F*p WITHOUT STORING THE MATRIX F.
        
        Theory: F = Cov(grad_log_pi). Storing F is impossible for large networks.
        Trick: F*p = grad( (grad(KL) * p) )
        
        1. Compute KL(pi_old || pi_new) (symbolically)
        2. Take gradient of KL w.r.t params -> kl_grad
        3. Dot product (kl_grad * p) -> scalar
        4. Gradient of scalar w.r.t params -> F*p
        """
        p.detach()
        mu, std = self.forward(states)
        dist = Normal(mu, std)
        
        # We compute KL against a fixed version of itself
        mu_old = mu.detach()
        std_old = std.detach()
        dist_old = Normal(mu_old, std_old)
        
        kl = torch.distributions.kl_divergence(dist_old, dist).mean()
        
        # 1st Derivative
        kl_grad = torch.autograd.grad(kl, self.actor_parameters(), create_graph=True)
        kl_grad_flat = torch.cat([grad.reshape(-1) for grad in kl_grad])
        
        # Dot Product
        kl_grad_p = (kl_grad_flat * p).sum()
        
        # 2nd Derivative
        kl_hessian_p = torch.autograd.grad(kl_grad_p, self.actor_parameters())
        kl_hessian_p_flat = torch.cat([grad.reshape(-1) for grad in kl_hessian_p])
        
        return kl_hessian_p_flat + damping * p

    def conjugate_gradient(self, states, b, n_steps=10, residual_tol=1e-10, damping=0.01):
        """
        Solves Fx = b for x using Conjugate Gradient.
        Used to find the search direction (Natural Gradient) inside the Trust Region.
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)

        for _ in range(n_steps):
            # The expensive part: calculating F*p
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
        """Utility to get actor parameters only."""
        return list(self.actor_cnn.parameters()) + \
               list(self.actor_mean.parameters()) + \
               [self.actor_log_std]

    # ==============================================================================
    # 3. TRPO UPDATE STEP
    # ==============================================================================

    def _trpo_step(self, batch, max_kl=0.01, damping=0.001):
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        advantages = batch['advantages'].to(self.device).reshape(-1)
        returns = batch['returns'].to(self.device).reshape(-1)
        old_log_probs = batch['old_log_probs'].to(self.device)

        # Standardize advantages (Critical for convergence stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 1. COMPUTE SURROGATE LOSS GRADIENT (g)
        mu, std = self.forward(states)
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        ratio = torch.exp(log_probs - old_log_probs)
        surr_loss = (ratio * advantages).mean()

        grads = torch.autograd.grad(surr_loss, self.actor_parameters())
        g = torch.cat([grad.reshape(-1) for grad in grads]).detach()

        # 2. CONJUGATE GRADIENT: Solve Fx = g for x
        x = self.conjugate_gradient(states, g, damping=damping)

        # 3. COMPUTE STEP SIZE (Beta)
        # We scale x so that the KL divergence is exactly max_kl
        Fx = self.fisher_vector_product(states, x, damping=damping)
        xFx = torch.dot(x, Fx)
        beta = torch.sqrt(2 * max_kl / (xFx + 1e-8))
        full_step = beta * x

        # 4. LINE SEARCH (Backtracking)
        # We ensure the update actually improves the policy and respects the KL constraint
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
            
            # Condition: KL constraint satisfied AND Loss improved
            if kl <= max_kl * 1.5 and surr_loss_new > surr_loss:
                success = True
                break
            step_frac *= 0.5 # Shrink step size if failed
        
        if not success:
            vector_to_parameters(old_params, self.actor_parameters())

        # 5. CRITIC UPDATE (Standard Supervised Learning)
        optimizer_critic = torch.optim.Adam(self.critic_cnn.parameters(), lr=1e-3)
        optimizer_critic.add_param_group({'params': self.critic_value.parameters()})
        
        for _ in range(5): 
            values_pred = self.get_value(states).squeeze()
            value_loss = F.mse_loss(values_pred, returns)
            optimizer_critic.zero_grad()
            value_loss.backward()
            optimizer_critic.step()

    # ==============================================================================
    # 4. TRAINING LOOP & REWARD SHAPING
    # ==============================================================================

    def train(self, render=False, reward_shaping=0):
        # Hyperparameters
        total_epochs = 400
        timesteps_per_batch = 4096 
        gamma = 0.99
        lam = 0.95
        damping = 0.01
        max_kl = 0.01

        best_true_reward = -float('inf')
        patience = 50 
        no_improvement_count = 0

        if render:
            env = gym.make('CarRacing-v2', continuous=self.continuous, render_mode='human')
        else:
            env = gym.make('CarRacing-v2', continuous=self.continuous)

        state, _ = env.reset()
    
        
        print(f"Starting Training: {total_epochs} epochs expected.")

        if reward_shaping == 0:
            print(">>> Reward Shaping DISABLED: Optim Score is equivalent to True Score.")
        else:
            if reward_shaping == 1:
                print(">>> Reward Shaping LEVEL 1: Stability Penalty Active and Brake Reward DISABLED")
            elif reward_shaping == 2:
                print(">>> Reward Shaping LEVEL 2: Stability Penalty Active and Brake Reward ENABLED")


        for update in range(total_epochs):
            batch_states, batch_actions, batch_rewards = [], [], []
            batch_values, batch_log_probs, batch_dones = [], [], []
            batch_true_rewards = []
            
            for t in range(timesteps_per_batch):
                state_tensor = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
                action, log_prob, value = self._select_action_train(state_tensor)
                
                clipped_action = action.copy()
                clipped_action[0] = np.clip(clipped_action[0], -1, 1) 
                clipped_action[1] = np.clip(clipped_action[1], 0, 1) 
                clipped_action[2] = np.clip(clipped_action[2], 0, 1) 
                
                next_state, reward, terminated, truncated, _ = env.step(clipped_action)
                true_reward = reward 
                
                if reward_shaping != 0:

                    # --- REWARD SHAPING LOGIC -----------------------------------------
                    # This section solves the "High Velocity on Curves" problem.
                
                    steer = np.abs(clipped_action[0]) 
                    gas   = clipped_action[1]         
                    brake = clipped_action[2]         

                    # 1. STABILITY PENALTY
                    # Problem: Agent drives full gas while turning -> spin out.
                    # Solution: Penalize (Steer * Gas).
                    # Logic: If Steer is High (1.0) and Gas is High (1.0) -> Penalty -0.05
                    # This teaches the agent to lift off gas when entering a turn.
                    stability_penalty = 0.05 * steer * gas
                    reward -= stability_penalty

                    if reward_shaping == 2:
                        # 2. BRAKE REWARD
                        # Problem: Agent rarely uses brake, leading to overshooting corners.
                        # Solution: Small reward for using brake appropriately.
                        reward += 0.03 * brake * steer  # Encourages braking during turns 
                    # ------------------------------------------------------------------

                # Clip reward to avoid large updates
                #reward = np.clip(reward, -5.0, 1.0)

                done = terminated or truncated
                
                batch_states.append(state)
                batch_actions.append(action)
                batch_rewards.append(reward)
                batch_true_rewards.append(true_reward)
                batch_values.append(value)
                batch_log_probs.append(log_prob)
                batch_dones.append(done)
                
                state = next_state
                if done:
                    state, _ = env.reset()
            
            # --- GAE (Generalized Advantage Estimation) ---
            next_state_tensor = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.get_value(next_state_tensor).cpu().item()
            
            batch_advantages = []
            batch_returns = []
            last_gae_lam = 0
            
            for t in reversed(range(timesteps_per_batch)):
                if t == timesteps_per_batch - 1:
                    next_non_terminal = 1.0 - batch_dones[t]
                    next_val = next_value
                else:
                    next_non_terminal = 1.0 - batch_dones[t]
                    next_val = batch_values[t+1]
                
                delta = batch_rewards[t] + gamma * next_val * next_non_terminal - batch_values[t]
                last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
                batch_advantages.insert(0, last_gae_lam)
                batch_returns.insert(0, last_gae_lam + batch_values[t])
            
            # Package for Update
            batch = {
                'states': torch.from_numpy(np.array(batch_states)).float().permute(0, 3, 1, 2),
                'actions': torch.tensor(np.array(batch_actions)).float(),
                'returns': torch.tensor(np.array(batch_returns)).float(),
                'advantages': torch.tensor(np.array(batch_advantages)).float(),
                'old_log_probs': torch.tensor(np.array(batch_log_probs)).float()
            }
            
            self._trpo_step(batch, max_kl=max_kl, damping=damping)
            
            if (update + 1) % 1 == 0:
                with torch.no_grad():
                    current_std = torch.exp(self.actor_log_std).mean().item()
                
                current_true_score = np.mean(batch_true_rewards) * timesteps_per_batch / max(1, sum(batch_dones))
                current_optim_score = np.mean(batch_rewards) * timesteps_per_batch / max(1, sum(batch_dones))

                # Print progress
                print(f"Update {update+1}/{total_epochs} | "
                      f"True Score: {current_true_score:.2f} | "
                      f"Optim Score: {current_optim_score:.2f} | "
                      f"Std: {current_std:.4f}")
                

            # --- EARLY STOPPING CHECK ---
            
            # Check if this is the best model so far
            if current_true_score > best_true_reward:
                best_true_reward = current_true_score
                no_improvement_count = 0
                print(f"    >>> NEW RECORD! Saving best_model.pt (Score: {best_true_reward:.2f})")
                self.save('best_model.pt')
            else:
                no_improvement_count += 1
            
            # Check if we should stop
            if no_improvement_count >= patience:
                print(f"Early Stopping triggered! No improvement for {patience} updates.")
                print(f"Best Score was: {best_true_reward:.2f}")
                break
            
            # Optional: Stop immediately if "Solved" (Score > 900 consistently)
            if current_true_score > 900 and current_optim_score > 900:
                 print("Environment Solved! Stopping training.")
                 self.save('solved_model.pt')
                 break
            # ---------------------------------
                
        env.close()

    def save(self, filename='model.pt'):
        torch.save(self.state_dict(), filename)

    def load(self, filename='model.pt'):
        self.load_state_dict(torch.load(filename, map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    
