import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("/code")

from tqdm import tqdm
import torch
# device = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# from memory_profiler import profile

import torch.nn.functional as F
import torch.optim as optim
from custom_losses import BPRLoss
from sklearn.utils import check_random_state

# implementing OPE of the IPWLearner using synthetic bandit data
import scipy
from scipy.special import softmax
# import debugpy


random_state=12345
random_ = check_random_state(random_state)


def calc_estimated_policy_rewards(pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        n = original_policy_actions.shape[0]

        pi_e_at_position = policy_prob[torch.arange(n), original_policy_actions].squeeze()
        iw = pi_e_at_position / pscore
        iw = iw.detach()
        q_hat_at_position = scores[torch.arange(n), original_policy_actions].squeeze()
        dm_reward = (scores * policy_prob.detach()).sum(dim=1)
        
        r_hat = ((iw * (original_policy_rewards - q_hat_at_position)) / iw.sum()) + dm_reward

        var_hat = r_hat.std()

        return r_hat.mean() - scipy.stats.t.ppf(0.95, n - 1) * var_hat


# 4. Define the training function
def train(model, train_loader, neighborhood_model, criterion, num_epochs=1, lr=0.0001, device='cpu'):
    model.to(device)

    for epoch in range(num_epochs):
        run_train_loop(model, train_loader, neighborhood_model, criterion, lr=lr, device=device)


# 4. Define the training function
def run_train_loop(model, train_loader, neighborhood_model, criterion, lr=0.0001, device='cpu'):

    optimizer = optim.Adam(model.parameters(), lr=lr) # here we can change the learning rate
    model.train() # Set the model to training mode

    for user_idx, action_idx, rewards, original_prob in train_loader:
        # Move data to GPU if available
        if torch.cuda.is_available():
            user_idx = user_idx.to(device) 
            action_idx = action_idx.to(device)
            rewards = rewards.to(device)
            original_prob = original_prob.to(device) 
        
        # Forward pass
        policy = model(user_idx)
        pscore = original_prob[torch.arange(user_idx.shape[0]), action_idx.type(torch.long)]
        
        scores = torch.tensor(neighborhood_model.predict(user_idx.cpu().numpy()), device=device)
        
        loss = criterion(
                            pscore,
                            scores,
                            policy, 
                            rewards, 
                            action_idx.type(torch.long), 
                            )
        
        # Zero the gradients Backward pass and optimization
        optimizer.zero_grad()

        loss.backward()                        
        optimizer.step()


# 4. Define the training function
def validation_loop(model, val_loader, neighborhood_model, device='cpu'):

    model.to(device)

    model.eval() # Set the model to evaluation mode
    estimated_rewards = 0.0

    for user_idx, action_idx, rewards, original_prob in val_loader:
        # Move data to GPU if available
        if torch.cuda.is_available():
            user_idx = user_idx.to(device) 
            action_idx = action_idx.to(device)
            rewards = rewards.to(device)
            original_prob = original_prob.to(device) 
        
        # Forward pass
        policy = model(user_idx)
        pscore = original_prob[torch.arange(user_idx.shape[0]), action_idx.type(torch.long)]
        
        scores = torch.tensor(neighborhood_model.predict(user_idx.cpu().numpy()), device=device)

        batch_reward = calc_estimated_policy_rewards(
            pscore, scores, policy, rewards, action_idx.type(torch.long)
        )
        
        estimated_rewards += batch_reward
        if batch_reward.item() == None:
            print("Estimated rewards is None, returning 0.0")
            print(f"pscore: {pscore.item()}, scores: {scores.item()}, policy: {policy.item()}, rewards: {rewards.item()}, action_idx: {action_idx.item()}")
            return 0.0
    
    return estimated_rewards.mean().item()



def fit_bpr(model, data_loader, loss_fn=BPRLoss(), num_epochs=5, lr=0.001, device=device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr) # here we can change the learning rate

    model.train() # Set the model to training mode
    tq = tqdm(range(num_epochs))
    for epoch in tq:
        running_loss = 0.0
        total_samples = 0
        
        for user_idx, action_idx, rewards, original_prob in data_loader:
            # Move data to GPU if available
            if torch.cuda.is_available():
                user_idx = user_idx.to(device) 
                action_idx = action_idx.to(device)
                rewards = rewards.to(device)
                original_prob = original_prob.to(device) 
            
            # Forward pass
            policy = model.calc_scores(user_idx)
            pscore = original_prob[torch.arange(user_idx.shape[0]), action_idx.type(torch.long)]
            
            # scores = torch.tensor(model.calc_scores(user_idx.numpy()), device=device)
            scores = policy.clone()
            
            loss = loss_fn(
                            pscore,
                            scores,
                            policy, 
                            rewards, 
                            action_idx.type(torch.long), 
                            )
            
            # Zero the gradients Backward pass and optimization
            optimizer.zero_grad()

            loss.backward()                        
            optimizer.step()
            
            # update neighborhood
            # action_emb, context_emb = model.get_params()
            
            # Calculate running loss and accuracy
            running_loss += loss.item()
            total_samples += 1

            # Print statistics after each epoch
            epoch_loss = running_loss / total_samples
            tq.set_description(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")