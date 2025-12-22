import warnings

from matplotlib.pyplot import step
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

# Top of notebook (once)
torch.backends.cudnn.benchmark = torch.cuda.is_available()
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")  # TF32 = big speedup on Ada


from custom_losses import BPRLoss
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt

# implementing OPE of the IPWLearner using synthetic bandit data
import scipy
from scipy.special import softmax

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
        lower_bound = r_hat.mean() - (scipy.stats.t.ppf(0.95, n - 1) * var_hat / (n ** 0.5))
        
        return lower_bound


# 4. Define the training function
def train(model, train_loader, scores_all,  criterion, num_epochs=1, lr=1e-4, device='cpu', log_gpu=False):
    model.to(device).train()
    if hasattr(criterion, "to"):
        criterion = criterion.to(device)

    # PROBE: explode if the model isn’t on CUDA (when CUDA is available)
    if torch.cuda.is_available():
        assert next(model.parameters()).is_cuda, "Model is on CPU!"

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        run_train_loop(model, train_loader, optimizer, scores_all, criterion, lr=lr, device=device)

        if log_gpu:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                print(
                    f"[epoch {epoch+1}/{num_epochs}] "
                    f"alloc={torch.cuda.memory_allocated()/1024**2:.0f}MB "
                    f"peak={torch.cuda.max_memory_allocated()/1024**2:.0f}MB",
                    flush=True,
                )


# 5. Define the training loop
def run_train_loop(model, train_loader, optimizer, scores_all, criterion, lr=1e-4, device='cpu'):
    model.train()
    # (Optional) assert once:
    if torch.cuda.is_available():
        assert next(model.parameters()).is_cuda, "Model is on CPU!"

    for step, (user_idx, action_idx, rewards, original_prob) in enumerate(train_loader, 1):
        # Move batch to device
        user_idx      = user_idx.to(device, non_blocking=True)
        action_idx    = action_idx.to(device, non_blocking=True)
        rewards       = rewards.to(device, non_blocking=True)
        original_prob = original_prob.to(device, non_blocking=True)

        # PROBE: assert batch is on CUDA when available
        if torch.cuda.is_available():
            assert user_idx.is_cuda and action_idx.is_cuda and rewards.is_cuda and original_prob.is_cuda, \
                "Batch tensors not on CUDA"

        # Forward
        policy = model(user_idx)  # stays on device

        if torch.isnan(policy).max().item() == True:
            print(f"NaN in policy : (, step {step})")
            break
            
        pscore = original_prob[torch.arange(user_idx.shape[0], device=device), action_idx]

        # *** Replace CPU round-trip with precomputed GPU lookup ***
        # scores = torch.tensor(neighborhood_model.predict(user_idx.cpu().numpy()), device=device)

        scores = scores_all[user_idx.long()]   # <-- from section A

        optimizer.zero_grad()

        loss = criterion(pscore, scores, policy, rewards, action_idx)
        loss.backward()

        # grad_user = model.user_transform.delta.clone()
        # grad_action = model.action_transform.delta.clone()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)

        optimizer.step()


# 6. Define the validation function
def validation_loop(model, val_loader, scores_all, device='cpu'):
    model.to(device).eval()
    if torch.cuda.is_available():
        assert next(model.parameters()).is_cuda

    estimated_rewards = []

    with torch.no_grad():
        for user_idx, action_idx, rewards, original_prob in val_loader:
            user_idx      = user_idx.to(device, non_blocking=True)
            action_idx    = action_idx.to(device, non_blocking=True)
            rewards       = rewards.to(device, non_blocking=True)
            original_prob = original_prob.to(device, non_blocking=True)

            policy = model(user_idx)
            pscore = original_prob[torch.arange(user_idx.shape[0], device=device), action_idx.long()]

            # scores on GPU via lookup
            scores = scores_all[user_idx.long()]

            batch_reward = calc_estimated_policy_rewards(
                pscore, scores, policy, rewards, action_idx.long()
            )
            # Make sure we collect a tensor, then average at the end
            estimated_rewards.append(batch_reward)

    avg = torch.stack(estimated_rewards).mean().item()
    std = torch.stack(estimated_rewards).std().item()

    return dict(value=avg, variance=std)


def fit_bpr(model, data_loader, loss_fn=BPRLoss(), num_epochs=5, lr=0.001, device=device):
    model.to(device)
    if torch.cuda.is_available():
        assert next(model.parameters()).is_cuda
    optimizer = optim.Adam(model.parameters(), lr=lr) # here we can change the learning rate

    model.train() # Set the model to training mode
    tq = tqdm(range(num_epochs))
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    for epoch in tq:
        running_loss = 0.0
        total_samples = 0

        n_steps = len(data_loader)  # <— this works for most DataLoaders    
        for step, (user_idx, action_idx, rewards, original_prob) in enumerate(data_loader, 1):
        
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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Calculate running loss and accuracy
            running_loss += loss.item()
            total_samples += 1

            # Print statistics after each epoch
            epoch_loss = running_loss / total_samples
            tq.set_description(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(
                f"[epoch {epoch+1}/{num_epochs}] "
                f"alloc={torch.cuda.memory_allocated()/1024**2:.0f}MB "
                f"peak={torch.cuda.max_memory_allocated()/1024**2:.0f}MB",
                flush=True,
            )
        