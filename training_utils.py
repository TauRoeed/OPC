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
from torch.utils.data import DataLoader, Dataset

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
# import debugpy
import numpy as np
import simulation_utils

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


def dros_shrinkage(iw: np.ndarray, lam: float):
    """Doubly Robust with optimistic shrinkage."""
    return (lam * iw) / (iw**2 + lam)


# sndr rewards for cross validation
def sndr_rewards(pscore, scores, policy_prob, original_policy_rewards, users, original_policy_actions, lam=3.0):
        
        pi_e_at_position = policy_prob[users, original_policy_actions].squeeze()
        iw = pi_e_at_position / pscore
        iw = dros_shrinkage(iw, lam=lam)

        q_hat_at_position = scores[users, original_policy_actions].squeeze()
        dm_reward = (scores * policy_prob)[users].sum(axis=1)
        
        r_hat = ((iw * (original_policy_rewards - q_hat_at_position))) + dm_reward

        return r_hat


# ipw rewards for cross validation
def ipw_rewards(pscore, policy_prob, original_policy_rewards, users, original_policy_actions):
        
        pi_e_at_position = policy_prob[users, original_policy_actions].squeeze()
        iw = pi_e_at_position / pscore

         # reinforce trick step
        r_hat = ((iw * (original_policy_rewards)))

        return r_hat


def perform_cv(ubiased_vec, estimator_vec, k=5):
    n = len(ubiased_vec)
    ratio = np.var(estimator_vec) / (np.var(ubiased_vec) + np.var(estimator_vec) + 1e-6)

    if ratio == 0 or ratio == 1 or np.isnan(ratio):
       ratio = 0.5
    
    ratio = max(0.2, ratio)
    ratio = min(0.8, ratio)

    estimator_size = int(n * ratio)
    results = []

    # Computing k-fold CV error estimates:
    for i in range(k):
        
        indices = np.random.default_rng().permutation(n)
        estimator_idx = indices[:estimator_size]
        unbiased_idx = indices[estimator_size:]
        res = (np.mean(ubiased_vec[unbiased_idx]) - np.mean(estimator_vec[estimator_idx]))**2

        results.append(res)

    results = np.array(results)

    #return results.mean() + (results.std() / np.sqrt(k))
    # return np.sqrt(results.mean() + results.std()) / np.sqrt(k) # note that this is different from the CV paper...
    return np.sqrt(results.mean() / k)


def cv_score_model(val_dataset, scores_all, policy_prob, lam=3.0):

    pscore = val_dataset['pscore']
    scores = scores_all.detach().cpu().numpy().squeeze()
    users = val_dataset['x_idx']
    reward = val_dataset['r']
    actions = val_dataset['a']

    prob = policy_prob[users, actions].squeeze()
    weights_info = simulation_utils.get_weights_info(prob, pscore)

    print(f'Validation weights_info: {weights_info}')

    # iw = prob / pscore
    
    # qq = np.quantile(iw, q=[q, 1-q])
    # mask = (iw > qq[0]) & (iw < qq[1])

    # users = users[mask]
    # actions = actions[mask]
    # reward = reward[mask]
    # pscore = pscore[mask]
    # prob = prob[mask]

    sndr_vec = sndr_rewards(pscore, scores, policy_prob, reward, users, actions, lam=lam)
    ipw_vec = ipw_rewards(pscore, policy_prob, reward, users, actions)

    err = perform_cv(sndr_vec, ipw_vec, k=100)
    
    r_hat = sndr_vec.mean()
    se_hat = sndr_vec.std() / np.sqrt(len(sndr_vec))

    print(f"Estimated reward: {r_hat:.6f}")

    print(f"Cross-validated error: {err:.6f}")
    print(f"Final score CI (reward +- 2*error): [{r_hat - 2 * err:.6f}, {r_hat + 2 * err:.6f}]")

    se = scipy.stats.t.ppf(0.975, len(sndr_vec)-1) * se_hat
    print(f"Standard error: {se_hat:.6f}")
    print(f"Final t_dist CI (reward +- t_0.975*se_hat): [{r_hat - se:.6f}, {r_hat + se:.6f}]")

    if weights_info['ess'] < len(reward) * 0.01:
        print("Warning: Low ESS in validation data!")
        return -np.inf, np.inf
    
    return r_hat, err


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
            

def compute_statistics_and_plots(df, n_bins=20):
    """
    Computes:
      - Pearson correlation
      - Spearman rank correlations
      - NDCG(score, actual)
      - NDCG(estimated, actual)

    Generates:
      - scatter
      - hexbin
      - KDE (raw)
      - KDE (centered)
      - Log-KDE + fitted log-normal PDF
      - Centered Log-KDE + fitted log-normal PDF
      - calibration curve
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde, spearmanr, lognorm

    # ===============================
    # Helper functions
    # ===============================
    def dcg(relevances):
        rel = np.asarray(relevances)
        return np.sum((2**rel - 1) / np.log2(np.arange(1, len(rel)+1) + 1))

    def ndcg(scores, rewards):
        scores = np.asarray(scores)
        rewards = np.asarray(rewards)
        order = np.argsort(scores)[::-1]
        ranked_rel = rewards[order]
        ideal = dcg(np.sort(rewards)[::-1])
        return dcg(ranked_rel) / ideal if ideal > 0 else 0.0

    # shorthand
    score = df["value"].values
    est = df["user_attrs_r_hat"].values
    actual = df["user_attrs_actual_reward"].values

    # ===============================
    # Correlations
    # ===============================
    pearson_corr = np.corrcoef(score, actual)[0, 1]
    spearman_score_actual = spearmanr(score, actual).correlation
    spearman_est_actual = spearmanr(est, actual).correlation

    ndcg_score_vs_actual = ndcg(score, actual)
    ndcg_estimated_vs_actual = ndcg(est, actual)

    # ===============================
    # Scatter
    # ===============================
    plt.figure(figsize=(6,5))
    plt.scatter(score, actual, alpha=0.4)
    plt.xlabel("Score")
    plt.ylabel("Actual Reward")
    plt.title("Scatter: Score vs Actual Reward")
    plt.grid(True)
    plt.show()

    # ===============================
    # Hexbin
    # ===============================
    plt.figure(figsize=(6,5))
    hb = plt.hexbin(score, actual, gridsize=40, cmap='viridis', mincnt=1)
    plt.colorbar(hb, label="Count")
    plt.xlabel("Score")
    plt.ylabel("Actual Reward")
    plt.title("Hexbin: Score vs Actual Reward")
    plt.show()

    # ===============================
    # KDE (raw)
    # ===============================
    def kde(values):
        kde_obj = gaussian_kde(values)
        xs = np.linspace(values.min(), values.max(), 300)
        return xs, kde_obj(xs)

    plt.figure(figsize=(6,5))
    xs, ys = kde(score); plt.plot(xs, ys, label="Score KDE")
    xs, ys = kde(est);   plt.plot(xs, ys, label="Estimated Reward KDE")
    xs, ys = kde(actual);plt.plot(xs, ys, label="Actual Reward KDE")
    plt.legend(); plt.grid(True)
    plt.title("KDE Distributions")
    plt.show()

    # ===============================
    # Centered KDE
    # ===============================
    plt.figure(figsize=(6,5))
    xs, ys = kde(score - score.mean()); plt.plot(xs, ys, label="Score (centered)")
    xs, ys = kde(est - est.mean());     plt.plot(xs, ys, label="Estimated (centered)")
    xs, ys = kde(actual - actual.mean()); plt.plot(xs, ys, label="Actual (centered)")
    plt.legend(); plt.grid(True)
    plt.title("Centered KDE Distributions")
    plt.show()

    # ===============================
    # Log-KDE + fitted Log-Normal PDFs
    # ===============================
    eps = 1e-8

    def log_kde(vals):
        logvals = np.log(vals + eps)
        kde_obj = gaussian_kde(logvals)
        xs = np.linspace(logvals.min(), logvals.max(), 500)
        return xs, kde_obj(xs), logvals

    def lognormal_pdf_fit(logvals, xs):
        # Fit lognormal parameters using raw values
        sigma, loc, scale = lognorm.fit(np.exp(logvals), floc=0)
        pdf = lognorm.pdf(np.exp(xs), sigma, loc=0, scale=scale) * np.exp(xs)
        return pdf, (sigma, scale)

    # Raw log-KDE plot
    plt.figure(figsize=(7,5))

    for arr, label in [(score, "Score"), (est, "Estimated"), (actual, "Actual")]:
        xs, ys, logvals = log_kde(arr)
        plt.plot(xs, ys, label=f"{label} Log-KDE", linewidth=2)

        # Fit true lognormal
        pdf, params = lognormal_pdf_fit(logvals, xs)
        plt.plot(xs, pdf, '--', linewidth=1.5, label=f"{label} LogNormal fit")

    plt.title("Log-KDE + Log-Normal Fit")
    plt.xlabel("log(Value)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ===============================
    # Centered Log-KDE + fits
    # ===============================
    plt.figure(figsize=(7,5))

    for arr, label in [
        (score - score.mean(), "Score (centered)"),
        (est - est.mean(),   "Estimated (centered)"),
        (actual - actual.mean(), "Actual (centered)")
    ]:
        xs, ys, logvals = log_kde(arr - arr.min() + eps)  # shift to positive for log
        plt.plot(xs, ys, label=f"{label} Log-KDE", linewidth=2)

        pdf, params = lognormal_pdf_fit(logvals, xs)
        plt.plot(xs, pdf, '--', linewidth=1.5, label=f"{label} LogNormal fit")

    plt.title("Centered Log-KDE + Log-Normal Fit")
    plt.xlabel("log(Value Shifted)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ===============================
    # Calibration curve
    # ===============================
    df_sorted = df.sort_values("value")
    bins = np.array_split(df_sorted, n_bins)
    avg_pred = [b["value"].mean() for b in bins]
    avg_actual = [b["user_attrs_actual_reward"].mean() for b in bins]

    plt.figure(figsize=(6,5))
    plt.plot(avg_pred, avg_actual, marker="o", label="Calibration Curve")
    lo, hi = min(avg_pred), max(avg_pred)
    plt.plot([lo, hi], [lo, hi], 'k--', label="Perfect Calibration")
    plt.xlabel("Mean Score (bin)")
    plt.ylabel("Mean Actual Reward (bin)")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ===============================
    # Return metrics
    # ===============================
    return {
        "pearson_corr": pearson_corr,
        "spearman_score_actual": spearman_score_actual,
        "spearman_est_actual": spearman_est_actual,
        "ndcg_score_vs_actual_reward": ndcg_score_vs_actual,
        "ndcg_estimated_vs_actual_reward": ndcg_estimated_vs_actual,
    }