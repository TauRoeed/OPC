import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.append("/code")

import torch
import torch.optim as optim
import scipy
from sklearn.utils import check_random_state

from models.custom_losses import sndr_r_hat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = torch.cuda.is_available()
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

random_state = 12345
random_ = check_random_state(random_state)

# Set True only when debugging; `.item()` syncs the device.
CHECK_POLICY_NAN = False


def calc_estimated_policy_rewards(
    pscore, scores, policy_prob, original_policy_rewards, original_policy_actions
):
    """Per-row SNDR value estimates (same estimator as Optuna validation)."""
    n = original_policy_actions.shape[0]
    idx = torch.arange(n, device=policy_prob.device)

    pi_e_at_position = policy_prob[idx, original_policy_actions].squeeze()
    iw = (pi_e_at_position / pscore).detach()
    q_hat_at_position = scores[idx, original_policy_actions].squeeze()
    dm_reward = (scores * policy_prob.detach()).sum(dim=1)
    return sndr_r_hat(iw, original_policy_rewards, q_hat_at_position, dm_reward)


def _set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def train(
    model,
    train_loader,
    scores_all,
    criterion,
    num_epochs=1,
    lr=1e-4,
    lr_decay=1.0,
    device="cpu",
    log_gpu=False,
    optimizer=None,
    check_nan: bool = CHECK_POLICY_NAN,
):
    """Train ``num_epochs`` with one Adam (momentum preserved across epochs)."""
    model.to(device).train()
    if hasattr(criterion, "to"):
        criterion = criterion.to(device)

    if torch.cuda.is_available():
        assert next(model.parameters()).is_cuda, "Model is on CPU!"

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        _set_optimizer_lr(optimizer, lr)

    current_lr = float(lr)
    for epoch in range(num_epochs):
        if epoch > 0:
            current_lr *= float(lr_decay)
            _set_optimizer_lr(optimizer, current_lr)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        run_train_loop(
            model,
            train_loader,
            optimizer,
            scores_all,
            criterion,
            device=device,
            check_nan=check_nan,
        )

        if log_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()
            print(
                f"[epoch {epoch + 1}/{num_epochs}] "
                f"alloc={torch.cuda.memory_allocated() / 1024 ** 2:.0f}MB "
                f"peak={torch.cuda.max_memory_allocated() / 1024 ** 2:.0f}MB",
                flush=True,
            )

    return optimizer


def run_train_loop(
    model,
    train_loader,
    optimizer,
    scores_all,
    criterion,
    device="cpu",
    check_nan: bool = CHECK_POLICY_NAN,
):
    model.train()
    if torch.cuda.is_available():
        assert next(model.parameters()).is_cuda, "Model is on CPU!"

    needs_qhat = bool(getattr(criterion, "needs_qhat", True))

    for step, (user_idx, action_idx, rewards, pscore) in enumerate(train_loader, 1):
        user_idx = user_idx.to(device, non_blocking=True)
        action_idx = action_idx.to(device, non_blocking=True)
        rewards = rewards.to(device, non_blocking=True)
        pscore = pscore.to(device, non_blocking=True)

        policy = model(user_idx)
        if policy.dim() == 3 and policy.shape[-1] == 1:
            policy = policy.squeeze(-1)

        if check_nan and torch.isnan(policy).any().item():
            print(f"NaN in policy : (, step {step})")
            break

        if needs_qhat:
            scores = scores_all[user_idx.long()]
            if scores.dim() == 3 and scores.shape[-1] == 1:
                scores = scores.squeeze(-1)
            if scores.shape[-1] != policy.shape[-1]:
                raise RuntimeError(
                    f"scores/policy action dim mismatch: scores {tuple(scores.shape)} "
                    f"vs policy {tuple(policy.shape)} (check regression catalog n_actions)"
                )
        else:
            scores = None

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(pscore, scores, policy, rewards, action_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0, error_if_nonfinite=True
        )
        optimizer.step()


def validation_loop(model, val_loader, scores_all, device="cpu"):
    """Full-split conservative CI: mean(DR) - t_{0.975,n-1} * se (not batch-averaged)."""
    model.to(device).eval()
    if torch.cuda.is_available():
        assert next(model.parameters()).is_cuda

    row_values = []

    with torch.no_grad():
        for user_idx, action_idx, rewards, pscore in val_loader:
            user_idx = user_idx.to(device, non_blocking=True)
            action_idx = action_idx.to(device, non_blocking=True)
            rewards = rewards.to(device, non_blocking=True)
            pscore = pscore.to(device, non_blocking=True)
            policy = model(user_idx)
            if policy.dim() == 3 and policy.shape[-1] == 1:
                policy = policy.squeeze(-1)

            scores = scores_all[user_idx.long()]
            row_values.append(
                calc_estimated_policy_rewards(
                    pscore, scores, policy, rewards, action_idx.long()
                )
            )

    r_hat = torch.cat(row_values)
    n = max(int(r_hat.numel()), 2)
    mean = float(r_hat.mean().item())
    se = float(r_hat.std(unbiased=True).item() / (n**0.5))
    tcrit = float(scipy.stats.t.ppf(0.975, n - 1))
    return dict(value=mean - tcrit * se, variance=float(r_hat.std(unbiased=True).item()))
