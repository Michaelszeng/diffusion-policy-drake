"""
PPO training for the Drake Push-T task.

Adapted from ManiSkill's ppo_fast.py for use with CPU-based Drake simulations.
Uses gymnasium.vector.AsyncVectorEnv to parallelize across CPU cores.

Example usage:
    python rl_push_t/ppo.py --n_envs=16 --total_timesteps=100_000_000

    # Auto-resume (looks for latest.ckpt in the run directory):
    python rl_push_t/ppo.py --exp_name=push_t_ppo --resume

    # Resume from a specific checkpoint (weights only, no training state):
    python rl_push_t/ppo.py --checkpoint=runs/<name>/model.pt

    # Evaluate:
    python rl_push_t/ppo.py --evaluate --checkpoint=runs/<name>/model.pt
"""

import argparse
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pydrake.all import Meshcat
from torch.distributions.normal import Normal

import wandb
from rl_push_t.envs.push_t_gym_env import PushTDrakeEnv

# ── Neural network ─────────────────────────────────────────────────────────────


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Shared-trunk MLP actor-critic (3 hidden layers, 256 units, Tanh activations)."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        std = logstd.exp()
        dist = Normal(mean, std, validate_args=False)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.critic(x)


# ── Argument parsing ───────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--wandb_project", type=str, default="drake-rl-push-t")
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-resume from latest.ckpt in the run directory if it exists",
    )

    # Env
    parser.add_argument("--cfg_path", type=str, default="rl_push_t/configs/rl_env.yaml")
    parser.add_argument("--debug", action="store_true", help="n_envs=1 with meshcat visualization")
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=50)

    # Algorithm
    parser.add_argument("--total_timesteps", type=int, default=500_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--anneal_lr", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.9)
    parser.add_argument("--num_minibatches", type=int, default=32)
    parser.add_argument("--update_epochs", type=int, default=8)
    parser.add_argument("--norm_adv", action="store_true", default=True)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--clip_vloss", action="store_true")
    parser.add_argument("--ent_coef", type=float, default=0.005)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=0.1)

    # Log dir
    parser.add_argument("--log_dir", type=str, default="rl_push_t/runs")

    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────


def make_env(cfg_path: str, meshcat=None):
    def _init():
        env = PushTDrakeEnv(cfg_path=cfg_path, meshcat=meshcat)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return _init


def evaluate(
    agent: Agent,
    cfg_path: str,
    num_episodes: int,
    device: torch.device,
    meshcat=None,
    recording_path: str = None,
):
    """Run a fixed number of evaluation episodes (single env) and return mean success rate and overlap.

    Always saves a Meshcat HTML recording to recording_path if provided, creating a local Meshcat
    instance for recording if one is not passed in.
    """
    print(f"Running evaluation for {num_episodes} episodes...")
    # Use provided meshcat or create a local one just for recording
    eval_meshcat = meshcat if meshcat is not None else (Meshcat() if recording_path is not None else None)

    envs = gym.vector.SyncVectorEnv([make_env(cfg_path, meshcat=eval_meshcat)])
    obs, _ = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    if eval_meshcat is not None:
        eval_meshcat.StartRecording()

    successes = []
    overlaps = []

    with torch.no_grad():
        # Continue stepping until we have num_episodes records
        while len(successes) < num_episodes:
            action, _, _, _ = agent.get_action_and_value(obs)
            obs_np, _, terminated, truncated, infos = envs.step(action.cpu().numpy())
            obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
            if terminated[0] or truncated[0]:
                successes.append(float(infos["success"][0]))
                overlaps.append(float(infos["overlap"][0]))

    if eval_meshcat is not None:
        eval_meshcat.StopRecording()
        eval_meshcat.PublishRecording()
        if recording_path is not None:
            with open(recording_path, "w") as f:
                f.write(eval_meshcat.StaticHtml())
            print(f"  [eval] Saved recording to {recording_path}")

    sr, ov = float(np.mean(successes)), float(np.mean(overlaps))
    print(f"  [eval] success_rate={sr:.3f}  mean_overlap={ov:.3f}")
    envs.close()
    return sr, ov


def _save_checkpoint(path, agent, optimizer, obs_dim, act_dim, args, iteration, global_step, wandb_run_id):
    """Save a full training checkpoint (model + optimizer + training progress)."""
    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "args": vars(args),
            "iteration": iteration,
            "global_step": global_step,
            "wandb_run_id": wandb_run_id,
        },
        path,
    )


def main():
    args = parse_args()

    exp_name = args.exp_name or "push_t_ppo"

    # Find the next available run number
    run_num = 1
    if not args.resume:
        while os.path.exists(os.path.join(args.log_dir, f"{exp_name}_{run_num}")):
            run_num += 1
    else:
        # When resuming, find the latest existing run
        while os.path.exists(os.path.join(args.log_dir, f"{exp_name}_{run_num + 1}")):
            run_num += 1

    run_name = f"{exp_name}_{run_num}"
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # ── Resume logic ──────────────────────────────────────────────────────────
    latest_ckpt_path = os.path.join(log_dir, "latest.ckpt")
    resume_ckpt = None
    if args.resume and os.path.isfile(latest_ckpt_path):
        print(f"Found existing checkpoint at {latest_ckpt_path}, resuming...")
        args.checkpoint = latest_ckpt_path
        resume_ckpt = torch.load(latest_ckpt_path, map_location="cpu")
    elif args.resume:
        print("--resume passed but no latest.ckpt found; starting from scratch.")

    wandb_resume = "must" if resume_ckpt is not None else None
    wandb_run_id = resume_ckpt["wandb_run_id"] if resume_ckpt is not None else wandb.util.generate_id()
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        id=wandb_run_id,
        resume=wandb_resume,
        config=vars(args),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    # ── Environments ───────────────────────────────────────────────────────────
    if args.debug:
        args.n_envs = 1
        meshcat = Meshcat()
        print(f"Debug mode: n_envs=1, meshcat at {meshcat.web_url()}")
        # SyncVectorEnv runs in the same process so Drake can publish to Meshcat. AsyncVectorEnv causes Meshcat problems
        envs = gym.vector.SyncVectorEnv([make_env(args.cfg_path, meshcat=meshcat)])
    else:
        meshcat = None
        print(f"Creating {args.n_envs} parallel Drake environments...")
        envs = gym.vector.AsyncVectorEnv([make_env(args.cfg_path) for _ in range(args.n_envs)])

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    print(f"obs_dim={obs_dim}, act_dim={act_dim}")

    # ── Agent ──────────────────────────────────────────────────────────────────
    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    start_iteration = 0
    start_global_step = 0
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        agent.load_state_dict(ckpt["agent_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "iteration" in ckpt:
            start_iteration = ckpt["iteration"]
        if "global_step" in ckpt:
            start_global_step = ckpt["global_step"]
        print(
            f"Loaded checkpoint from {args.checkpoint} (iteration={start_iteration}, global_step={start_global_step})"
        )

    if args.evaluate:
        sr, ov = evaluate(
            agent,
            args.cfg_path,
            args.num_eval_episodes,
            device,
            meshcat=meshcat,
            recording_path=os.path.join(log_dir, "eval.html"),
        )
        envs.close()
        wandb.finish()
        return

    # ── Rollout buffers ────────────────────────────────────────────────────────
    n_envs = args.n_envs
    num_steps = args.num_steps
    batch_size = n_envs * num_steps
    minibatch_size = max(1, batch_size // args.num_minibatches)

    obs_buf = torch.zeros((num_steps, n_envs, obs_dim), device=device)
    actions_buf = torch.zeros((num_steps, n_envs, act_dim), device=device)
    logprobs_buf = torch.zeros((num_steps, n_envs), device=device)
    rewards_buf = torch.zeros((num_steps, n_envs), device=device)
    dones_buf = torch.zeros((num_steps, n_envs), device=device)
    values_buf = torch.zeros((num_steps, n_envs), device=device)

    # ── Training loop ──────────────────────────────────────────────────────────
    num_iterations = args.total_timesteps // batch_size
    global_step = start_global_step

    next_obs_np, _ = envs.reset()
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(n_envs, device=device)

    start_time = time.time()

    # ── Initial evaluation (step 0) — skip if resuming ────────────────────────
    if start_iteration == 0:
        sr, ov = evaluate(
            agent,
            args.cfg_path,
            args.num_eval_episodes,
            device,
            meshcat=meshcat,
            recording_path=os.path.join(log_dir, "eval_0.html"),
        )
        wandb.log({"eval/success_rate": sr, "eval/mean_overlap": ov}, step=0)
    else:
        print(f"Resuming from iteration {start_iteration}, global_step {global_step}")

    for iteration in range(start_iteration + 1, num_iterations + 1):
        # Optional learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.lr

        # ── Rollout collection ────────────────────────────────────────────────
        ep_returns = []
        ep_lengths = []

        for step in range(num_steps):
            global_step += n_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(action.cpu().numpy())
            next_obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
            rewards_buf[step] = torch.tensor(reward_np, dtype=torch.float32, device=device)
            next_done = torch.tensor((terminated_np | truncated_np), dtype=torch.float32, device=device)

            # Log episode stats
            if "episode" in infos:
                for i in range(n_envs):
                    if infos["_episode"][i]:
                        ep_returns.append(infos["episode"]["r"][i])
                        ep_lengths.append(infos["episode"]["l"][i])
            if "final_info" in infos:
                for fi in infos["final_info"]:
                    if fi is not None and "episode" in fi:
                        ep_returns.append(fi["episode"]["r"])
                        ep_lengths.append(fi["episode"]["l"])

        # ── GAE computation ───────────────────────────────────────────────────
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # ── PPO update ────────────────────────────────────────────────────────
        b_obs = obs_buf.reshape((-1, obs_dim))
        b_actions = actions_buf.reshape((-1, act_dim))
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        clipfracs = []
        for epoch in range(args.update_epochs):
            idx = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = idx[start : start + minibatch_size]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_idx], b_actions[mb_idx])
                logratio = newlogprob - b_logprobs[mb_idx]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb_idx]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(correction=0) + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_clipped = b_values[mb_idx] + torch.clamp(
                        newvalue - b_values[mb_idx], -args.clip_coef, args.clip_coef
                    )
                    v_loss = (
                        0.5
                        * torch.max(
                            (newvalue - b_returns[mb_idx]) ** 2,
                            (v_clipped - b_returns[mb_idx]) ** 2,
                        ).mean()
                    )
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_idx]) ** 2).mean()

                loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ── Logging ───────────────────────────────────────────────────────────
        sps = int(global_step / (time.time() - start_time))
        log = {
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy.mean().item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "charts/SPS": sps,
        }
        if ep_returns:
            log["train/episodic_return"] = np.mean(ep_returns)
            log["train/episodic_length"] = np.mean(ep_lengths)
        wandb.log(log, step=global_step)

        if iteration % 10 == 0:
            print(
                f"iter={iteration}/{num_iterations} | step={global_step} | SPS={sps} | ep_ret={np.mean(ep_returns):.3f}"
                if ep_returns
                else ""
            )

        # ── Periodic evaluation ───────────────────────────────────────────────
        if iteration % args.eval_freq == 0:
            sr, ov = evaluate(
                agent,
                args.cfg_path,
                args.num_eval_episodes,
                device,
                meshcat=meshcat,
                recording_path=os.path.join(log_dir, f"eval_{global_step}.html"),
            )
            wandb.log({"eval/success_rate": sr, "eval/mean_overlap": ov}, step=global_step)

        # ── Save checkpoint ───────────────────────────────────────────────────
        if args.save_model and iteration % 100 == 0:
            model_path = os.path.join(log_dir, f"model_{iteration}.pt")
            _save_checkpoint(model_path, agent, optimizer, obs_dim, act_dim, args, iteration, global_step, wandb_run_id)
            _save_checkpoint(
                latest_ckpt_path, agent, optimizer, obs_dim, act_dim, args, iteration, global_step, wandb_run_id
            )
            print(f"  Saved checkpoint: {model_path} + latest.ckpt")

    # Final save
    if args.save_model:
        model_path = os.path.join(log_dir, "model.pt")
        _save_checkpoint(model_path, agent, optimizer, obs_dim, act_dim, args, iteration, global_step, wandb_run_id)
        _save_checkpoint(
            latest_ckpt_path, agent, optimizer, obs_dim, act_dim, args, iteration, global_step, wandb_run_id
        )
        print(f"Saved final model to {model_path} + latest.ckpt")

    envs.close()
    wandb.finish()


if __name__ == "__main__":
    main()
