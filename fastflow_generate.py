# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Training-free FastFlow sampling for Wan2.1 T2V 1.3B.

FastFlow is described in "FastFlow: Accelerating the Generative Flow Matching
Models with Bandit Inference". The implementation below applies the paper's
finite-difference velocity extrapolation and UCB bandit controller to Wan2.1's
flow-matching sigma schedule. The Wan DiT is not modified.

Copy this file to the root of the Wan2.1 repository and run it from there.
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import gc
import json
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import cache_video, str2bool


warnings.filterwarnings("ignore")


EXAMPLE_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves "
    "fight intensely on a spotlighted stage."
)


@dataclass
class FastFlowConfig:
    """Runtime controls for the FastFlow controller."""

    arms: Tuple[int, ...] = (0, 2, 4, 6)
    gamma: float = 2.0
    mu: Optional[float] = None
    bootstrap_arm: Optional[int] = None
    error_low: float = 0.03
    error_high: float = 0.10
    stats_path: Optional[str] = None
    verbose: bool = False
    max_mse: float = 0.0
    target_arm: Optional[int] = None
    bandits: Dict[int, "UCBBandit"] = field(default_factory=dict)
    model_evaluations: int = 0
    baseline_evaluations: int = 0

    def __post_init__(self) -> None:
        self.arms = tuple(sorted(set(int(arm) for arm in self.arms)))
        if not self.arms or self.arms[0] != 0:
            raise ValueError("FastFlow arms must be non-empty and contain 0.")
        if any(arm < 0 for arm in self.arms):
            raise ValueError("FastFlow arms must be non-negative.")
        if self.gamma < 0:
            raise ValueError("FastFlow gamma must be non-negative.")
        if self.mu is not None and self.mu < 0:
            raise ValueError("FastFlow mu must be non-negative.")
        if not 0 <= self.error_low < self.error_high:
            raise ValueError("FastFlow error thresholds must satisfy 0 <= low < high.")
        if self.bootstrap_arm is None:
            middle = max(0, len(self.arms) // 2 - 1)
            self.bootstrap_arm = self.arms[middle]
        if self.target_arm is None:
            self.target_arm = self.bootstrap_arm


class UCBBandit:
    """UCB1 statistics for one timestep position."""

    def __init__(self, arms: Iterable[int]) -> None:
        self.arms = tuple(arms)
        self.counts = {arm: 0 for arm in self.arms}
        self.reward_sums = {arm: 0.0 for arm in self.arms}

    def select(self, available_arms: Sequence[int], gamma: float) -> int:
        available = tuple(arm for arm in available_arms if arm in self.counts)
        if not available:
            return 0

        untried = [arm for arm in available if self.counts[arm] == 0]
        if untried:
            # UCB1 gives every untried arm infinite priority. Choosing the
            # smallest one avoids an unnecessarily aggressive uncalibrated skip.
            return untried[0]

        total = max(1, sum(self.counts[arm] for arm in available))
        log_total = math.log(total)
        best_arm = available[0]
        best_score = -float("inf")

        for arm in available:
            count = self.counts[arm]
            mean_reward = self.reward_sums[arm] / count
            bonus = gamma * math.sqrt(log_total / count)
            score = mean_reward + bonus
            if score > best_score:
                best_score = score
                best_arm = arm

        return best_arm

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.reward_sums[arm] += float(reward)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "counts": {str(arm): self.counts[arm] for arm in self.arms},
            "reward_sums": {
                str(arm): self.reward_sums[arm] for arm in self.arms
            },
        }

    @classmethod
    def from_dict(cls, data: Dict, arms: Iterable[int]) -> "UCBBandit":
        bandit = cls(arms)
        counts = data.get("counts", {})
        reward_sums = data.get("reward_sums", {})
        for arm in bandit.arms:
            bandit.counts[arm] = int(counts.get(str(arm), 0))
            bandit.reward_sums[arm] = float(reward_sums.get(str(arm), 0.0))
        return bandit


class FastFlowController:
    """Paper-faithful UCB controller with a safe first-run warm start."""

    def __init__(self, config: FastFlowConfig, total_steps: int) -> None:
        self.config = config
        self.total_steps = total_steps
        self._ema_relative_error: Optional[float] = None
        self._load_stats()

    def _load_stats(self) -> None:
        path = self.config.stats_path
        if not path or not os.path.isfile(path):
            return

        with open(path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
        if data.get("format") != "wan21-fastflow-v1":
            logging.warning("Ignoring FastFlow stats with an unknown format: %s", path)
            return

        self.config.max_mse = float(data.get("max_mse", 0.0))
        saved_target = data.get("target_arm")
        if saved_target in self.config.arms:
            self.config.target_arm = int(saved_target)
        raw_bandits = data.get("bandits", {})
        for step_key, raw_bandit in raw_bandits.items():
            step = int(step_key)
            self.config.bandits[step] = UCBBandit.from_dict(
                raw_bandit, self.config.arms
            )
        logging.info(
            "Loaded FastFlow bandit stats for %d timesteps from %s.",
            len(self.config.bandits),
            path,
        )

    def save_stats(self) -> None:
        path = self.config.stats_path
        if not path:
            return

        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        data = {
            "format": "wan21-fastflow-v1",
            "arms": list(self.config.arms),
            "gamma": self.config.gamma,
            "max_mse": self.config.max_mse,
            "target_arm": self.config.target_arm,
            "bandits": {
                str(step): bandit.to_dict()
                for step, bandit in sorted(self.config.bandits.items())
            },
        }
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
        logging.info("Saved FastFlow bandit stats to %s.", path)

    def _nearest_arm(self, available_arms: Sequence[int]) -> int:
        target = self.config.target_arm
        return min(available_arms, key=lambda arm: (abs(arm - target), arm))

    def _bandit_for_step(self, step: int) -> UCBBandit:
        if step not in self.config.bandits:
            self.config.bandits[step] = UCBBandit(self.config.arms)
        return self.config.bandits[step]

    def select(self, step: int, available_arms: Sequence[int]) -> int:
        available = tuple(arm for arm in available_arms if arm in self.config.arms)
        if not available:
            return 0

        bandit = self._bandit_for_step(step)
        has_paper_statistics = any(bandit.counts[arm] > 0 for arm in available)
        if not has_paper_statistics:
            return self._nearest_arm(available)
        return bandit.select(available, self.config.gamma)

    def _update_target_arm(self, relative_error: float) -> None:
        if self._ema_relative_error is None:
            self._ema_relative_error = relative_error
        else:
            self._ema_relative_error = (
                0.8 * self._ema_relative_error + 0.2 * relative_error
            )

        arms = self.config.arms
        current = self.config.target_arm
        if self._ema_relative_error > self.config.error_high:
            smaller = [arm for arm in arms if arm < current]
            if smaller:
                self.config.target_arm = smaller[-1]
        elif self._ema_relative_error < self.config.error_low:
            larger = [arm for arm in arms if arm > current]
            if larger:
                self.config.target_arm = larger[0]

    def update(
        self,
        step: int,
        arm: int,
        predicted_velocity: Optional[torch.Tensor],
        true_velocity: torch.Tensor,
    ) -> None:
        if predicted_velocity is None:
            return

        with torch.no_grad():
            error = (
                (predicted_velocity.float() - true_velocity.float())
                .pow(2)
                .mean()
                .item()
            )
            relative_error = (
                (predicted_velocity.float() - true_velocity.float())
                .abs()
                .mean()
                / (true_velocity.float().abs().mean().clamp_min(1e-8))
            ).item()

        self.config.max_mse = max(self.config.max_mse, error)
        if self.config.mu is None:
            # With mu = max_mse / T, dividing the paper reward by max_mse
            # keeps UCB independent of the model's velocity scale.
            reward = arm / self.total_steps
            if self.config.max_mse > 0:
                reward -= error / self.config.max_mse
        else:
            reward = self.config.mu * arm - error

        self._bandit_for_step(step).update(arm, reward)
        self._update_target_arm(relative_error)

        if self.config.verbose:
            logging.info(
                "step=%03d arm=%d mse=%.6g rel_l1=%.4g reward=%.6g next_target=%d",
                step,
                arm,
                error,
                relative_error,
                reward,
                self.config.target_arm,
            )


def _available_arms(
    arms: Sequence[int], step: int, last_timestep: int
) -> Tuple[int, ...]:
    # A selected arm m performs a full evaluation at step + m + 1.
    max_arm = max(0, last_timestep - step - 1)
    available = tuple(arm for arm in arms if arm <= max_arm)
    return available or (0,)


def _predict_velocity(
    pipeline,
    latent: torch.Tensor,
    timestep: torch.Tensor,
    context: List[torch.Tensor],
    context_null: List[torch.Tensor],
    seq_len: int,
    guide_scale: float,
) -> torch.Tensor:
    """Run classifier-free guidance and return the Wan flow velocity."""

    pipeline.model.to(pipeline.device)
    latent_input = [latent]
    timestep_input = torch.stack([timestep]).to(pipeline.device)

    noise_pred_cond = pipeline.model(
        latent_input,
        t=timestep_input,
        context=context,
        seq_len=seq_len,
    )[0]
    noise_pred_uncond = pipeline.model(
        latent_input,
        t=timestep_input,
        context=context_null,
        seq_len=seq_len,
    )[0]
    return noise_pred_uncond + guide_scale * (
        noise_pred_cond - noise_pred_uncond
    )


def _euler_step(
    latent: torch.Tensor,
    velocity: torch.Tensor,
    sigma_from: torch.Tensor,
    sigma_to: torch.Tensor,
) -> torch.Tensor:
    return latent + (sigma_to - sigma_from) * velocity


def fastflow_t2v_generate(
    self,
    input_prompt,
    size=(832, 480),
    frame_num=81,
    shift=5.0,
    sampling_steps=50,
    guide_scale=5.0,
    n_prompt="",
    seed=-1,
    offload_model=True,
):
    """Wan2.1 T2V generation accelerated by FastFlow."""

    fastflow: FastFlowConfig = self._fastflow_config
    frame_count = frame_num
    target_shape = (
        self.vae.model.z_dim,
        (frame_count - 1) // self.vae_stride[0] + 1,
        size[1] // self.vae_stride[1],
        size[0] // self.vae_stride[2],
    )
    seq_len = (
        math.ceil(
            (target_shape[2] * target_shape[3])
            / (self.patch_size[1] * self.patch_size[2])
            * target_shape[1]
            / self.sp_size
        )
        * self.sp_size
    )

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device("cpu"))
        context_null = self.text_encoder([n_prompt], torch.device("cpu"))
        context = [item.to(self.device) for item in context]
        context_null = [item.to(self.device) for item in context_null]

    noise = torch.randn(
        target_shape[0],
        target_shape[1],
        target_shape[2],
        target_shape[3],
        dtype=torch.float32,
        device=self.device,
        generator=seed_g,
    )

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, "no_sync", noop_no_sync)

    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(
            sampling_steps, device=self.device, shift=shift
        )
        timesteps = scheduler.timesteps
        sigmas = scheduler.sigmas.to(self.device, dtype=torch.float32)
        last_timestep = len(timesteps) - 1
        if last_timestep < 1:
            raise ValueError("FastFlow requires at least two sampling steps.")

        controller = FastFlowController(fastflow, len(timesteps))
        controller.config.model_evaluations = 0
        controller.config.baseline_evaluations = len(timesteps) * 2
        latent = noise

        # FastFlow needs two consecutive real velocities before it can form
        # the finite-difference velocity slope.
        previous_velocity = _predict_velocity(
            self,
            latent,
            timesteps[0],
            context,
            context_null,
            seq_len,
            guide_scale,
        )
        controller.config.model_evaluations += 2

        latent = _euler_step(
            latent, previous_velocity, sigmas[0], sigmas[1]
        )
        current_step = 1
        previous_step = 0
        current_velocity = _predict_velocity(
            self,
            latent,
            timesteps[current_step],
            context,
            context_null,
            seq_len,
            guide_scale,
        )
        controller.config.model_evaluations += 2

        progress = tqdm(
            total=last_timestep,
            initial=current_step,
            desc="FastFlow",
            disable=self.rank != 0,
        )
        while current_step < last_timestep:
            arms = _available_arms(
                controller.config.arms, current_step, last_timestep
            )
            arm = controller.select(current_step, arms)
            terminal_step = current_step + arm + 1

            sigma_current = sigmas[current_step]
            sigma_terminal = sigmas[terminal_step]
            sigma_previous = sigmas[previous_step]
            dt = sigma_current - sigma_previous
            if abs(float(dt.item())) > 1e-12:
                velocity_slope = (current_velocity - previous_velocity) / dt
            else:
                velocity_slope = torch.zeros_like(current_velocity)

            if arm > 0:
                sigma_before_terminal = sigmas[terminal_step - 1]
                predicted_velocity = current_velocity + (
                    sigma_before_terminal - sigma_current
                ) * velocity_slope
                terminal_latent = latent + (
                    sigma_before_terminal - sigma_current
                ) * predicted_velocity

                predicted_terminal_velocity = current_velocity + (
                    sigma_terminal - sigma_current
                ) * velocity_slope
                terminal_latent = terminal_latent + (
                    sigma_terminal - sigma_before_terminal
                ) * predicted_terminal_velocity
            else:
                predicted_terminal_velocity = current_velocity
                terminal_latent = _euler_step(
                    latent, current_velocity, sigma_current, sigma_terminal
                )

            true_terminal_velocity = _predict_velocity(
                self,
                terminal_latent,
                timesteps[terminal_step],
                context,
                context_null,
                seq_len,
                guide_scale,
            )
            controller.config.model_evaluations += 2
            controller.update(
                current_step,
                arm,
                predicted_terminal_velocity,
                true_terminal_velocity,
            )

            latent = terminal_latent
            previous_velocity = current_velocity
            previous_step = current_step
            current_velocity = true_terminal_velocity
            current_step = terminal_step
            progress.update(arm + 1)

        # The final Euler update uses the last exact velocity prediction.
        latent = _euler_step(
            latent,
            current_velocity,
            sigmas[current_step],
            sigmas[len(timesteps)],
        )
        progress.update(1)
        progress.close()

        used = controller.config.model_evaluations
        baseline = controller.config.baseline_evaluations
        logging.info(
            "FastFlow DiT evaluations: %d (baseline %d, %.2fx fewer).",
            used,
            baseline,
            baseline / max(used, 1),
        )
        if fastflow.stats_path and self.rank == 0:
            controller.save_stats()

        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        if self.rank == 0:
            videos = self.vae.decode([latent])

    del noise, latent, scheduler, controller
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None


def install_fastflow(pipeline, config: FastFlowConfig) -> None:
    """Attach FastFlow to a WanT2V instance without touching model weights."""

    pipeline._fastflow_config = config
    pipeline.generate = types.MethodType(fastflow_t2v_generate, pipeline)


def _default_arms(sample_steps: int) -> Tuple[int, ...]:
    if sample_steps >= 50:
        return (0, 2, 4, 6)
    if sample_steps >= 25:
        return (0, 1, 2, 3)
    return (0, 1, 2)


def _parse_arms(value: str) -> Tuple[int, ...]:
    arms = tuple(sorted(set(int(item.strip()) for item in value.split(","))))
    if not arms:
        raise argparse.ArgumentTypeError("FastFlow arm set cannot be empty.")
    return arms


def _parse_args():
    parser = argparse.ArgumentParser(
        description="FastFlow-accelerated Wan2.1 T2V 1.3B generation."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-1.3B",
        choices=["t2v-1.3B"],
        help="Only the Wan2.1 T2V 1.3B model is targeted by this script.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="832*480",
        choices=list(SUPPORTED_SIZES["t2v-1.3B"]),
        help="Output area as width*height.",
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="Number of frames. It must be 4n+1.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to Wan2.1-T2V-1.3B checkpoints.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt.",
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help=(
            "Move models to CPU around generation to save GPU memory. "
            "Defaults to False for distributed runs and True otherwise."
        ),
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Keep the T5 encoder on CPU.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="Random seed. A random seed is chosen when negative.",
    )
    parser.add_argument(
        "--sample_steps", type=int, default=50, help="Sampling steps."
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=5.0,
        help="Flow-matching sigma shift.",
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="Output mp4 path.",
    )
    parser.add_argument(
        "--fastflow_arms",
        type=_parse_arms,
        default=None,
        help="Skip arms, e.g. 0,2,4,6. Defaults depend on sample_steps.",
    )
    parser.add_argument(
        "--fastflow_gamma",
        type=float,
        default=2.0,
        help="UCB exploration constant from the FastFlow paper.",
    )
    parser.add_argument(
        "--fastflow_mu",
        type=float,
        default=None,
        help=(
            "Absolute speed/error trade-off. By default FastFlow uses "
            "mu=max_mse/T as described in the paper."
        ),
    )
    parser.add_argument(
        "--fastflow_bootstrap_arm",
        type=int,
        default=None,
        help="First-run warm-start arm before bandit statistics exist.",
    )
    parser.add_argument(
        "--fastflow_error_low",
        type=float,
        default=0.03,
        help="Relative velocity error below which the warm start increases skips.",
    )
    parser.add_argument(
        "--fastflow_error_high",
        type=float,
        default=0.10,
        help="Relative velocity error above which the warm start reduces skips.",
    )
    parser.add_argument(
        "--fastflow_stats",
        type=str,
        default=None,
        help="Optional JSON file for persistent bandit statistics.",
    )
    parser.add_argument(
        "--fastflow_verbose",
        action="store_true",
        default=False,
        help="Log every FastFlow bandit decision.",
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="Ulysses sequence-parallel size.",
    )
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="Ring-attention parallel size.",
    )
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Enable FSDP for T5.",
    )
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Enable FSDP for the Wan DiT.",
    )

    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args) -> None:
    assert args.size in SUPPORTED_SIZES[args.task]
    assert args.frame_num >= 1 and (args.frame_num - 1) % 4 == 0
    assert args.sample_steps >= 2
    if args.fastflow_arms is None:
        args.fastflow_arms = _default_arms(args.sample_steps)
    if args.fastflow_bootstrap_arm is None:
        middle = max(0, len(args.fastflow_arms) // 2 - 1)
        args.fastflow_bootstrap_arm = args.fastflow_arms[middle]
    assert args.fastflow_bootstrap_arm in args.fastflow_arms
    args.base_seed = (
        args.base_seed
        if args.base_seed >= 0
        else random.randint(0, sys.maxsize)
    )


def _init_logging(rank: int) -> None:
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args) -> None:
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = world_size == 1
        logging.info("offload_model defaulted to %s.", args.offload_model)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), "FSDP requires a distributed environment."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), "Context parallel requires a distributed environment."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert (
            args.ulysses_size * args.ring_size == world_size
        ), "ulysses_size * ring_size must equal WORLD_SIZE."
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size()
        )
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT
    logging.info("Input prompt: %s", args.prompt)
    logging.info("Generation args: %s", args)

    pipeline = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )
    fastflow_config = FastFlowConfig(
        arms=args.fastflow_arms,
        gamma=args.fastflow_gamma,
        mu=args.fastflow_mu,
        bootstrap_arm=args.fastflow_bootstrap_arm,
        error_low=args.fastflow_error_low,
        error_high=args.fastflow_error_high,
        stats_path=args.fastflow_stats,
        verbose=args.fastflow_verbose,
    )
    install_fastflow(pipeline, fastflow_config)

    logging.info("Generating video with FastFlow ...")
    video = pipeline.generate(
        args.prompt,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
    )

    if rank == 0:
        if args.save_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_tag = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            args.save_file = (
                f"{args.task}_{args.size}_fastflow_{prompt_tag}_{timestamp}.mp4"
            )
        logging.info("Saving video to %s", args.save_file)
        cache_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
    logging.info("Finished.")


if __name__ == "__main__":
    generate(_parse_args())
