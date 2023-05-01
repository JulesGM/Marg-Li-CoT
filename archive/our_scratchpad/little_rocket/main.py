#!/usr/bin/env python
# coding: utf-8
import matplotlib
matplotlib.use("TkAgg")

import enum

import itertools
import os
from pathlib import Path
import re
import time
from typing import *

import fire


import gym
import numpy as np
import rich
import torch
from tqdm import tqdm

import constants
import neural_net
import rl
import utils

SCRIPT_DIR = Path(__file__).absolute().parent


APPROACH_TYPE = constants.ApproachType.PPO
SARSA_TYPE = rl.SARSA.SarsaTypes.EXPECTATION

IS_DOUBLE_DQN = False
DOUBLE_DQN_TARGET_UPDATE_PERIOD = 1
DOUBLE_DQN_SOFT_UPDATE_RATIO = 0.02

###############################################################################
# Hyperparameters
###############################################################################
CLIP_NORM: Optional[float] = None

LR = 1E-5
LAMBDA = 0.97
GAMMA = 0.99
TAU = 0.001
SEED = 42
H = 64
L = 3

EPS_DECAY = 0.9995
EPS_MIN = 0.05
EPS_MAX = 1

STEP_SIZE = 1


QTY_MEMORY_SAMPLES = 64
REPLAY_BUFFER_SIZE = 50000

EPOCHS = 1000000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIM_NAME = "LunarLander-v2"

DEFAULT_RENDER_EVERY_N_EPOCHS = 100
DEFAULT_RENDER_N_ROLLOUTS = 10

SAVE_EVERY_N_EPOCH = 100
PLOT_EVERY_N_EPOCHS = 20


def make_approach_inst_predict(approach_name, config, policy, value, device):
    gamma = config.get("gamma", None)
    lambda_ = config.get("lambda_", None)
    tau = config.get("tau", None)

    if approach_name == constants.ApproachType.SARSA:
        return rl.SARSA(
            q=policy,
            device=device,
            gamma=gamma,
            tau=tau,
            sarsa_type=None,
            action_space_size=config["output_size"],
        )
    elif approach_name == constants.ApproachType.DQN:
        return rl.DQN(
            q=policy,
            device=device,
            gamma=None,
            tau=None, 
            replay_buffer=None,
            eps_min=0, 
            eps_max=0, 
            eps_decay=0,
            action_space_size=config["output_size"],
            double_dqn=None,
            double_dqn_target_update_period=None,
            double_dqn_soft_update_ratio=None,
        )
        
    elif approach_name == constants.ApproachType.ACTOR_CRITIC:
        return rl.AC(
            policy=policy,
            value=value,
            device=device,
            lambda_=lambda_,
            gamma=gamma,
        )
    elif approach_name == constants.ApproachType.PPO:
        assert False
        return rl.AC(
            policy=policy,
            value=value,
            device=device,
            lambda_=lambda_,
            gamma=gamma,
        )
    else:
        raise ValueError(f"Unknown approach: {approach_name}")


def main(
    device=DEVICE,
    sim_name=SIM_NAME,

    epochs=EPOCHS,    
    qty_memory_samples=QTY_MEMORY_SAMPLES,

    step_size=STEP_SIZE,

    learning_rate=LR,
    hidden_size=H,
    num_layers=L,

    tau=TAU,
    lambda_=LAMBDA,
    gamma=GAMMA,

    do_plot=utils.we_are_in_jupyter(), 
    num_rollouts_demo=5,
    demo_every_n_seconds=60,
    disable_demo=utils.we_are_on_slurm(),

    approach_name=APPROACH_TYPE,

    ppo_batch_size=128,
    steps_between_updates=32,
    ppo_buffer_max_size=int(1e7),

    seed=SEED,
    save_dir_root=SCRIPT_DIR / Path("saves")
):
    main_call_arguments = locals()
    rich.print(main_call_arguments)

    if do_plot:
        rich.print("[bold]Plotting is enabled.[/bold]")
    else:
        rich.print("[bold]Plotting is disabled.[/bold]")

    env = gym.make(sim_name)
    env.action_space.seed(seed)

    assert isinstance(env.action_space, gym.spaces.Discrete), type(env.action_space)
    assert isinstance(env.observation_space, gym.spaces.Box), type(env.observation_space)
    assert len(env.observation_space.shape) == 1, env.observation_space.shape

    policy_model = neural_net.build_mlp(
        input_size=env.observation_space.shape[0], 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        output_size=env.action_space.n,
    ).to(device)
    
    policy_model.apply(neural_net.init_mlp)


    if approach_name in constants.HAVE_VALUE_FUNCTION:
        approach_has_value_function = True
    elif approach_name in constants.DONT_HAVE_VALUE_FUNCTION:
        approach_has_value_function = False
    else:
        raise ValueError(approach_name)

    if approach_has_value_function:
        value_model = neural_net.build_mlp(
            input_size=env.observation_space.shape[0], 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            output_size=1,
        ).to(device)
        value_model.apply(neural_net.init_mlp)


    parameters = list(itertools.chain(policy_model.parameters(), value_model.parameters()) 
        if approach_has_value_function else policy_model.parameters())
    optimizer = torch.optim.Adam(
        parameters,
        lr=learning_rate,
    )

    optimizer.zero_grad()

    all_rewards = []
    all_losses = []

    demo_timer = utils.DemoTimer(
        show_n_rollouts=num_rollouts_demo, 
        time_period_in_seconds=demo_every_n_seconds, 
        disable=disable_demo,
    )

    if approach_name == constants.ApproachType.ACTOR_CRITIC:
        method = rl.AC(
            policy=policy_model, 
            value=value_model, 
            device=device,
            lambda_=lambda_,
            gamma=gamma,
        )
    elif approach_name == constants.ApproachType.PPO:
        method = rl.PPO(
            policy=policy_model, 
            value=value_model, 
            buffer_max_size=ppo_buffer_max_size,
            device=device,
            dtype=torch.float32,
            lambda_=lambda_,
            gamma=gamma,
            act_dim=env.action_space.n, 
            obs_dim=env.observation_space.shape[0],
        )
    elif approach_name == constants.ApproachType.SARSA:
        method = rl.SARSA(
            q=policy_model,
            device=device,
            gamma=gamma,
            tau=tau,
            sarsa_type=SARSA_TYPE,
            action_space_size=env.action_space.n,
        )
    elif approach_name == constants.ApproachType.DQN:
        method = rl.DQN(
            q=policy_model,
            device=device,
            gamma=gamma,
            tau=tau,
            
            eps_min=EPS_MIN,
            eps_max=EPS_MAX,
            eps_decay=EPS_DECAY,
            
            action_space_size=env.action_space.n,
            observation_space_shape=env.observation_space.shape,

            double_dqn=IS_DOUBLE_DQN,
            double_dqn_target_update_period=DOUBLE_DQN_TARGET_UPDATE_PERIOD,
            double_dqn_soft_update_ratio=DOUBLE_DQN_SOFT_UPDATE_RATIO,

            replay_buffer=rl.ReplayBuffer(
                size=REPLAY_BUFFER_SIZE,
                output_device=device, 
                action_shape=(env.action_space.n,),
                state_shape=env.observation_space.shape, 
                qty_memory_samples=qty_memory_samples,
            )
        )
    else:
        raise ValueError(approach_name)

    # ticker = tqdm()
    live_plotter = utils.LivePlots(
        main_call_arguments=main_call_arguments,
        titles = ["Reward", "Loss"],
        grand_title = f"Approach: {approach_name}",
    )
    for epoch in tqdm(range(epochs)):
        rollouts = []
        total_rewards = []
        observation, env = utils.reset_env_maybe_show(
            sim_name=sim_name,
            do_show=False,
        )
        terminated = False
        truncated = False

        start_batch = time.perf_counter()
        
        for rollout_idx in range(steps_between_updates):
            # rich.print(f"[bold]Rollout[/] {rollout_idx} / {steps_between_updates}")
            assert not terminated and not truncated

            # ticker.update(1)
            current_rollout = []
            
            next_action = None
            next_logits = None

            while True:
                if next_action is None:
                    action, logits = method(observation)
                else:
                    action, logits = next_action, next_logits

                accumulated_reward = 0.

                ################################################################################
                # Execute the action `step_size` times.
                ################################################################################
                assert step_size == 1, step_size
                (next_observation, reward, is_done, info) = env.step(action)
                
                # Necessary for regular SARSA
                if not is_done:
                    next_action, next_logits = method(next_observation)

                accumulated_reward += reward

                current_rollout.append(dict(
                    observation=observation, 
                    next_observation=next_observation,
                    action=action, 
                    reward=reward,
                    done=is_done,
                    logits=logits,
                    next_action=next_action,
                    next_logits=next_logits,
                ))

                assert isinstance(method, rl.PPO), type(method)

                if isinstance(method, rl.PPO):
                    method.add(
                        observation=observation, 
                        action=action, 
                        reward=reward,
                        done=is_done,
                        logp=logits.log_softmax(dim=-1)[action],
                    )

                ################################################################################
                # End of Step
                ################################################################################
                if is_done:
                    ################################################################################
                    # End of Rollout
                    ################################################################################
                    next_action = None
                    next_logits = None

                    if isinstance(method, rl.PPO):
                        method.buffer_finish_path()
                    
                    rollouts.append(current_rollout)
                    total_rewards += [sum(x["reward"] for x in current_rollout)]
                    do_show = demo_timer.step()
                    observation, env = utils.reset_env_maybe_show(
                        sim_name = sim_name,
                        do_show  = do_show,
                    )
                    terminated = False
                    truncated = False
                    break
                else:
                    observation = next_observation
            # rich.print(f"Finished rollouts: {time.perf_counter() - start_batch:.2}s")

        ################################################################################
        # Optimization Stuff
        ################################################################################
        
        if isinstance(method, rl.PPO):
            for i, minibatch in enumerate(method.get_minibatches(ppo_batch_size)):
                logprobs = method.policy(minibatch["obs"]).log_softmax(dim=-1)

                per_action_new_log_softmax = logprobs.gather(
                    index=minibatch["act"].long().unsqueeze(-1), 
                    dim=-1,
                )
                
                new_values = method.value(minibatch["obs"])
                policy_loss, value_loss = method.loss(
                    rollouts=minibatch,
                    new_logprobs=per_action_new_log_softmax,
                    new_values=new_values,
                )
                
                all_losses.append(value_loss.item())
                (policy_loss + value_loss).backward()

                optimizer.step()
                optimizer.zero_grad()
            rich.print("Finished optimization")
        else:
            assert False
            if approach_has_value_function:
                policy_loss, value_loss = method.update(rollouts)
                all_losses.append(value_loss.item())
                (policy_loss + value_loss).backward()
            else:
                loss = method.update(rollouts)
                loss.backward()
                all_losses.append(loss.item())

            if CLIP_NORM:
                torch.nn.utils.clip_grad_norm_(parameters, CLIP_NORM)

            optimizer.step()
            optimizer.zero_grad()

        ################################################################################
        # Stats and Plotting
        ################################################################################
        all_rewards.append(np.mean(total_rewards))

        if not do_plot:
            rich.print(f"[bold]Total rewards:[/] {np.mean(total_rewards)}")
            rich.print(f"[bold]Value Loss:[/]    {all_losses[-1]}")

        if do_plot and epoch % PLOT_EVERY_N_EPOCHS == 0:
            live_plotter.live_plots(
                dict_data_dict=dict(
                    rewards = dict(all_rewards=np.convolve(all_rewards, np.ones(10) / 10., mode="valid")),
                    loss    = dict(value_loss=all_losses),
                ), 
            )

        if epoch % SAVE_EVERY_N_EPOCH == 0:
            dir_target = Path(save_dir_root) / str(epoch)
            dir_target.mkdir()
            
            neural_net.save_models(
                policy_model, 
                value_model if approach_has_value_function else None, 
                dict(
                    steps_between_updates = steps_between_updates, 
                    learning_rate         = learning_rate, 
                    hidden_size           = hidden_size,
                    output_size           = env.action_space.n,
                    input_size            = env.observation_space.shape[0],
                    step_size             = step_size,
                    num_layers            = num_layers, 
                    sim_name              = sim_name,
                    method                = approach_name,
                    has_value_function    = approach_has_value_function,
                    tau                   = tau,
                ), 
                dir_target,
            )

    env.close()


def latest_epoch(directory):
    directory = Path(directory)

    try:
        next(directory.glob("*"))
    except StopIteration:
        return None

    return max(directory.glob("*"), key=lambda x: [int(x) for x in re.findall(r"\d+", str(x))])


def predict(
    load_path: Path=latest_epoch("saves"),
    device=DEVICE,
):
    rich.print(locals())

    rich.print(f"[bold]Matplotlib backend:[/] {matplotlib.get_backend()}")

    policy_model_dict, value_model_dict, config = neural_net.load_model(load_path)

    policy_model = neural_net.build_mlp(
        hidden_size = config["hidden_size"], 
        output_size = config["output_size"],
        input_size  = config["input_size"], 
        num_layers  = config["num_layers"], 
    ).to(device)
    policy_model.load_state_dict(policy_model_dict)

    if config["has_value_function"]:
        value_model  = neural_net.build_mlp(
            hidden_size = config["hidden_size"], 
            input_size  = config["input_size"], 
            num_layers  = config["num_layers"], 
            output_size = 1
        ).to(device)
        value_model .load_state_dict(value_model_dict)
    else:
        value_model = None

    method = make_approach_inst_predict(
        approach_name=config["method"], 
        policy=policy_model, 
        value=value_model, 
        device=device,
        config=config, 
    )

    env = gym.make(config["sim_name"], render_mode="human")
    while True:
        observation = env.reset()
        is_done = False

        total_reward = 0.
        while True:
            action, _ = method(observation)

            for i in range(config["step_size"]):
                observation, reward, is_done, info = env.step(action)
                env.render()
                total_reward += reward
                if is_done:
                    rich.print(f"[bold]Total reward:[/] {total_reward}")
                    break

            if is_done:
                observation = env.reset()


ENTRYPOINTS = dict(
    main=main,
    predict=predict,
)

if __name__ == "__main__":
    fire.Fire(ENTRYPOINTS)





