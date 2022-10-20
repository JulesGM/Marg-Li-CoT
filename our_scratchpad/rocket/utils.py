import json
from pathlib import Path
import os
import time
from typing import *

import IPython
from IPython import display as jupyter_display
import more_itertools
import numpy as np
import rich
from typing import *
import torch

def unzip_dict(d):
    keys, values = zip(*d.items())
    for values_line in zip(*values):
        yield dict(zip(keys, values_line))

def we_are_in_jupyter():
    jupyter_object = IPython.get_ipython()
    if jupyter_object is not None:
        shell = jupyter_object.__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    else:
        return False      # Probably standard Python interpreter

if not we_are_in_jupyter():
    import matplotlib
    matplotlib.use('Agg')

import gym
import matplotlib.pyplot as plt


def we_are_on_slurm():
    return any(x.startswith("SLURM_") for x in os.environ)


def reset_env_maybe_show(
    *, 
    env,
    do_show,
    sim_name
):
    if do_show:
        env = gym.make(sim_name, render_mode="human")
        observation, _ = env.reset()
        env.render()
    else:
        env = gym.make(sim_name)
        observation, _ = env.reset()
    return observation, env


class DemoTimer:
    """
    
    """
    def __init__(self, show_n_rollouts=5, time_period_in_seconds=60 * 2, disable=False):
        self._time_period_in_seconds = time_period_in_seconds
        self._show_n_steps           = show_n_rollouts
        self._last_shown             = None
        self._disable                = disable
        
        self._started                = False
        self._counter                = 0

    def _start_or_continue(self):
        if not self._started:
            self._started = True
            self._counter = 1
        else:
            self._counter += 1

    def step(self) -> bool:
        """
        State possibilities:
            1. - We are not enabled
            2. - We are enabled:
                2.1 - We have started
                    2.1.1 - the demo should end
                    2.1.2 - the demo should continue
                2.2 - We have not started
                    2.2.1 - we have never shown the demo
                        set "last_shown" to now
                    2.2.2 - we have shown the demo before
                        2.2.2.1 - The period not passed
                        2.2.2.2 - The period has passed

        Cases:
            1. The timer is disabled.
            2. The timer is enabled, 
        """

        # State 1 - We are not enabled
        if self._disable:
            return False 

        # State 2 - We are enabled
        # State 2.1 - We have started
        if self._started:
            demo_should_end = self._counter >= self._show_n_steps
            
            # State 2.1.1 - the demo should end
            if demo_should_end:
                self._started = False
                period_ellapsed = (time.perf_counter() - self._last_shown) > self._time_period_in_seconds
                assert not period_ellapsed, (
                    "The demo should end but the period has ellapsed already."
                    "This is a bug, we would be in an infinite demo with just "
                    "one step between each demo."
                )
                return False

            # State 2.1.2 - the demo should continue
            else:
                self._start_or_continue()
                return True

        # State 2.2 - We have not started
        else:
            # State 2.2.1 - Never shown
            never_shown = self._last_shown is None
            if never_shown:
                self._last_shown = time.perf_counter()
                return False
            
            # State 2.2.2 - We have shown the demo before
            else:
                period_ellapsed = (time.perf_counter() - self._last_shown) > self._time_period_in_seconds

                # State 2.2.2.1 - The period not passed
                if not period_ellapsed:
                    return False

                # State 2.2.2.2 - The period has passed
                else:
                    self._start_or_continue()
                    self._last_shown = time.perf_counter()
                    return True
            
        assert False


class LivePlots:
    def __init__(
        self,
        main_call_arguments,
        titles:  List[str],
        grand_title: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5), 
    ):

        self._main_call_arguments = main_call_arguments
        self._titles = titles
        self._subplot_ids: dict[str, int] = {}
        self._subplot_axes: dict[str, plt.Axes] = {}
        self._grand_title = grand_title
        self._figsize = figsize
        
        self._already_done = False

        self._reset_figure()

    def _reset_figure(self):
        plt.figure(figsize=self._figsize)
        self._fig = plt.gcf()
        if self._grand_title:
            self._fig.suptitle(self._grand_title)
        

    # def live_plots(
    #     self,
    #     dict_data_dict: Dict[str, Dict[str, Any]], 
    # ):
        
    #     for i, (data_dict_k, data_dict_v) in enumerate(dict_data_dict.items()):
    #         self._subplot_ids[data_dict_k] = self._subplot_ids.get(data_dict_k, i + 1)

    #         if data_dict_k not in self._subplot_axes:
    #             if data_dict_k not in self._subplot_axes:
    #                 print(f"Initializing subplot of `{data_dict_k}`")
    #                 ax = plt.subplot(
    #                     1, len(dict_data_dict), self._subplot_ids[data_dict_k]
    #                 )
    #                 self._subplot_axes[data_dict_k] = ax.get_figure(), ax
            

    #         subfig, ax = self._subplot_axes[data_dict_k]
            

    #         if ax.lines:
    #             print("Redrawing lines")
    #             for line, (label, data) in more_itertools.zip_equal(ax.lines, data_dict_v.items()):
    #                 print(f"Redrawing lines -> {label} - {len(data)}")
    #                 line.set_xdata(np.arange(len(data)))
    #                 line.set_ydata(data)
    #                 # line.set_label(label)
                
    #         else:
    #             print("Intializing lines")
    #             for label, data in data_dict_v.items():
    #                 print(f"Intializing lines -> {label} - {len(data)}")
    #                 ax.plot(data, label=label)

                
    #         subfig.canvas.draw()
    #     self._fig.canvas.draw()
        
    #     plt.show()
    def live_plots(
            self,
            dict_data_dict: Dict[str, Dict[str, Any]], 
        ):
            if matplotlib.get_backend().lower() != "agg":
                # clear jupyter output
                IPython.display.clear_output(wait=True)
                rich.print(self._main_call_arguments)

                self._reset_figure()
                for i, (data_dict_k, data_dict_v) in enumerate(dict_data_dict.items()):
                    self._subplot_ids[data_dict_k] = self._subplot_ids.get(data_dict_k, i + 1)
                    ax = plt.subplot(1, len(dict_data_dict), self._subplot_ids[data_dict_k])
                    for label, data in data_dict_v.items():
                        ax.plot(data, label=label)
                plt.show()
            else:
                import plotext as plt
                
                for i, (data_dict_k, data_dict_v) in enumerate(dict_data_dict.items()):
                    for label, data in data_dict_v.items():
                        plt.plot(data, label=label)
                        plt.show()
