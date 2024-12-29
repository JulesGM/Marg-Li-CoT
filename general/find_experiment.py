import pathlib
import rich
import edit_distance
import inquirer
import more_itertools as mit
# import diskcache
import pathlib

# We decrement slowly. Should probably be date based too.
CACHE_INCREMENT_VALUE = 3
CACHE_DECREMENT_VALUE = 1
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def list_experiments(experiment_folder: str | pathlib.Path) -> set[str]:
    """
    Create a list of experiments from the experiment folder,
    recreate the experiment name from the file path.
    """

    experiment_folder = pathlib.Path(experiment_folder)
    assert experiment_folder.exists(), experiment_folder
    list_experiments_files = experiment_folder.glob("**/*.yaml")

    return set(
        str(x.relative_to(experiment_folder).parent / x.stem) 
        for x in list_experiments_files
    )


def inquire_experiment(experiments) -> str:
    """
    Show a list of experiments to the user.
    """
    experiment = inquirer.list_input(
        message="Which experiment do you want to run?",
        choices=experiments,
    )
    return experiment


def check_experiment_and_suggest(experiment: str, folder: str | pathlib.Path) -> None:
    """
    If the user attempts to run an experiment that doesn't exist,
    sort the available experiments by the edit distance between the

    If the user doesn't attempt to run an experiment, sort the available
    experiments by the number of times the experiment was run.
    """
    assert isinstance(experiment, (str, type(None))), experiment
    experiments = sorted(list_experiments(folder))

    # cache = diskcache.Cache(SCRIPT_DIR / ".experiment_cache")
    # counts = cache.get(str(folder))
    cache = None
    counts = None
    if counts is None:
        counts = {}
        
    if not experiment in experiments: 
        if experiment:
            # Don't print the message if the experiment is None or "".
            # The user already knows that the experiment is not found.
            rich.print(f"Experiment [bold]{experiment}[/] not found.")

        # If there isn't an attempt, we can't compute the edit distance.
        if experiment and edit_distance:
            # If there is an attempted spelling by the user, sort by
            # the edit distance between the experiment choices and the user's input.
            exps_with_distances = [
                (possible_experiment, edit_distance.edit_distance(possible_experiment, experiment))
                for possible_experiment in experiments
            ]
            
            exps_with_distances.sort(key=lambda x: x[1])
            sorted_exps, _ = mit.zip_equal(*exps_with_distances)
            experiment = inquire_experiment(sorted_exps)

        else:
            # If no attempted spelling by the user, sort by 
            # the number of times the experiment was run.

            present_in_counts = [x for x in experiments if x in counts]
            not_present_in_counts = [x for x in experiments if x not in counts]
            present_in_counts.sort(key=lambda x: counts[x], reverse=True)

            if present_in_counts:
                experiments = present_in_counts + not_present_in_counts
            experiment = inquire_experiment(experiments)
            
            # Update the count of the experiment.

    counts[experiment] = counts.get(experiment, 0) + CACHE_INCREMENT_VALUE
    # Decrement the count of the other experiments.
    # If you don't pick something for a while, 
    # it becomes alphabetically sorted again.
    for exp, value in counts.copy().items(): # We copy because we modify the dictionary during the iteration.
        if exp != experiment:
            counts[exp] = max(value - CACHE_DECREMENT_VALUE, 0)
        
        # We put <= in case we accidentally
        # decremented it before.
        if counts[exp] <= 0:
            del counts[exp]

    # cache.set(str(folder), counts)

    return experiment