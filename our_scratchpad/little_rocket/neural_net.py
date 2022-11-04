import json

import rich
import torch
import torch.nn


def init_mlp(layer):
    if isinstance(layer, torch.nn.Linear):
        rich.print(f"[bold green]Initializing:[/] {layer.__class__.__name__}")
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
    else:
        rich.print(f"[bold red]Not initializing:[/] {layer.__class__.__name__}")


def build_mlp(
    *, 
    input_size:  int, 
    num_layers:  int, 
    hidden_size: int, 
    output_size: int,
):
    layers = [
        torch.nn.Linear(
            input_size, 
            hidden_size if num_layers > 1 
            else output_size
        ),
    ]

    for i in range(1, num_layers):
        is_last = i == num_layers - 1
        layers.extend([
            # torch.nn.LayerNorm(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(
                hidden_size, 
                hidden_size if not is_last 
                else output_size, 
            ),
        ])

    print(layers)
    
    return torch.nn.Sequential(*layers)


def save_models(actor, value, config, path):
    rich.print("[bold red]Saving to ", path / "actor.pth")
    torch.save(actor.state_dict(), path / "actor.pth")

    if value:
        torch.save(value.state_dict(), path / "critic.pth")
    
    (path / "config.json").write_text(json.dumps(config, indent=4))


def load_model(path):
    config = json.loads((path / "config.json").read_text())
    actor = torch.load(path / "actor.pth")
    
    if config["has_value_function"]:
        value_model = torch.load(path / "critic.pth")
    else:
        value_model = None

    return actor, value_model, config
