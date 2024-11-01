import wandb
YOUR_WANDB_USERNAME = "dan-amler-team"
project = "NLP2024_PROJECT_dan4151"

command = [
        "${ENVIRONMENT_VARIABLE}",
        "${interpreter}",
        "StrategyTransfer.py",
        "${project}",
        "${args}"
    ]

sweep_config = {
    "name": "LSTM: SimFactor=0/4 for any features representation",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        "ENV_HPT_mode": {"values": [False]},
        "architecture": {"values": ["LSTM"]},
        "seed": {"values": list(range(1, 4))},
        "online_simulation_factor": {"values": [4]},
        "features": {"values": ["EFs", "GPT4", "BERT"]},
        "theta": {"values": [0.02, 0.05, 0.07, 0.1]},
        "basic_nature": {"values": [0, 6, 12, 13]}

    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}") #changed this line to run on windows
