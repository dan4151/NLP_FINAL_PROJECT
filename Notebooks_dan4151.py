import pandas as pd
import wandb
import matplotlib.pyplot as plt
import itertools

# Function to fetch sweep results
def get_sweep_results(api, sweep_id, username, project_name):
    try:
        sweep = api.sweep(f"{username}/{project_name}/{sweep_id}")
        runs = sweep.runs
        return runs
    except wandb.errors.CommError as e:
        print(f"Error: {e}")
        return None

# Initialize wandb API
api = wandb.Api()

# Project details
project_name = 'NLP2024_PROJECT_b206100224'  # Replace with your project name
username = 'dan-amler-team'  # Replace with your wandb username
sweep_id = '4v93g65v'  # Replace with your sweep ID

# Fetch sweep results
runs = get_sweep_results(api, sweep_id, username, project_name)
if runs is None:
    print("Sweep not found. Please check the sweep ID and try again.")
    exit()

# Extract data from runs
runs_data = []
for run in runs:
    summary = run.summary._json_dict
    config = {k: v for k, v in run.config.items() if not k.startswith('_')}
    name = run.name
    run_id = run.id
    history = run.history(keys=[], pandas=True)
    for i, row in history.iterrows():
        row_dict = row.to_dict()
        row_dict.update(config)
        row_dict.update(summary)
        row_dict['run_id'] = run_id
        row_dict['run_name'] = name
        runs_data.append(row_dict)

# Convert list of dictionaries to DataFrame
df = pd.DataFrame(runs_data)

# Print DataFrame to debug
print("DataFrame head:\n", df.head())

# Save the DataFrame to a CSV file
df.to_csv('sweep_results_all_runs_second2.csv', index=False)
print("Sweep results saved to 'sweep_results_all_runs.csv'")



# Load the CSV file
file_path_new = '/mnt/data/sweep_results_all_runs_second2.csv'
sweep_results_new_df = pd.read_csv(file_path_new)

# List of columns to retain
columns_to_retain = ['run_name', 'online_simulation_factor', 'basic_nature', 'theta', 'features', 'seed'] + \
                    [col for col in sweep_results_new_df.columns if
                     col.startswith('ENV_Test_accuracy_per_mean_user_and_bot_epoch')]

# Create a new dataframe with only the specified columns
filtered_sweep_results_new_df = sweep_results_new_df[columns_to_retain]

# Save the filtered dataframe to a new CSV file
filtered_csv_path = '/mnt/data/filtered_sweep_results.csv'
filtered_sweep_results_new_df.to_csv(filtered_csv_path, index=False)

# Define colors to match the first graphs
colors = itertools.cycle(['b', 'g', 'r', 'c'])

# Plot for online_simulation_factor = 4
online_simulation_factor = 4
fig, ax = plt.subplots(figsize=(10, 6))

for theta in filtered_sweep_results_new_df['theta'].unique():
    subset_df = filtered_sweep_results_new_df[
        (filtered_sweep_results_new_df['online_simulation_factor'] == online_simulation_factor) &
        (filtered_sweep_results_new_df['theta'] == theta)
        ]
    if not subset_df.empty:
        epoch_columns = [col for col in subset_df.columns if 'ENV_Test_accuracy_per_mean_user_and_bot_epoch' in col]
        epochs = sorted([int(col.split('epoch')[1]) for col in epoch_columns])
        accuracies = [subset_df[f'ENV_Test_accuracy_per_mean_user_and_bot_epoch{epoch}'].mean() for epoch in epochs]
        ax.plot(epochs, accuracies, label=f'Theta = {theta}', color=next(colors))

ax.set_title('Accuracy over all samples, online_simulation_factor - 4')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.legend()
ax.grid(True)

# Plot for online_simulation_factor = 0
online_simulation_factor = 0
fig, ax = plt.subplots(figsize=(10, 6))

for theta in filtered_sweep_results_new_df['theta'].unique():
    subset_df = filtered_sweep_results_new_df[
        (filtered_sweep_results_new_df['online_simulation_factor'] == online_simulation_factor) &
        (filtered_sweep_results_new_df['theta'] == theta)
        ]
    if not subset_df.empty:
        epoch_columns = [col for col in subset_df.columns if 'ENV_Test_accuracy_per_mean_user_and_bot_epoch' in col]
        epochs = sorted([int(col.split('epoch')[1]) for col in epoch_columns])
        accuracies = [subset_df[f'ENV_Test_accuracy_per_mean_user_and_bot_epoch{epoch}'].mean() for epoch in epochs]
        ax.plot(epochs, accuracies, label=f'Theta = {theta}', color=next(colors))

ax.set_title('Accuracy over all samples, online_simulation_factor - 0')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.legend()
ax.grid(True)

plt.show()


