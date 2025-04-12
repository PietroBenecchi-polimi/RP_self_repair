from main import run_oversampling_pipeline
import matplotlib.pyplot as plt
import json
import os

points_to_test = [150, 300, 500, 650, 800, 900]
stats_per_points = []

# Path to save the results
save_path = "oversampling_results.json"

# Load previous results if they exist
if os.path.exists(save_path):
    with open(save_path, "r") as f:
        stats_per_points = json.load(f)
    existing_points = {int(list(d.keys())[0]) for d in stats_per_points}
else:
    existing_points = set()

# Run and save results incrementally
for points in points_to_test:
    if points in existing_points:
        print(f"Skipping already processed point count: {points}")
        continue
    stats = run_oversampling_pipeline(200, 150, True, points)
    stats_per_points.append({points: stats})
    
    # Save progress after each point
    with open(save_path, "w") as f:
        json.dump(stats_per_points, f, indent=2)

# Prepare data for plotting
methods = set()
for stats_dict in stats_per_points:
    for point, stats in stats_dict.items():
        for stat in stats:
            methods.add(stat['method'])

method_performance = {method: [] for method in methods}

for stats_dict in stats_per_points:
    for point, stats in stats_dict.items():
        for stat in stats:
            method_performance[stat['method']].append((int(point), stat['success_percentage']))

# Plotting
plt.figure(figsize=(12, 6))
for method, values in method_performance.items():
    values.sort()
    x = [v[0] for v in values]
    y = [v[1] for v in values]
    plt.plot(x, y, marker='o', label=method)

plt.title("Method Performance vs Number of Training Points")
plt.xlabel("Number of Points Used to Train Regressor")
plt.ylabel("Success Percentage")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("oversampling_results")