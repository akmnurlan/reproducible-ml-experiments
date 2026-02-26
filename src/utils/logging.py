import csv
from datetime import datetime

def log_run(config, metrics):
    with open("results/runs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            config["model"],
            metrics["accuracy"],
            config["seed"]
        ])
