
import pandas as pd
import numpy as np

np.random.seed(42)

data_size = 1000

data = {
    "code_changes": np.random.randint(1, 100, data_size),
    "files_modified": np.random.randint(1, 30, data_size),
    "errors": np.random.randint(0, 15, data_size),
    "warnings": np.random.randint(0, 25, data_size),
    "build_duration": np.random.randint(10, 300, data_size),
    "previous_failures": np.random.randint(0, 10, data_size)
}

df = pd.DataFrame(data)

# Generate simulated build outcomes
failure_score = (
    df["errors"] * 5
    + df["warnings"] * 0.5
    + df["code_changes"] * 0.08
    + df["files_modified"] * 0.3
    + df["previous_failures"] * 2
)

df["build_status"] = (failure_score > 35).astype(int)

df.to_csv("dataset/build_data.csv", index=False)

print("Dataset generated successfully!")
print(df.head())
