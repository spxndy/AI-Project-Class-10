import pandas as pd
from pathlib import Path
from exploration.checks import checks
from exploration.plots import plot

from modelling.model import generate_model, test_model

# CSV filepath
DATA_PATH = Path("data", "StressLevelDataset.csv")

# Uploading dataframe
df = pd.read_csv(DATA_PATH)

checks(df)

for i in range(3):
    plot(df, i, True, show=False)

# Modelling
model, X_test, y_test = generate_model(df)

metrics_df = test_model(model, X_test, y_test)
print(metrics_df)
