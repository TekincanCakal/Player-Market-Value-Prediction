
import pandas as pd
try:
    df = pd.read_json('players_data.json')
    print("Columns:", df.columns.tolist())
except Exception as e:
    print(e)
