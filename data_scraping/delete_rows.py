import pandas as pd

# Read the CSV file
df = pd.read_csv('misc.csv')

# Specify the column names and values to delete
columns_values_to_delete = {
    'Rk': ['Rk'],
    'Player': ['Player'],
    'Nation': ['Nation'],
    'Pos': ['Pos'],
    'Squad': ['Squad'],
    'Comp': ['Comp'],
    'Age': ['Age'],
    'Born': ['Born'],
    '90s': ['90s'],
    'CrdY': ['CrdY'],
    'CrdR': ['CrdR'],
    '2CrdY': ['2CrdY'],
    'Fls': ['Fls'],
    'Fld': ['Fld'],
    'Off': ['Off'],
    'Crs': ['Crs'],
    'Int': ['Int'],
    'TklW': ['TklW'],
    'PKwon': ['PKwon'],
    'PKcon': ['PKcon'],
    'OG': ['OG'],
    'Recov': ['Recov'],
    'Won': ['Won'],
    'Lost': ['Lost'],
    'Won%': ['Won%'],
    'Matches': ['Matches']
}

# Delete the specified values within each column
for column, values in columns_values_to_delete.items():
    df[column] = df[column].replace(values, '')

# Save the modified data back to the CSV file
df.to_csv('misc', index=False)

import pandas as pd

# Read the CSV file
df = pd.read_csv('misc.csv')

# Specify the column names and values to delete
columns_values_to_delete = {
'Rk': ['Rk'],
    'Player': ['Player'],
    'Nation': ['Nation'],
    'Pos': ['Pos'],
    'Squad': ['Squad'],
    'Comp': ['Comp'],
    'Age': ['Age'],
    'Born': ['Born'],
    '90s': ['90s'],
    'CrdY': ['CrdY'],
    'CrdR': ['CrdR'],
    '2CrdY': ['2CrdY'],
    'Fls': ['Fls'],
    'Fld': ['Fld'],
    'Off': ['Off'],
    'Crs': ['Crs'],
    'Int': ['Int'],
    'TklW': ['TklW'],
    'PKwon': ['PKwon'],
    'PKcon': ['PKcon'],
    'OG': ['OG'],
    'Recov': ['Recov'],
    'Won': ['Won'],
    'Lost': ['Lost'],
    'Won%': ['Won%'],
    'Matches': ['Matches']    
}

# Delete the specified values within each column
for column, values in columns_values_to_delete.items():
    df[column] = df[column].replace(values, '')

# Delete rows with empty values in the specified columns
columns_to_check = list(columns_values_to_delete.keys())
df = df.dropna(subset=columns_to_check, how='all')

# Reset the row index after deleting rows
df = df.reset_index(drop=True)

# Save the modified data back to the CSV file
df.to_csv('misc.csv', index=False)
