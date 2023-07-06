import pandas as pd

# Read the CSV file
df = pd.read_csv('misc.csv')

# List of columns to perform the calculation
columns_to_calculate = ['CrdY','CrdR','2CrdY','Fls','Fld','Off','Crs','Int','TklW','PKwon','PKcon','OG','Recov','Won','Lost']

# Perform the calculation and replace the original columns with the calculated values rounded to 2 decimal places
for column in columns_to_calculate:
    df[column] = round((df[column] / df['90s']) , 2)

# Save the modified data back to the CSV file
df.to_csv('misc.csv', index=False)
