import pstats
import pandas as pd
import matplotlib.pyplot as plt

# Load the stats
stats = pstats.Stats('getfr_output.prof')

# Convert stats to a pandas DataFrame
df = pd.DataFrame(stats.stats.items())

# Process the DataFrame to get the data we're interested in
df['name'] = df[0].apply(lambda x: x[2])
df['ncalls'] = df[1].apply(lambda x: x[0])
df['tottime'] = df[1].apply(lambda x: x[2])

# Filter to the top 10 functions by total time
df = df.nlargest(10, 'tottime')

# Plot the data
plt.barh(df['name'], df['tottime'], color='blue')
plt.xlabel('Total Time')
plt.title('Top 10 Functions by Total Time')
plt.gca().invert_yaxis()
plt.show()