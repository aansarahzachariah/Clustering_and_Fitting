import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

# defining function

# read data

def read_data(data, country, years):
    ''' Function to read CSV files. In this assignment four CSV files or
    datasets are chosen. The values of the following files are called using
    this function. Arguments such as data, country and years are used where
    data is used to get the values of the CSV file, country and years are used
    to get the list of countries and years to be read from whole data. This
    function uses filtering technique to take the required values such as
    dropping and retrieving values using iloc[]. Once filtering is done, the
    dataframes are transposed'''
    df = pd.read_csv(data, skiprows=4)
    df.drop(columns=['Country Code'], axis=1, inplace=True)
    df1 = df.iloc[country, years]
    df2 = df1.T
    return df1, df2

#Function to plot


def plot(data, kind, title, x, y):
    ''' Function to create plots. This function is used to give an insight of
    the dataframes. Arguments such as data, kind, title, x, y are used. data
    arguement is used to get the dataframe, kind argument is to specify which
    graph to be plotted. title, x, and y are used to label the coordinates on
    the graph'''
    data.plot(kind=kind)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))
    plt.show()
    
# Load the climate change data from the World Bank dataset

dataset = 'Assignment file.csv'
df = pd.read_csv(dataset)

# Transpose the data
world_bank_data_t = df.transpose()
print(world_bank_data_t)

# Select relevant columns for analysis
columns = ['Country Name', 'Country Code', 'Indicator Name', '2015']
df = df[columns]

# Pivot the data to have indicators as columns
pivoted = df.pivot(index='Country Code', columns='Indicator Name', values='2015')

# Select specific indicators for analysis
indicators = ['Agricultural land (% of land area)', 'Arable land (% of land area)']
df_indicators = pivoted[indicators]

# Remove missing values
df_indicators = df_indicators.dropna()

# Normalize the data
normalized = (df_indicators - df_indicators.mean()) / df_indicators.std()

# Perform clustering using k-means
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(normalized)

# Define a function for curve fitting
def exponential_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Fit the curve to the data
x_data = normalized[indicators[0]].values
y_data = normalized[indicators[1]].values
popt, _ = curve_fit(exponential_func, x_data, y_data)

# Plot the cluster membership and cluster centers
plt.figure(figsize=(15, 9))
colors = ['r', 'g', 'b', 'c', 'y']
for i in range(n_clusters):
    cluster_points = normalized.iloc[clusters == i, :]
    plt.scatter(
        cluster_points[indicators[0]],
        cluster_points[indicators[1]],
        color=colors[i],
        label=f'Cluster {i+1}'
    )
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    color='k',
    marker='X',
    label='Cluster Centers'
)
plt.title('Clusters of Agricultural land and Areable Land in 2015')
plt.xlabel('Agricultural land (% of land area)')
plt.ylabel('Arable land (% of land area)')
plt.legend()

# Plot the fitted curve
plt.figure(figsize=(15, 9))
plt.scatter(
    normalized[indicators[0]],
    normalized[indicators[1]],
    color='g',
    label='Data Points'
)
plt.plot(
    x_data,
    exponential_func(x_data, *popt),
    color='r',
    label='Curve Fit'
)
plt.title('Curve Fitting of Agricultural land and Areable Land in 2015')
plt.xlabel('Agricultural land (% of land area)')
plt.ylabel('Arable land (% of land area)')
plt.legend()

def linear_model(x, a, b):
    return a*x + b

# Fit the linear model
popt, pcov = curve_fit(linear_model, x_data, y_data)

def err_ranges(popt, pcov, x):
    perr = np.sqrt(np.diag(pcov))
    y = linear_model(x, *popt)
    lower = linear_model(x, *(popt - perr))
    upper = linear_model(x, *(popt + perr))
    return y, lower, upper

# Predicting future values and corresponding confidence Ranges
x_future = np.array(range(0, 2020))
y_future, lower_future, upper_future = err_ranges(popt, pcov, x_future)

# Predicting future values and corresponding confidence Ranges
x_future = np.array(range(0, 2020))
y_future, lower_future, upper_future = err_ranges(popt, pcov, x_future)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(x_data, y_data, 'o', label='Data Points')
plt.plot(x_future, y_future, '-', label='Curve fit')
plt.fill_between(x_future, lower_future, upper_future, alpha=0.3, label='Confidence Range')
plt.xlabel('Agricultural land (% of land area)')
plt.ylabel('Arable land (% of land area')
plt.title('Curve Fitting of Agricultural land and Areable Land in 2015')
plt.legend()
plt.show()



# Show the plots
plt.show()