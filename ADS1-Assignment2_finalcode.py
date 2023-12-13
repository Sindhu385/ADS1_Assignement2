

# importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis



# Function to read and preprocess data
def assign_data(filename, indicator_name):
    '''Creating the define function to read a raw data, selecting the 
countries, dropping unnecessary columns and Transposing the filtered data'''
    dataframe = pd.read_csv(filename, skiprows=3)
    country = ['China', 'Indonesia', 'India',
               'Japan', 'United States', 'United Kingdom']
    selection = dataframe[(dataframe['Indicator Name'] == indicator_name) & (
        dataframe['Country Name'].isin(country))]
    df = selection.drop(dataframe.columns[1:4], axis=1)
    df = df.drop(dataframe.columns[-1:], axis=1)
    ''' dropping the years from 1960 to 2001 for having NaN values'''
    years = [str(year) for year in range(1960, 2001)] + [
        str(year) for year in range(2021, 2023)]
    data = df.drop(columns=years)
    data = data.reset_index(drop=True)
    
    data_t = data.transpose()
    data_t.columns = data_t.iloc[0]
    data_t = data_t.iloc[1:]
    data_t.index = pd.to_numeric(data_t.index)
    data_t['Years'] = data_t.index
    return data, data_t

# Function to slice dataframe for correlation
def slicing(df):
    '''Slicing the dataframe of year 2020 to create a new dataframe for
correlation'''
    df = df[['Country Name', '2020']]
    return df

#Function to merge sliced data
def merge(df1, df2, df3, df4, df5):
    '''Merging the sliced data to create a merged dataframe'''
    mer1 = pd.merge(df1, df2, on='Country Name', how='outer')
    mer2 = pd.merge(mer1, df3, on='Country Name', how='outer')
    mer3 = pd.merge(mer2, df4, on='Country Name', how='outer')
    mer4 = pd.merge(mer3, df5, on='Country Name', how='outer')
    mer4 = mer4.reset_index(drop=True)
    return mer4

#Function to create a heatmap
def heatmap_fig(df):
    '''    Generate a heatmap visualization of the correlation matrix for a 
    given DataFrame, by fixing the size of the plot, title.
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing numerical columns.

'''
    plt.figure(figsize = (7, 4))
    sns.heatmap(df.corr(), cmap ='viridis', square = True,
                linewidths = .5, annot = True, fmt = ".2f", center = 0)
    plt.title("Correlation matrix of Indicators")
    plt.show()

#Function to create line plot
def lineplot_fig(df, y_label, title):
    '''Generate a line plot for selected countries over the years.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing data for countries 
    over the years.
    - y_label (str): Label for the y-axis.
    - title (str): Title of the plot.
'''
    sns.set_style("whitegrid")
    df.plot(x = 'Years', y = ['China', 'Indonesia', 'India', 'Japan',
                          'United States', 'United Kingdom'], xlabel = 'Years',
            ylabel = y_label, marker = '.')
    plt.title(title)
    plt.xlabel('Years')
    plt.ylabel(y_label)
    plt.xticks(range(2000, 2021, 2))
    plt.legend(loc = 'best', bbox_to_anchor = (1, 0.4))
    plt.show()

#Function to create box plot
def boxplot_fig(df, countries, shownotches=True):
    '''Generate a box plot to visualize population growth for selected 
    countries.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing population data for 
    countries.
    - countries (list): List of countries to be included in the box plot.
    - shownotches (bool): Whether to show notches in the box plot. Default is
    True.
'''
    plt.figure(figsize = (10,5))
    plt.boxplot([df[country] for country in countries])
    plt.title('Population Growth in box plot')
    plt.xticks(range(1, len(countries) + 1), countries) 
    plt.savefig('boxplot.png')
    plt.show()

#Function to create a bar plot
def barplot_fig(df, x_value, y_value, head_title, x_label, y_label,
                colors, figsize = (6, 4)):
    '''Generate a bar plot to visualize specific data points over the years.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing data for the specified
    years.
    - x_value (str): The column name for the x-axis values.
    - y_value (str): The column name for the y-axis values.
    - head_title (str): Title of the plot.
    - x_label (str): Label for the x-axis.
    - y_label (str): Label for the y-axis.
    - colors (list): List of colors for the bars.
    - figsize (tuple): Size of the figure. Default is (6, 4).
'''
    sns.set_style('whitegrid')
    bar_df = df[df['Years'].isin([2001, 2005, 2010, 2015, 2020])]
    bar_df.plot(x = x_value, y = y_value, kind = 'bar', title = head_title,
                color = colors,width = 0.65, figsize = figsize,
                xlabel = x_label, ylabel = y_label)
    plt.legend(loc = 'best', bbox_to_anchor = (1, 0.4))
    plt.savefig('barplot.png')
    plt.show()

#Function to create a pie plot
def pieplot_fig(df, Years, title, autopct='%1.0f%%', fontsize = 11):
    '''Generate a pie chart to represent the distribution of data for specific
    years.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing data for the specified
    years.
    - Years (int): The year for which the pie chart is generated.
    - title (str): Title of the pie chart.
    - autopct (str): Format string for percentage labels. Default is '%1.0f%%'.
    - fontsize (int): Font size for labels. Default is 11.
'''
    explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    label = ['China', 'Indonesia', 'India',
             'Japan', 'United States', 'United Kingdom']
    plt.figure(figsize = (5, 6))
    plt.pie(df[str(Years)], autopct = autopct, labels = label,
            explode = explode, startangle=180,
            wedgeprops = {"edgecolor": "black", "linewidth": 2,
                                        "antialiased": True},)
    plt.title(title)
    plt.savefig('pieplot.png')
    plt.show()

#Function to create a dot plot
def dotplot_fig(df, title, y_label):
    '''Generate a dot plot to visualize data for different countries over the
    years.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing data for countries and
    years.
    - title (str): Title of the dot plot.
    - y_label (str): Label for the y-axis.
'''
    sns.set_style("whitegrid")
    dot = sns.catplot(x = 'Years', y = 'Value', hue = 'Country', 
        data = df.melt(id_vars = ['Years'], 
                       var_name='Country', value_name = 'Value'), kind="point")
    dot.set_xticklabels(rotation=90)
    plt.title(title)
    plt.ylabel(y_label)
    plt.savefig('dotplot.png')
    plt.show()

#Function to calculate skewness and kurtosis
def skew_kurt(df):
    '''Calculate skewness, kurtosis and mean of a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
'''
    #calculating skew value
    df_s = df.skew()
    #calculating kurtosis value
    df_k = df.kurtosis()
    #calculating mean value
    df_m = np.mean(df)
    return df_s, df_k, df_m

# Main programme to analyze and visualize data


filename = r"C:\Users\sindh\OneDrive\Desktop\API_19_DS2_en_csv_v2_5998250.csv"
gre, gre_t = assign_data(
    filename, 'Total greenhouse gas emissions (kt of CO2 equivalent)')

popul, popul_t = assign_data(filename, 'Population growth (annual %)')

meth, meth_t = assign_data(
    filename, 'Methane emissions (kt of CO2 equivalent)')

fore, fore_t = assign_data(filename, 'Forest area (% of land area)')

co2, co2_t = assign_data(filename, 'CO2 emissions (kt)')

a_cor = slicing(gre).rename(columns={'2020': 'Total greenhouse gas emissions'})
b_cor = slicing(popul).rename(columns = {'2020': 'Population growth'})
c_cor = slicing(meth).rename(columns = {'2020': 'Methane emissions'})
d_cor = slicing(fore).rename(columns = {'2020': 'Forest area'})
e_cor = slicing(co2).rename(columns = {'2020': 'CO2 emissions'})

d1_d2_d3_d4_d5 = merge(a_cor, b_cor, c_cor, d_cor, e_cor)

heatmap_fig(d1_d2_d3_d4_d5)
lineplot_fig(gre_t, 'kt of co2 equivalent',
             'Total greenhouse gas emission in line plot')

boxplot_fig(popul_t, ['China', 'Indonesia', 'India', 'Japan',
            'United States', 'United Kingdom'], shownotches=True)


barplot_fig(meth_t, 'Years', ['China', 'Indonesia', 'India', 'Japan',
                              'United States', 'United Kingdom'],
            'Methane emissions in Barplot', 'Years', 'kilotons', (
                'brown', 'pink', 'grey', 'purple', 'yellow', 'red'))


pieplot_fig(fore, 2005, 'Forest area in 2000 for Pie plot')


dotplot_fig(co2_t, 'co2 emissions in dot plot', 'kilotonns')
print(d1_d2_d3_d4_d5.describe())

skewness, kurtosis, mean = skew_kurt(gre_t['India'])


#printing the value of skewness
print(f"Skewness: {skewness}")

#printing the value of kurtosis 
print(f"Kurtosis: {kurtosis}")

#printing the value of mean
print(f"Mean: {mean}")
