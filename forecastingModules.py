#########################################################################################################
# Forecasting Module to import the Narraben profile data, convert to shoreline positions, and then 
# visualize the different interpolation methods to determine which interpolation will provide the most
# beneficial data to forecast. 
#                              *****Created by Marc W. Brittain*****
#
#########################################################################################################
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
sns.set_context('notebook',font_scale=1.5)
sns.set_style('ticks')
#########################################################################################################

def grab_profileData(filename):
    return pd.read_csv(filename)

#########################################################################################################

def selectProfile(dataset,ID):  
    # convert the dataframe above to an array
    array = dataset.values.astype(str)
    # create 3 empty lists
    chain = []
    elevation = []
    date = []

    # in each line of the above dataframe, if the profile is not PF1, it is ignored
    for line in array:
        if not line[1].startswith(ID): continue

        # if it is PF1, append all the information to the 3 empty lists
        chain.append(float(line[3]))
        elevation.append(float(line[4]))
        date.append(line[2])

    # once the lists are complete we convert them to a dataframe to work with
    chain_df = pd.DataFrame(chain)
    elev_df = pd.DataFrame(elevation)
    # in this last line, we introduced pd.to_datetime(), this converts from a string to datetime objects
    date_df = pd.DataFrame(pd.to_datetime(date))

    # each time a dataframe is created from a list, we must rename the column
    # to be able to join them  together
    
    chain_df.rename(columns={0:'chainage'},inplace=True)
    elev_df.rename(columns={0:'elevation'},inplace=True)
    date_df.rename(columns={0:'Date'},inplace=True)

    # join the dataframes together
    main = chain_df.join(elev_df)
    main = main.join(date_df)
    
    return main, chain_df, elev_df, date_df

#########################################################################################################

def shorelinePositions(df,chain_df, elev_df, date_df,ID):
    
    # convert the above dataframe to an array
    array = df.values.astype(str)

    # initialize an empty dataframe
    new_x = pd.DataFrame()

    # empty list
    dates = []

    #initialize variables to 0
    count = 0
    value = 0

    # loop through each string in the array
    for i in array[:]:
        count += 1    
        if i[1].startswith('-') and value >= 0:
            df_y = elev_df[count-3: count+1]
            df_x = chain_df[count-3:count+1]
            df_x = df_x.sort_values("chainage")

            x_array = df_x.values.astype(float)
            y_array = df_y.values.astype(float)

            x_array = x_array.reshape(-1,)
            y_array = y_array.reshape(-1,)

            x = x_array.tolist()
            y = y_array.tolist()

            f = interpolate.UnivariateSpline(x_array, y_array, s=0)
            x_new = np.arange(1,150,1)

            #plt.plot(x,y,'x',x_new,f(x_new))   

            # y value to find is 0, that is the shoreline position

            yToFind = 0        
            yreduced = np.array(y) - yToFind        
            f_reduced = interpolate.UnivariateSpline(x, yreduced, s=0)
            df = pd.DataFrame(f_reduced.roots())       
            dates.append(i[2])

            if new_x.empty:
                new_x = df                
            else:
                new_x = new_x.append(df)
        value = float(i[1])

    new_x.rename(columns = {0:"Chainage"}, inplace=True)
    new_x = new_x.reset_index()
    date_df = pd.DataFrame(pd.to_datetime(dates))
    date_df.rename(columns={0:'Dates'}, inplace=True)
    
    x = new_x.values
    shoreline = []
    for i in x:
        if not i[0] == 0: continue
        shoreline.append(i[1])
    shoreline_df = pd.DataFrame(shoreline)
    shoreline_df.rename(columns={0:'shoreline'},inplace=True)

    shoreline_df = shoreline_df.join(date_df)
    shoreline_df = shoreline_df[shoreline_df.shoreline != 0.07803480405574043]
    plots = shoreline_df.set_index('Dates')
    
    plt.figure()
    plots.plot(figsize=(10,7))
    plt.title('Shoreline Positions '+ID)
    plt.ylabel('Chainage (m)')
    shoreline_df = shoreline_df.set_index('Dates')
    return shoreline_df

#########################################################################################################

def optimize(df, resample_method):
    greatest_so_far = 0
    integer = 0
    for i in range(1,resample_method+1):
        resample = df.resample(str(i)+'D').mean()
        length = len(resample)
        actual = len(resample.dropna())
        percent = float(actual)/float(length)
        
        if percent > greatest_so_far:
            greatest_so_far = percent
            integer = i
    greatest = greatest_so_far*100.0
    interp_percent = 100.0-greatest
    print 'Precent of actual values:         '+str(greatest)
    print 'Percent of interpolated values:   '+str(interp_percent)
    print 'Number of days to resample by:    '+str(integer)
    print 'Number of data entries:           '+str(len(resample))


def interpolationMethod(df, resample_method):
    print ('                           Types of Interpolation')
    print ('')
    print ('Linear')
    print ('Cubic')
    print ('nearest')
    print ('pchip')
    print ('quadratic')
    print ('slinear')
    print ('polynomial')
    print ('')
    method = raw_input('Enter Method of Interpolation: ')
    print ('')
    print ('')
    print ('')
    if method == 'all':
        #########################################################################################
        linearResample = df.resample(resample_method).mean()
        linearInterp = linearResample.interpolate()
        print ('Linear Interpolation: ')
        print ('     ignore the index and treat the values as equally spaced.')
        print ('')
        print ('')
        plotShoreline(df,linearInterp,'Linear',resample_method)
        
        
        #########################################################################################
        cubicResample = df.resample(resample_method).mean()
        cubicInterp = cubicResample.interpolate(method='cubic')
        print ('Cubic Interpolation: ')
        print ('     Interpolate data with a piecewise cubic polynomial which is twice continuously differentiable')
        print ('')
        print ('')
        plotShoreline(df,cubicInterp,'Cubic',resample_method)
        
        #########################################################################################
        
        nearestResample = df.resample(resample_method).mean()
        nearestInterp = nearestResample.interpolate(method='nearest')
        print ('nearest Interpolation: ')
        print ('     Nearest-neighbour interpolation.')
        print ('')
        print ('')
        plotShoreline(df,nearestInterp,'nearest',resample_method)
        
        #########################################################################################
        
        pchipResample = df.resample(resample_method).mean()
        pchipInterp = pchipResample.interpolate(method='pchip')
        print ('pchip Interpolation: ')
        print ('     x and y are arrays of values used to approximate some function f, with y = f(x). The interpolant uses')
        print ('     monotonic cubic splines to find the value of new points. (PCHIP stands for Piecewise Cubic Hermite')
        print ('     Interpolating Polynomial).')
        print ('')
        print ('')
        plotShoreline(df,pchipInterp,'pchip',resample_method)
        
#########################################################################################################

def plotShoreline(shoreline,interp_df,interpMethod,resample_method):
    interp_df.rename(columns={'shoreline':interpMethod+' Interpolation'},inplace=True)
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    interp_df.plot(title=interpMethod+' Interpolation',ax=axs[0], colormap='summer')
    plt.xlabel('Dates')
    plt.ylabel('Chainage (m)')
    shoreline.plot(ax=axs[1])
    plt.title('Shoreline Positions')
    plt.xlabel('Dates')
    plt.ylabel('Chainage (m)')