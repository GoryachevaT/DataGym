import pandas as pd
import numpy as mp
import re
from datetime import datetime

def hv(val):
    if re.findall('( and more)', str(val)):
        val = 10.0
    try:
        return float(val)
    except:
        return -1
    
    
def weather_prep(df):
    
    df['short_time'] = df['local_time'].apply(lambda x: x[:6]+x[8:])
    dt = [datetime.strptime(x, "%d/%m/%y %H:%M") for x in [x.replace('.', '/') for x in df['short_time'].values]]
    df['dt'] = dt
    df['hour_dt'] = df['dt'].dt.round('60min')
    
    df['wind_s']        = df['mean_wind_direction'].apply(lambda x: 1 if x=='Wind blowing from the south' else 0)
    df['wind_sse']      = df['mean_wind_direction'].apply(lambda x: 1 if x=='Wind blowing from the south-southeast' else 0)
    df['wind_nw']       = df['mean_wind_direction'].apply(lambda x: 1 if x=='Wind blowing from the north-west' else 0)
    df['wind_ssw']      = df['mean_wind_direction'].apply(lambda x: 1 if x=='Wind blowing from the south-southwest' else 0)
    df['wind_variable'] = df['mean_wind_direction'].apply(lambda x: 1 if x=='variable wind direction' else 0)
    df['wind_n']        = df['mean_wind_direction'].apply(lambda x: 1 if x=='Wind blowing from the north' else 0)
    df['wind_nnw']      = df['mean_wind_direction'].apply(lambda x: 1 if x=='Wind blowing from the north-northwest' else 0)
    df['wind_sw']       = df['mean_wind_direction'].apply(lambda x: 1 if x=='Wind blowing from the south-west' else 0)
    df['wind_nne']      = df['mean_wind_direction'].apply(lambda x: 1 if x=='Wind blowing from the north-northeast' else 0)
    df['wind_se']       = df['mean_wind_direction'].apply(lambda x: 1 if x=='Wind blowing from the south-east' else 0)
    df['wind_wnw']      = df['mean_wind_direction'].apply(lambda x: 1 if x=='Wind blowing from the west-northwest' else 0)
    df['wind_wsw']      = df['mean_wind_direction'].apply(lambda x: 1 if x=='Wind blowing from the west-southwest' else 0)
    df['wind_ne']       = df['mean_wind_direction'].apply(lambda x: 1 if x=='Wind blowing from the north-east' else 0)
    df['wind_w']        = df['mean_wind_direction'].apply(lambda x: 1 if x=='Wind blowing from the west' else 0)
    df['wind_ese']      = df['mean_wind_direction'].apply(lambda x: 1 if x=='Wind blowing from the east-southeast' else 0)
    df['wind_e']        = df['mean_wind_direction'].apply(lambda x: 1 if x=='Wind blowing from the east' else 0)
    df['wind_ene']      = df['mean_wind_direction'].apply(lambda x: 1 if x=='east-northeast' else 0)
    
    df['clouds_few'] = df['clouds'].str.lower().apply(lambda x: 1 if re.findall('(few clouds)', str(x)) else 0)
    df['clouds_broken'] = df['clouds'].str.lower().apply(lambda x: 1 if re.findall('(broken clouds)', str(x)) else 0)
    df['clouds_scattered'] = df['clouds'].str.lower().apply(lambda x: 1 if re.findall('(scattered clouds)', str(x)) else 0)
    df['about_visibility'] = df['clouds'].str.lower().apply(lambda x: 1 if re.findall('(visibility)', str(x)) else 0)
    
    
    df['horizontal_visibility'] = df['horizontal_visibility'].apply(hv)
    
    df['weather_rain']    = df['special_weather_phenomena'].str.lower().apply(lambda x: 1 if re.findall('(rain)', str(x)) else 0)
    df['weather_fog']     = df['special_weather_phenomena'].str.lower().apply(lambda x: 1 if re.findall('(fog)', str(x)) or re.findall('(mist)', str(x)) else 0)
    df['weather_drizzle'] = df['special_weather_phenomena'].str.lower().apply(lambda x: 1 if re.findall('(drizzle)', str(x)) else 0)
    
    df['Td'] = df['Td'].apply(hv)
    df['Td'] = df['Td'].astype(float)
    
    df['month_jan'] = df.hour_dt.apply(lambda x: 1 if x.month==1 else 0)
    df['month_feb'] = df.hour_dt.apply(lambda x: 1 if x.month==2 else 0)
    df['month_mar'] = df.hour_dt.apply(lambda x: 1 if x.month==3 else 0)
    df['month_apr'] = df.hour_dt.apply(lambda x: 1 if x.month==4 else 0)
    df['month_may'] = df.hour_dt.apply(lambda x: 1 if x.month==5 else 0)
    df['month_jun'] = df.hour_dt.apply(lambda x: 1 if x.month==6 else 0)
    df['month_jul'] = df.hour_dt.apply(lambda x: 1 if x.month==7 else 0)
    df['month_aug'] = df.hour_dt.apply(lambda x: 1 if x.month==8 else 0)
    df['month_sep'] = df.hour_dt.apply(lambda x: 1 if x.month==9 else 0)
    df['month_oct'] = df.hour_dt.apply(lambda x: 1 if x.month==10 else 0)
    df['month_nov'] = df.hour_dt.apply(lambda x: 1 if x.month==11 else 0)
    
    df['morning'] = df.hour_dt.apply(lambda x: 1 if x.hour >=0 and x.hour < 7 else 0)
    df['day']     = df.hour_dt.apply(lambda x: 1 if x.hour >=7 and x.hour < 12 else 0)
    df['evening'] = df.hour_dt.apply(lambda x: 1 if x.hour >=12 and x.hour < 19 else 0)
    
    df.drop(['local_time', 'short_time', 'dt', 'mean_wind_direction', 'clouds', 'special_weather_phenomena'], axis=1, inplace=True)
    
    df_2 = df.groupby('hour_dt').max().reset_index()
    
    return df_2