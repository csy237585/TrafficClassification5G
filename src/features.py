import pandas as pd
import numpy as np

def getBurstSeries(): 
    pcap_data = pd.read_csv('./../data/youtube_data/youtube1.csv')
    # Filter TCP flows
    tcp_flows = pcap_data[pcap_data['Protocol'] == 'TCP']

    # Group TCP flows by source and destination
    grouped_flows = tcp_flows.groupby(['Source', 'Destination'])

        # Function to calculate bytes per second (BPS), packets per second (PPS), and average packet length (PLEN)
    def calculate_features(flow):
        time_diff = flow['Time'].diff().fillna(0)  # Time difference between packets
        byte_diff = flow['Length'].diff().fillna(0)  # Byte difference between packets
        
        # Bytes per second (BPS)

        flow['BPS'] = byte_diff / time_diff 
        # Replace NaN and inf with zeros in BPS
        flow['BPS'].replace([np.nan, np.inf], 0, inplace=True)
        
        # Packets per second (PPS)
        flow['PPS'] = 1 / time_diff 

        # Replace NaN and inf with zeros in PPS
        flow['PPS'].replace([np.nan, np.inf], 0, inplace=True)
        
        # Average packet length (PLEN)
        flow['PLEN'] = byte_diff / flow['PPS']
        flow['PLEN'].replace([np.nan, np.inf], 0, inplace=True)
        
        return flow

    processed_flows = grouped_flows.apply(calculate_features)
    processed_flows = processed_flows.drop(columns=['No.', 'Source', 'Destination', 'Protocol', 'Info'])
    processed_flows.set_index('Time', inplace=True)
    # Group by 0.25-second intervals and calculate the mean
    aggregated_features = processed_flows.groupby(np.ceil(processed_flows.index / .250)).mean().fillna(0)
    
    # Threshold for burst detection
    # I = 0.5  # I is in seconds
    I = 2

    # Initialize variables to store burst series
    burst_series = []

    # Initialize variables to keep track of burst
    burst_start_index = None
    burst_sum = 0

    # Iterate through the time series data
    for i in range(1, len(aggregated_features)):
        # Calculate time difference between consecutive points
        # time_diff = (aggregated_features.index[i] - aggregated_features.index[i-1]) / 1000  # Convert to seconds
        time_diff = (aggregated_features.index[i] - aggregated_features.index[i-1])  # Convert to seconds
        
        # Check if time difference is less than threshold
        if time_diff < I:
            # If burst has not started yet, mark the start index
            if burst_start_index is None:
                burst_start_index = i - 1
            
            # Add the value of the point to burst sum
            burst_sum += aggregated_features.iloc[i]['Length']  # the value of each point in the time series
        
        else:
            # If burst was ongoing, add the burst sum to burst series
            if burst_start_index is not None:
                burst_series.append(burst_sum)
                # Reset burst variables
                burst_start_index = None
                burst_sum = 0

    # If burst was ongoing at the end of the time series, add the burst sum to burst series
    if burst_start_index is not None:
        burst_series.append(burst_sum)


    # Create a custom index based on 250-millisecond intervals
    custom_index = pd.timedelta_range(start=0, periods=len(burst_series), freq='250ms')
    # Create a Series with burst_series data and custom index
    burst_series_with_index = pd.Series(burst_series, index=custom_index)
    # Resample the Series into 250-millisecond intervals
    aggregated_bursts = burst_series_with_index.resample('250ms').sum().fillna(0)

    return aggregated_bursts