"""
This is a helper script that visualizes river stations on map
"""

import plotly.graph_objects as go

import pandas as pd
import numpy as np

if __name__ == '__main__':

    # reading the data
    df = pd.read_csv('river_stations.csv')

    # fixing the column with next stations numbers, that are usually integers, if not missing.
    # we will fill missing ones by -1

    df.Nextdownstreamstation = df.Nextdownstreamstation.replace('-', -1)
    df.Nextdownstreamstation = df.Nextdownstreamstation.astype(np.int64)

    # creating the interactive plot
    fig = go.Figure()

    # plotting the lines for connected stations:
    for i in range(len(df)):
        if df[df['GRDC-No.'] == df['Nextdownstreamstation'][i]].shape[0] == 0: continue
        fig.add_trace(
            go.Scattergeo(
                locationmode='ISO-3',
                lon=[df['Longitude(DD)'][i],
                     df[df['GRDC-No.'] == df['Nextdownstreamstation'][i]]['Longitude(DD)'].values[0]],
                lat=[df['Latitude(DD)'][i],
                     df[df['GRDC-No.'] == df['Nextdownstreamstation'][i]]['Latitude(DD)'].values[0]],
                mode='lines',
                line=dict(width=1, color='blue'),
                hoverinfo='none'
            )
        )

    # adding the labels for stations:
    fig.add_trace(
        go.Scattergeo(
            lon=df['Longitude(DD)'],
            lat=df['Latitude(DD)'],
            text=df['Station']+', #'+df['GRDC-No.'].astype(str),
            marker=dict(
                color="green"
            ),
            mode='text',
        )
    )

    # setting the title:
    fig.update_layout(
        title='River Stations',
        geo_scope='europe',
    )

    # the figure is ready. The map is interactive. Please, adjust the zoom and the center of the map. Enjoy!
    fig.show()
