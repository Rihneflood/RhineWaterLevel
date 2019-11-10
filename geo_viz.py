import plotly.graph_objects as go

import pandas as pd
import numpy as np

if __name__ == '__main__':

    df = pd.read_csv('river_stations.csv')

        df.Nextdownstreamstation = df.Nextdownstreamstation.replace('-', -1)
        df.Nextdownstreamstation = df.Nextdownstreamstation.astype(np.int64)

    print(df.info())

    fig = go.Figure()

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
    fig.add_trace(
        go.Scattergeo(
            lon=df['Longitude(DD)'],
            lat=df['Latitude(DD)'],
            text=df['GRDC-No.'],
            marker=dict(
                color="green"
            ),
            mode='text',
        )
    )

    fig.update_layout(
        title='River Stations',
        geo_scope='europe',
    )

    fig.show()
