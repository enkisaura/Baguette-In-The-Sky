"""
Functions to plot positions.
"""

__authors__ = ("Enki SAURA")
__contact__ = ("esaura@ikosconsulting.com")
__copyright__ = "IKOS"
__date__ = "12/02/2025"
__version__ = "0.0.1"


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import folium
from folium.plugins import FastMarkerCluster
import requests

from bits.src.const import RE as earth_radius


def plot3d(pd_gnss:pd.DataFrame, sv_pos_ecef_cols: tuple=("x_sv_m", "y_sv_m", "z_sv_m"),
           sv_name_col: str | None = "sv_id",
           rx_pos_ecef_cols: tuple | None =("x_rx_m", "y_rx_m", "z_rx_m"), plot_all_timestamps: bool=False):
    """
    3D plot of GNSS satellites and receiver in ECEF coordinates.

    :param pd_gnss: GNSS dataframe.
    :param sv_pos_ecef_cols: Columns containing satellite ECEF coordinates.
    :param sv_name_col: Satellite name column.
    :param rx_pos_ecef_cols: Receiver ECEF coordinate columns.
    :param plot_all_timestamps: If False, keeps only first occurrence of each satellite.
    :return: fig, ax
    """
    pd_gnss = pd_gnss.copy()

    # Keep only one occurrence of each satellite
    if not plot_all_timestamps and sv_name_col is not None:
        pd_gnss = pd_gnss.drop_duplicates(subset=[sv_name_col], keep="first")

    # Create figure
    fig = plt.figure("3D plot of visible satellites")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])
    # Remove grid and panels
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_axis_off()
    # Find maximum altitude
    pd_gnss["range_m"] = np.sqrt(pd_gnss[sv_pos_ecef_cols[0]]**2
                                 + pd_gnss[sv_pos_ecef_cols[1]]**2
                                 + pd_gnss[sv_pos_ecef_cols[2]]**2)
    lim = pd_gnss["range_m"].max() + 100
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    # Plot orientation
    scale = 1e7
    ax.quiver(0, 0, 0, scale, 0, 0, color="r")
    ax.text(scale, 0, 0, "X")
    ax.quiver(0, 0, 0, 0, scale, 0, color="g")
    ax.text(0, scale, 0, "Y")
    ax.quiver(0, 0, 0, 0, 0, scale, color="b")
    ax.text(0, 0, scale, "Z")

    # Plot earth
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = earth_radius * np.outer(np.cos(u), np.sin(v))
    y = earth_radius * np.outer(np.sin(u), np.sin(v))
    z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="lightblue", alpha=0.5, linewidth=0)

    # Plot satellite vehicles
    ax.scatter(pd_gnss[sv_pos_ecef_cols[0]], pd_gnss[sv_pos_ecef_cols[1]], pd_gnss[sv_pos_ecef_cols[2]], color='r', s=1)

    # Satellite labels
    if sv_name_col is not None:
        pd_gnss = pd_gnss.drop_duplicates(subset=[sv_name_col], keep="last") # Only keep one occurence of each SV
        for _, row in pd_gnss.iterrows():
            ax.text(row[sv_pos_ecef_cols[0]], row[sv_pos_ecef_cols[1]], row[sv_pos_ecef_cols[2]],
                    str(row[sv_name_col]),
                    size=8,)

    # Plot receiver
    if rx_pos_ecef_cols is not None:
        ax.scatter(pd_gnss[rx_pos_ecef_cols[0]].iloc[0], pd_gnss[rx_pos_ecef_cols[1]].iloc[0],
                   pd_gnss[rx_pos_ecef_cols[2]].iloc[0], color='r', s=1)


    return fig, ax


def plot(pd_gnss_pvt: pd.DataFrame, m:folium.map=None, plot_name: str = "Plot", plot_rail=True) -> folium.map:
    """
    Plots locations on an open street map from a Dataframe. Dataframe must include columns "lat" and "lon".
    :param pd_gnss_pvt: Dataframe containing columns "lat" and "lon".
    :param m: folium map
    :param plot_name: Name of the dataset
    :return: folium map
    """
    # Check columns
    if not {'lat', 'lon'}.issubset(pd_gnss_pvt.columns):
        raise ValueError("DataFrame must contain 'lat' and 'lon' columns.")

    locations = pd_gnss_pvt[['lat', 'lon']].dropna().values.tolist()

    if m is None:
        # Create an empty map
        center = locations[0] if locations else [0, 0]
        m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap", control_scale=True)

    if len(locations) < 10000: # Use heavy plotting method with small number of points
        for lat, lon in locations:
            folium.Marker(location=(lat, lon), popup=plot_name).add_to(m)
    else: # Use light plotting method
        FastMarkerCluster(locations).add_to(m)

    # Add railway lines using Overpass API
    if locations and plot_rail:
        # Define bounding box based on GNSS data
        lats = pd_gnss_pvt['lat']
        lons = pd_gnss_pvt['lon']
        bbox = f"{lats.min() - 0.01},{lons.min() - 0.01},{lats.max() + 0.01},{lons.max() + 0.01}"

        query = f"""
        [out:json][timeout:25];
        (
          way["railway"="rail"]({bbox});
        );
        out body;
        >;
        out skel qt;
        """
        url = "https://overpass-api.de/api/interpreter"
        response = requests.post(url, data=query)
        if response.status_code == 200:
            data = response.json()

            # Extract node locations
            nodes = {el['id']: (el['lat'], el['lon']) for el in data['elements'] if el['type'] == 'node'}

            # Plot railway ways
            for el in data['elements']:
                if el['type'] == 'way':
                    latlon = [nodes[node_id] for node_id in el['nodes'] if node_id in nodes]
                    folium.PolyLine(latlon, color='black', weight=2.5, opacity=0.8, tooltip="Railway").add_to(m)
        else:
            print("Failed to load railway data from Overpass API.")

    m.save("plot.html")
    print("Map saved as 'plot.html'")

    return m