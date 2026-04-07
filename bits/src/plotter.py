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


def plot3d(data):
    """
    Traceurs 3d de satellites

    :param data: [liste_x_ecef, liste_y_ecef, liste_z_ecef, liste_noms]
    :return:
    """
    plt.figure(label="Satellites en orbites")
    ax = plt.axes(projection="3d")
    ax.set_xlim(-50000000, 50000000)
    ax.set_ylim(-50000000, 50000000)
    ax.set_zlim(-50000000, 50000000)
    ax.set_aspect("equal", adjustable="box")

    # Plot earth
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = 6378000 * np.outer(np.cos(u), np.sin(v))
    y = 6378000 * np.outer(np.sin(u), np.sin(v))
    z = 6378000 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b')

    # Plot data
    ax.scatter3D(data[0], data[1], data[2], color='r', s=1)

    try:
        for i in range(len(data[3])):
            ax.text(data[0][i], data[1][i], data[2][i], data[3][i], size=5, zorder=1)
    except:
        print("Plot 3D : Noms inutilisables")


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

    locations = pd_gnss_pvt[['lat', 'lon']].values.tolist()

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