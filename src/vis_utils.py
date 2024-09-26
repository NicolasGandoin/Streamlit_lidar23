"""
Package d'utilitaire de visualisation DEC23 - LIDAR
"""
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as path_effects
import numpy as np
import cv2


from matplotlib.colors import ListedColormap

def plot_lidar_scatter3D(df, ctype=None, ax= None, exageration=1, height='Z', psize=0.2, color='r', alpha=1):
    """
    plot scatter3D avec type de couleur
    :param ctype: indicate type of colors to be shown  'rgb' (uses R,G,B), or other df field (default classification)
    :param ax: axes on which we plot the scatter3D
    :param exageration: how many points we want to exaggerate
    :param height: 'Z' for altitude, 'H' or  height (without altitude)
    :type df: Dataframe
    """
    df = df.copy()
    if ax is None:
        ax = plt.subplot(111, projection='3d')
    if ctype == 'rgb':
        colors = df.apply(lambda x: '#%02x%02x%02x' % (int(x.R), int(x.G), int(x.B)), axis=1)
        ax.scatter(df.X / 100, df.Y / 100, df[height] / 100, c=colors,  s=psize,alpha=alpha)
    elif ctype in ['intensity', 'return_number', 'number_of_returns', 'scan_direction_flag', 'scan_angle', 'point_source_id','H']:
        colors = 255 * (df[ctype] - df[ctype].min()) / (df[ctype].max() - df[ctype].min())
        ax.scatter(df.X / 100, df.Y / 100, df[height] / 100, c=colors, cmap='viridis', s=psize, alpha=alpha)
    elif ctype == 'classification' : # default classification
        #todo défaut d'affichage légende
        df.loc[df.classification > 6, 'classification'] = 0 #seulement les 6 premières
        cmap = plt.colormaps.get_cmap('viridis')  # Obtenir la colormap
        colors = cmap(np.linspace(0, 1, 7))  # Générer les couleurs par classif
        color_dict = {i: colors[i] for i in range(7)}
        ax.scatter(df.X/100, df.Y/100, df[height]/100, c=df['classification'], cmap=ListedColormap(colors), s=psize , alpha=alpha)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[i], markersize=10, label=str(i)) for i in color_dict]
        # Ajouter une légende
        ax.legend(handles=handles, title='Classification', loc='upper right')
        plt.title(f'Lidar colored by classification')
    else:
        ax.scatter(df.X / 100, df.Y / 100, df[height]/100, c=color,  s=psize, alpha=alpha)

    # range sur x
    xrange = ax.get_xlim()
    diffx = (xrange[1] - xrange[0])
    # range sur z
    zrange = ax.get_zlim()  # ajustement z sur la même dimension avec facteur d'exagération
    ratio = (zrange[1] - zrange[0])/diffx
    ax.set_box_aspect([1, 1, exageration * ratio])
    return ax



def plotly_lidar_scatter3D(df, ctype=None, exageration=1, height='Z', psize=0.2, alpha=1, wwidth=800, wheight=800):
    """
    plot scatter3D avec type de couleur
    :param ctype: indicate type of colors to be shown  'rgb' (uses R,G,B), or other df field (default classification)
    :param exageration: how many points we want to exaggerate
    :param height: 'Z' for altitude, 'H' or  height (without altitude)
    :param width: width of the plot
    :param height: height of the plot
    :type df: Dataframe

    """
    df = df.copy()
    xmin, ymin = df.X.min()/100 , df.Y.min() /100 # cm

    # BDA 20240716 Ramener les X,Y en unités locales car erreurs d'arrondis sur Y (tracé en bandes)
    df['X'] = df['X'] / 100 - xmin
    df['Y'] = df['Y'] / 100 - ymin
    df[height] = df[height] / 100

    if ctype == 'rgb':
        colors = df.apply(lambda x: 'rgb(%d,%d,%d)' % (int(x.R), int(x.G), int(x.B)), axis=1)
        fig = go.Figure(data=[go.Scatter3d(
            x=df['X'], y=df['Y'], z=df[height],
            mode='markers',
            marker=dict(size=psize * 10, color=colors, opacity=alpha),

        )] )
    elif ctype in ['intensity', 'return_number', 'number_of_returns', 'scan_direction_flag', 'scan_angle',
                   'point_source_id', 'H']:
        df['color'] = 255 * (df[ctype] - df[ctype].min()) / (df[ctype].max() - df[ctype].min())
        fig = px.scatter_3d(df, x='X', y='Y', z=height, color='color', opacity=alpha, color_continuous_scale='Viridis')
    elif ctype == 'classification':  # default classification
        df.loc[df.classification > 6, 'classification'] = 0  # seulement les 6 premières
        fig = px.scatter_3d(df, x='X', y='Y', z=height, color='classification', opacity=alpha,
                            color_continuous_scale='Viridis')
        fig.update_layout(title='Lidar colored by classification')
    elif ctype in ['plane_id']:
        df['plane_id'] = df.plane_id.astype('str')
        df = df.sort_values(by=['plane_id'])
        fig = px.scatter_3d(df, x='X', y='Y', z=height, color='plane_id', opacity=alpha,
                            hover_data={'classification': True, 'H': True})
    else:
        # color rgba fournie
        fig = px.scatter_3d(df, x='X', y='Y', z=height, color='color', opacity=alpha,
                            hover_data={'classification': True, 'H': True})

    x_range = [df['X'].min(), df['X'].max()]
    y_range = [df['Y'].min(), df['Y'].max()]
    z_range = [df[height].min(), df[height].max()]

    # Compute aspect ratio
    x_diff = x_range[1] - x_range[0]
    y_diff = y_range[1] - y_range[0]
    z_diff = (z_range[1] - z_range[0]) * exageration

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range),
            aspectmode='manual',
            aspectratio=dict(x=1, y=y_diff / x_diff, z=z_diff / x_diff),

        ),
        width=wwidth,
        height=wheight,
        margin=dict(l=0, r=0, t=0, b=0),

    )
    return fig


def plotly_add_plane(fig, plane):
    """
    TODO ATTENTION NE SEMBLE PAS FONTIONNER CORRECTEMENT DO NOT USE AS IT IS
    Ajoute un plan défini par l'équation ax + by + cz + d = 0 à une figure Plotly 3D existante.
    :param fig: Figure Plotly existante
    :param plane: Liste des coefficients [a, b, c, d] de l'équation du plan
    """
    a, b, c, d = plane

    # Récupérer les limites des axes x et y depuis la figure
    x_range = fig.layout.scene.xaxis.range
    y_range = fig.layout.scene.yaxis.range

    # Si les limites ne sont pas définies, utiliser les valeurs par défaut
    if x_range is None:
        x_range = [df['X'].min(), df['X'].max()]
    if y_range is None:
        y_range = [df['Y'].min(), df['Y'].max()]

    # Création d'une grille de points x, y
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 10), np.linspace(y_range[0], y_range[1], 10))

    # Calcul des valeurs z correspondantes
    zz = (-a * xx - b * yy - d) / c

    # Ajout du plan à la figure
    fig.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=1, showscale=False))




def plot_lidar_scatter2D(df, ctype=None, ax= None, psize=0.2, color='r', alpha=1):
    """
    plot scatter3D avec type de couleur
    :param ctype: indicate type of colors to be shown  'rgb' (uses R,G,B), or other df field (default classification)
    :param ax: axes on which we plot the scatter3D
    :type df: Dataframe
    """
    df = df.copy()
    if ax is None:
        ax = plt.subplot(111)
    if ctype == 'rgb':
        colors = df.apply(lambda x: '#%02x%02x%02x' % (int(x.R), int(x.G), int(x.B)), axis=1)
        ax.scatter(df.X / 100, df.Y / 100, c=colors,  s=psize, alpha=alpha)
    elif ctype in ['intensity', 'return_number', 'number_of_returns', 'scan_direction_flag', 'scan_angle', 'point_source_id','H']:
        colors = 255 * (df[ctype] - df[ctype].min()) / (df[ctype].max() - df[ctype].min())
        ax.scatter(df.X / 100, df.Y / 100,  c=colors, cmap='viridis', s=psize, alpha=alpha)
    elif ctype == 'classification' : # default classification
        #todo défaut d'affichage légende
        df.loc[df.classification > 6, 'classification'] = 0 #seulement les 6 premières
        cmap = plt.colormaps.get_cmap('viridis')  # Obtenir la colormap
        colors = cmap(np.linspace(0, 1, 7))  # Générer les couleurs par classif
        color_dict = {i: colors[i] for i in range(7)}
        ax.scatter(df.X/100, df.Y/100, c=df['classification'], cmap=ListedColormap(colors), s=psize, alpha=alpha)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[i], markersize=10, label=str(i)) for i in color_dict]
        # Ajouter une légende
        ax.legend(handles=handles, title='Classification', loc='upper right')
        plt.title(f'Lidar colored by classification')
    else:
        ax.scatter(df.X / 100, df.Y / 100, c=color,  s=psize, alpha=alpha)

    return ax


def plot_polygons(ax, gdf, bounds=None, text_field=None, text_color='black'):
    """
    Plot Multipolygons and polygones from geodataframe in ax,
    limit size to boundaries passed with bounds
    """
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            z = [0] * len(x)  # Assumer que les multipolygones sont plats (z = 0)
            ax.plot(x, y, color='r')
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                z = [0] * len(x)  # Assumer que les multipolygones sont plats (z = 0)
                ax.plot(x, y, color='r')
        if text_field is not None:
            text = row[text_field]
            centroid = geom.centroid
            text = ax.text(centroid.x, centroid.y, f'{text[-9:]}', color=text_color, fontsize=7, ha='center', va='center')
            # Ajouter l'effet de contour blanc
            text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                                   path_effects.Normal()])

    if bounds is not None:
        ax.set(xlim=(bounds[0],bounds[2]), ylim=(bounds[1],bounds[3]))


if __name__ == '__main__':
    # test file
    df = pd.read_pickle('../data/')

    pass
