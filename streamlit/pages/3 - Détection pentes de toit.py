
import os
import sys
sys.path.append(os.getcwd())
rootdir = os.getcwd()

import folium
import pandas as pd
import streamlit as st
import pyproj
import src.IGN_API_utils as api
import src.topo_utils as tu
import src.raster_utils as ru
import src.las_utils as lu
import src.vis_utils as vu
import src.o3d_utils as odu
import geopandas as gpd
import matplotlib.pyplot as plt
import laspy

from streamlit_folium import st_folium

st.set_page_config(
    page_title="LidarDec23 - Calcul des pans de toiture - RANSAC ",
    page_icon="🗺️",
    layout="wide"
)


st.title("Lidar - Détection des pans de toiture")

@st.cache_data
def load_batiments(bounds):
    df = api.load_batiments(bounds)
    df['superficie'] = df.geometry.apply(lambda g: g.area)
    return df

@st.cache_data
def load_ortho_img(bounds):
    print(f'load_ortho_img {bounds}')
    img = api.load_bdortho(bounds)
    img = ru.add_batiment_on_image(img, bounds, bat)
    return img

@st.cache_data
def load_lidar_urls(bounds):
    # TODO traiter le cas des chevauchements (concat 2 urls)
    return api.load_lidar_urls(bounds)


@st.cache_data
def load_las(bounds):
    rootdir = os.getcwd()
    urls = load_lidar_urls(bounds)  # TODO gérer les urls "à cheval"
    if len(urls) > 0:
        url = urls[0]
        # st.text(f"Source lidar de  {url['name']}")
        # téléchargement du fichier las
        lasfile = api.download_las_file(url['name'], url['url'], filepath=os.path.join(rootdir, 'data', 'lidar'))
        lasdata = laspy.read(lasfile)
        fulllasdf = lu.las_to_df(lasdata)  # là on a tout le las, il faut filtre
        las = fulllasdf[
            (fulllasdf.X >= bounds[0] * 100) & (fulllasdf.X <= bounds[2] * 100) & (fulllasdf.Y >= bounds[1] * 100) & (
                        fulllasdf.Y <= bounds[3] * 100)].copy()
        # facultatif ajout RGB et H
        lu.add_rgb_column(las, bounds)
        lu.add_height_column(las, bounds)
        las = lu.add_bdtopo_cleabs_column(las, bounds)  # Il se peut qu'il y ait plusieurs batiments (collés)
        return las
    else:
        return pd.DataFrame()

# reprise de fonction d'origine
@st.cache_data
def compute_planes(bat_df, bat_topo, min_points=200, distance_threshold=25, ransac_n=3, num_iteration=1000):
    print(bat_topo.cleabs)
    df = bat_df.copy()
    df['idx'] = df.index  # keep track of old index
    df.reset_index(drop=True, inplace=True)  # les indexes doivent être propres

    df, planes_ = odu.detect_planes_from_pcd(df, min_points=min_points, distance_threshold=distance_threshold, ransac_n=3,
                                             num_iterations=1000)
    df.set_index('idx', inplace=True)  # Remettre l'index original
    return df, planes_


def generate_map(gdf):
    # tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
    m = gdf.explore(column='usage_1', popup=["cleabs", "nature", "usage_1", "superficie", "hauteur"], tooltip=["cleabs", "nature", "usage_1", "superficie", "hauteur"]) # , tiles=tiles, attr='ESRI')
    return m

tab1, tab2  = st.tabs(['choix du batiment', 'Calcul des pans de toiture'])
with tab1:
    starting_point = (45.7459, 4.8850)  # latitude longitude
    transformer = pyproj.Transformer.from_crs(4326, 2154)
    la93 = transformer.transform(starting_point[0], starting_point[1])
    # TODO make boundaries fit map. Button refresh
    st.session_state.bounds = la93[0] - 300, la93[1] - 300, la93[0] + 300, la93[1] + 300
    batiments = load_batiments(st.session_state.bounds) #cached version
    if 'batiments' not in st.session_state:
        st.session_state.batiments = batiments
    gdf = batiments.to_crs(4326)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(height=550, border=0):
            m = generate_map(st.session_state.batiments)
            # m = folium.Map(location=list(starting_point), zoom_start=18)
            map_ = st_folium(m, width=500, height=500)
            print(map_['bounds'])
        with st.container():
            if st.button('reload'):
                bounds_SW = transformer.transform(map_['bounds']['_southWest']['lat'], map_['bounds']['_southWest']['lng'])
                bounds_NE = transformer.transform(map_['bounds']['_northEast']['lat'], map_['bounds']['_northEast']['lng'])
                st.session_state.bounds = bounds_SW + bounds_NE
                # cleabs = '' # removing current cleabs
                st.session_state.batiments = load_batiments(st.session_state.bounds)
                print(st.session_state.cleabs)
                m = generate_map(st.session_state.batiments)
                map_ = st_folium(m, width=500, height=500)
            # Capture de l'événement de clic cf NBL
            if map_ and map_.get('last_object_clicked'):
                clicked_info = map_['last_object_clicked_tooltip']
                cleabs_clicked = clicked_info.split("cleabs\n")[1].split("\n")[1].strip()  # Extrait 'cleabs' de la popup
                st.session_state.selected_cleabs = cleabs_clicked
                # st.session_state.cleabs.value = cleabs_clicked --- on ne peut pas updater la value !!!
                cleabs = st.sidebar.text_input('cleabs BDTOPO', st.session_state.selected_cleabs, key='cleabs')
            else:
                cleabs = st.sidebar.text_input('cleabs BDTOPO', '', key='cleabs')
                pass

    with col2:
        if cleabs !='':
            batiments = st.session_state.batiments
            bats = batiments[batiments.cleabs == cleabs]
            if bats.shape[0] > 0:
                bat = bats.iloc[0]
                bat_bounds = tu.get_squared_bounds(bat.geometry, 2)
                print(bat_bounds)
                img = load_ortho_img(bat_bounds)
                fig = plt.figure(figsize=(3, 3))
                plt.imshow(img)
                plt.ylim([0, 1000])
                plt.axis('off')
                st.pyplot(fig, use_container_width=False)
                st.sidebar.pyplot(fig)

with tab2:
    if cleabs !='':
        col1, col2 = st.columns(2)
        urls = load_lidar_urls(st.session_state.bounds)
        if len(urls) > 0:
            url = urls[0] # gérer
            bats = batiments[batiments.cleabs == cleabs]
            if bats.shape[0] > 0:
                bat = bats.iloc[0]
                bat_bounds = tu.get_squared_bounds(bat.geometry, 2)
                las = load_las(bat_bounds)
                if las.shape[0] > 0:
                    with col1:
                        fig1 = vu.plotly_lidar_scatter3D(las, ctype='rgb', height='Z', psize=0.5, alpha=0.5, wwidth=1200, wheight=600)
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        bat = batiments[batiments.cleabs == cleabs].copy().iloc[0]
                        batdf = las[las.cleabs == bat.cleabs].copy()
                        c1, c2 = st.columns(2)
                        with c1:
                            min_points = st.slider('Ransac min points', value=200, min_value=50, max_value=1000 , key="min_points")
                        with c2:
                            distance_threshold = st.slider('Seuil distance', min_value=1, max_value=50, value=25, key="distance_threshold")
                        batdf, planes = compute_planes(batdf, bat, min_points=min_points, distance_threshold=distance_threshold)
                        fig2 = vu.plotly_lidar_scatter3D(batdf, height='Z', ctype='plane_id', psize=0.5, alpha=0.5, wwidth=1200, wheight=600);
                        # ne pas afficher les points du plane_id zéro
                        for i, trace in enumerate(fig2.data):
                            if trace.name == '0':
                                trace.visible = 'legendonly'
                        st.plotly_chart(fig2, use_container_width=True)
                        for idx, (p, nb) in enumerate(planes):
                            st.sidebar.text(f'Plan {idx + 1} {nb} points- {int(odu.calculate_inclination(p))}°')
                else:
                    st.sidebar.text(f'No las file available')
        else:
            st.text('No lidar urls available')
    # st.dataframe(las)
# call to render Folium map in Streamlit
