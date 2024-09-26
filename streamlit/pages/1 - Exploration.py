
import folium
import pandas as pd
import streamlit as st
import cv2
import numpy as np
import sys
import os
import pyproj
sys.path.append(os.getcwd())
rootdir = os.getcwd()
import src.o3d_utils as odu
import geopandas as gpd
import laspy
import src.topo_utils as tu
import src.vis_utils as vu
import src.las_utils as lu
import src.raster_utils as ru
#from src.raster_utils import show_colors_dominance
import src.IGN_API_utils as api
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
from folium.plugins import Geocoder
from PIL import Image


@st.cache_data
def load_batiments(bounds):
    df = api.load_batiments(bounds)
    df['superficie'] = round(df.geometry.apply(lambda g: g.area),2)
    return df

@st.cache_data
def load_ortho_img(bounds):
    print(f'load_ortho_img {bounds}')
    img = api.load_bdortho(bounds)
    img = ru.add_batiment_on_image(img, bounds, bat)
    return img

def generate_map(gdf):
    # tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
    m = gdf.explore(column='usage_1', popup=["cleabs", "nature", "usage_1", "superficie", "hauteur"], 
    tooltip=["cleabs", "nature", "usage_1", "superficie", "hauteur"])  
    Geocoder().add_to(m)  
    return m

#Configuration page 
st.set_page_config(
    page_title="LidarDec23 - Exploration",
    page_icon="🗺️",
    layout="wide")

st.title('Exploration des données')

st.sidebar.title("Types de données")
sections = ['BDTOPO', 'ORTHOHR', 'LIDAR']
section = st.sidebar.radio('', sections)

#initialisation point de départ
starting_point = (45.7459, 4.8850)  # latitude longitude
transformer = pyproj.Transformer.from_crs(4326, 2154)
la93 = transformer.transform(starting_point[0], starting_point[1])

st.write("Sélectionner une zone sur la carte et cliquer sur 'Reload/Clear selection' pour charger les bâtiments (BDTOPO). Les bâtiments seront colorés selon le champ 'usage_1' de la BDTOPO.")

#traitement selection sidebar BDTOPO

if section == "BDTOPO":
    st.header('BDTOPO (bâtiments)')
    col1, col2, col3 = st.columns(3, vertical_alignment="top", gap='medium')
    if 'bounds' not in st.session_state:
        st.session_state.bounds = la93[0] - 300, la93[1] - 300, la93[0] + 300, la93[1] + 300
    batiments = load_batiments(st.session_state.bounds) 
    st.session_state.selected_cleabs=''
    if 'batiments' not in st.session_state:
        st.session_state.batiments = batiments
    cleabs=''

    with col1:
        st.subheader('Polygones') 
        cont = st.container(height=500, border=0)
        with cont: 
            m = generate_map(st.session_state.batiments)
            map_ = st_folium(m, width=450, height=450)
            print(map_['bounds'])        
        if st.button('Reload/Clear selection'):
            bounds_SW = transformer.transform(map_['bounds']['_southWest']['lat'], map_['bounds']['_southWest']['lng'])
            bounds_NE = transformer.transform(map_['bounds']['_northEast']['lat'], map_['bounds']['_northEast']['lng'])
            st.session_state.bounds = bounds_SW + bounds_NE
            st.session_state.batiments = load_batiments(st.session_state.bounds)
            print(st.session_state.cleabs) 
            st.write('Bounds:', st.session_state.bounds)               
            m = generate_map(st.session_state.batiments)
            map_ = st_folium(m, width=450, height=450)
        st.write('Bounds:', st.session_state.bounds)  
        # Capture de l'événement de clic
        if map_ and map_.get('last_object_clicked'):
            clicked_info = map_['last_object_clicked_tooltip']
            cleabs_clicked = clicked_info.split("cleabs\n")[1].split("\n")[1].replace(" ","")  # Extrait 'cleabs' de la popup
            st.session_state.selected_cleabs = cleabs_clicked    
            cleabs = st.sidebar.text_input('cleabs BDTOPO', st.session_state.selected_cleabs, key='cleabs') 
        else:
            cleabs = st.sidebar.text_input('cleabs BDTOPO', '', key='cleabs')  
            
    # Chargement de la BDTOPO associée sous les colonnes 
    st.subheader('La BDTOPO: données')
    bats = load_batiments(st.session_state.bounds)
    bats['geometry_str'] = bats['geometry'].apply(lambda geom: geom.wkt if geom else None)
    bats.drop(columns=['geometry'], inplace= True)
    st.write("Nb de bâtiments de l'emprise:", len(bats))  
    st.write('Bâtiment(s) sélectionné(s):')
    if 'selected_cleabs' in st.session_state and st.session_state.selected_cleabs != '':
        st.dataframe(bats[bats['cleabs'] == st.session_state.selected_cleabs])
    else:
        st.dataframe(bats)

    with col2:# Graphique camembert valeurs manquantes (bdtopo chargée)
        st.subheader("% de valeurs manquantes") 
        with st.container(height=500, border=0):               
            bats_isna = pd.DataFrame(bats.isna().sum(), columns=['nb_na'])
            bats_isna['%_na'] = round((bats_isna.nb_na / bats_isna.nb_na.sum() *100),2)
            bats_isna=bats_isna.reset_index()
            bats_isna=bats_isna[bats_isna.nb_na !=0].sort_values('%_na', ascending = False)             
            fig = px.pie(bats_isna,values='%_na', names = 'index')
            fig.update_layout(
            width=450,        
            height=450,
            showlegend=False)                      
            st.plotly_chart(fig)

    with col3:# Graphique Histogrammes/Boxplots selon colonne (bdtopochargée)
        st.subheader("Distribution par colonne")
        with st.container(height=500, border=0): 
            columns = bats.columns.tolist()
            columns = [col for col in bats.columns if col not in(['cleabs','geometry_str'])]
            selected_column = st.selectbox("Sélectionnez une colonne pour la répartition :", columns)
            # Créer le graphique de répartition en fonction du type de données
            if pd.api.types.is_numeric_dtype(bats[selected_column]):
            # Si la colonne est numérique, afficher un histogramme
                fig = px.box(bats, y=selected_column, title=f"Répartition de {selected_column}")
            else:
            # Si la colonne est catégorielle, afficher un graphique en barres
                fig = px.histogram(bats, x=selected_column, color=selected_column, 
                category_orders={selected_column: bats[selected_column].unique()})
            # Afficher le graphique
            fig.update_layout(
            width=400,        
            height=400,
            showlegend=False)     
            st.plotly_chart(fig)

    #Ajout de l'image ortho dans le sidebar         
    if cleabs !='':
        batiments_ = st.session_state.batiments
        bats = batiments_[batiments_.cleabs == cleabs]
        if bats.shape[0] > 0:
            bat = bats.iloc[0]
            bat_bounds = tu.get_squared_bounds(bat.geometry, 2)
            print(bat_bounds)
            img = load_ortho_img(bat_bounds)
            fig = plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.ylim([0, 1000])
            plt.axis('off')
            st.sidebar.pyplot(fig)     

if section == "ORTHOHR":
    tab1, tab2 = st.tabs(['Choix de l\'emprise', 'Colorimétrie RGB/HSV'])
    #onglet Choix de l'emprise
    with tab1:
        st.header('ORTHO Photos')
        st. write ("Les images satellites de la BD ORTHO sont corrigées des effets d'angles de prise de vues, sont produites avec une résolution de 20 cm (=1 pixel), en couleurs et en infra-rouge couleur")
       
        if 'bounds' not in st.session_state:
            st.session_state.bounds = la93[0] - 300, la93[1] - 300, la93[0] + 300, la93[1] + 300
            batiments = load_batiments(st.session_state.bounds) 
        
        col1, col2= st.columns(2 ,vertical_alignment="top", gap='small')
        if 'batiments' not in st.session_state:
            st.session_state.batiments = batiments
        
        with col1:
            st.subheader('Emprise avec bâtiments BDTOPO')
            with st.container(height=600, border=0):
                m = generate_map(st.session_state.batiments)
                map_ = st_folium(m, width=500, height=500, use_container_width=False)
                print(map_['bounds'])  
                if st.button('Reload/Clear selection'):
                    #st.sidebar.text_input('cleabs BDTOPO', '', key='cleabs') 
                    bounds_SW = transformer.transform(map_['bounds']['_southWest']['lat'], map_['bounds']['_southWest']['lng'])
                    bounds_NE = transformer.transform(map_['bounds']['_northEast']['lat'], map_['bounds']['_northEast']['lng'])
                    st.session_state.bounds = bounds_SW + bounds_NE
                    st.session_state.batiments = load_batiments(st.session_state.bounds)
                    print(st.session_state.cleabs)
                    m = generate_map(st.session_state.batiments)
                    map_ = st_folium(m, width=500, height=500,use_container_width=False )
                # Capture de l'événement de clic
                if map_ and map_.get('last_object_clicked'):
                    clicked_info = map_['last_object_clicked_tooltip']
                    cleabs_clicked = clicked_info.split("cleabs\n")[1].split("\n")[1].replace(" ","")  # Extrait 'cleabs' de la popup
                    st.session_state.selected_cleabs = cleabs_clicked    
                    cleabs = st.sidebar.text_input('cleabs BDTOPO', st.session_state.selected_cleabs, key='cleabs') 
                else:
                    cleabs = st.sidebar.text_input('cleabs BDTOPO', '', key='cleabs')
    with col2:
        st.subheader('ORTHO Photo')
        with st.container(height=600, border=0):
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
            else:
                bounds_SW = transformer.transform(map_['bounds']['_southWest']['lat'], map_['bounds']['_southWest']['lng'])
                bounds_NE = transformer.transform(map_['bounds']['_northEast']['lat'], map_['bounds']['_northEast']['lng'])
                st.session_state.bounds = bounds_SW + bounds_NE
                img = api.load_bdortho(st.session_state.bounds)
                fig = plt.figure(figsize=(3, 3))
                plt.imshow(img)
                plt.ylim([0, 1000])
                plt.axis('off')
                st.pyplot(fig, use_container_width=False)
            
            st.sidebar.pyplot(fig)

    # onglet Colorimétrie HSV/RGB
    with tab2:
            col1, col2= st.columns(2 ,vertical_alignment="top", gap='large')
            # ORTHOPHOTO - HSV
            with col1:
                with st.container(height=650, border=0):
                    st.subheader('ORTHO Photo HSV')
                    channel = st.radio("Choisir un canal:", ['H - Teinte', 'S - Saturation', 'V - Valeur'], horizontal=True)

                    # Application du flou et conversion en HSV
                    #imgh = cv2.blur(img, ksize=(3, 3))  
                    imgh=img
                    imgh = cv2.cvtColor(imgh, cv2.COLOR_RGB2HSV) 

                    # Sélection du canal approprié
                    if channel == 'H - Teinte':
                        selected_channel = imgh[..., 0]  # Canal H
                        vmin, vmax = 0, 180  # Échelle de 0 à 180 pour la teinte
                        title = 'Teinte (H)'
                        colorscale = [[i/360, f"hsl({i}, 100%, 50%)"] for i in range(0, 361, 60)]  # Couleurs de la teinte basées sur la plage HSL
                    
                    elif channel == 'S - Saturation':
                        selected_channel = imgh[..., 1]  # Canal S
                        vmin, vmax = 0, 255  # Échelle de 0 à 255 pour la saturation
                        title = 'Saturation (S)'
                        colorscale = 'gray' 
                    else:
                        selected_channel = imgh[..., 2]  # Canal V
                        vmin, vmax = 0, 255  # Échelle de 0 à 255 pour la valeur
                        title = 'Valeur (V)'
                        colorscale = 'gray'  

                    # Création de la figure avec Plotly
                    fig = go.Figure(data=go.Heatmap(
                        z=selected_channel,
                        colorscale=colorscale,  
                        zmin=vmin,  
                        zmax=vmax,  
                        colorbar=dict(
                        title=title,
                        tickvals=np.linspace(vmin, vmax, num=10),  # Ajout de graduations à intervalles réguliers
                        ticktext=[f'{int(v)}' for v in np.linspace(vmin, vmax, num=10)],  # Affichage des valeurs sur la colorbar
                        ticks="outside",  # Les ticks de la colorbar sont à l'extérieur
                        x=1.05,  # Décalage horizontal de la colorbar (plus proche ou éloigné de l'image)
                        xpad=10,  # Distance entre la colorbar et le graphique
                        )))

                    # Mise en page
                    fig.update_layout(
                        title=title,
                        xaxis=dict(scaleanchor="y",showticklabels=False),  
                        yaxis=dict(showticklabels=False),
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        margin=dict(t=30, b=30, l=30, r=50)
                    )

                    # Affichage du graphique dans Streamlit
                    st.plotly_chart(fig, use_container_width=True)     

            with col2:
                # Répartition des pixels 
                st.subheader("Distribution des pixels de l'image")
                with st.container(height=650, border=0):
                    channel2 = st.radio("Choisir:", ['HSV', 'RGB'],horizontal=True )
                   
                    if channel2 == 'RGB':
                        fig = plt.figure(figsize=(3, 3))
                        blue_channel = img[..., 2]
                        green_channel = img[..., 1]
                        red_channel = img[..., 0]
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=blue_channel.ravel(), histnorm='probability density', name='Canal Bleu', marker_color='blue', opacity=0.7))
                        fig.add_trace(go.Histogram(x=green_channel.ravel(), histnorm='probability density', name='Canal Vert', marker_color='green', opacity=0.7))
                        fig.add_trace(go.Histogram(x=red_channel.ravel(), histnorm='probability density', name='Canal Rouge', marker_color='red', opacity=0.7))
                        fig.update_layout(
                            title='Densité des valeurs des canaux cf. photo sidebar ou onglet 1',
                            xaxis_title='Valeur des pixels',
                            yaxis_title='Densité',
                            barmode='overlay',
                            legend_title_text='Canaux'
                            )
                            # Ajuster l'opacité pour que les courbes se chevauchent de manière lisible
                        fig.update_traces(opacity=0.5)
                        st.plotly_chart(fig, use_container_width=True)

                    if channel2 == 'HSV':
                     
                        # Traitement selon le canal choisi
                        if  channel == 'H - Teinte':
                            selected_channel = imgh[..., 0]  # Canal H (Hue)
                            hist, bins = np.histogram(selected_channel.flatten(), bins=8, range=[0, 180])
                            # Normalisation pour la teinte (0 à 180)
                            colors = ['hsl({}, 100%, 50%)'.format(int(v / 180 * 360)) for v in bins[:-1]]
                        elif channel == 'S - Saturation':
                            selected_channel = imgh[..., 1]  # Canal S (Saturation)
                            hist, bins = np.histogram(selected_channel.flatten(), bins=8, range=[0, 255])
                            # Normalisation pour la saturation (0 à 255)
                            colors = ['hsl(0, 0%, {}%)'.format(int(v / 255 * 100)) for v in bins[:-1]]
                        else:
                            selected_channel = imgh[..., 2]  # Canal V (Value)
                            hist, bins = np.histogram(selected_channel.flatten(), bins=8, range=[0, 255])
                            # Normalisation pour la valeur (0 à 255)
                            colors = ['hsl(0, 0%, {}%)'.format(int(v / 255 * 100)) for v in bins[:-1]]

                        # Création de l'histogramme avec Plotly
                        fig = go.Figure()

                        for i in range(len(hist)):
                            fig.add_trace(go.Bar(
                                x=[(bins[i] + bins[i + 1]) / 2],  # Position au centre du bin
                                y=[hist[i]],
                                width=[bins[i + 1] - bins[i]],
                                marker_color=colors[i],  # Couleur des barres
                                showlegend=False
                            ))

                        # Mise en forme
                        fig.update_layout(
                            xaxis_title=channel,
                            yaxis_title='Fréquence',
                            plot_bgcolor='rgba(0, 0, 0, 0)',
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                            bargap=0.1,
                            height=400
                        )

                        # Affichage du graphique dans Streamlit
                        st.plotly_chart(fig, use_container_width=True)
if section == "LIDAR":
    st.subheader('Données LIDAR')
    st.write("Ces nuages de points sont issus de relevés aériens. Chaque point est identifié en coordonnées x,y,z (x,y en coordonnées Lambert et z l’altitude) et porte des informations complémentaires le caractérisant. (Information sur la prise de vue, l’acquisition mais aussi une classification établies en post-traitement par l’IGN. La classification suit la norme ASPRS (pour plus d’information voir le descriptif contenuIGN ici: https://geoservices.ign.fr/sites/default/files/2023-10/DC_LiDAR_HD_1-0_PTS.pdf")

    st.write("Nous avons choisi de présenter ces données à travers la détection de plans avec l'algorithme RANSAC et vous renvoyons à la rubrique 'Détection de pentes de toits'")