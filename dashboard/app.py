import math
import requests
import streamlit as st
import folium

from streamlit_folium import st_folium
from folium.plugins import Draw


# ============================================================
# CONFIGURATION
# ============================================================

# URL de l'API FastAPI déployée sur Cloud Run.
API_URL = (
    "https://deforestation-api-702314968420.europe-west1.run.app"
)

# Coordonnées par défaut : Manaus
DEFAULT_LAT = -3.1190
DEFAULT_LON = -60.0217

# ============================================================
# CONFIGURATION DU PIPELINE V2
# ============================================================

# 256 pixels × 10 m/pixel = 2560 m
ZONE_SIZE_METERS = 2560

# Recherche de l'image Sentinel-2 récente
RECENT_DAYS = 90

# Seuil du modèle U-Net
THRESHOLD = 0.70


# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Deforestation AI",
    page_icon="🌳",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "selected_zone" not in st.session_state:
    st.session_state.selected_zone = None


# ============================================================
# FONCTION : CONVERSION DU RECTANGLE EN ZONE
# ============================================================

def rectangle_to_zone(drawing):
    """
    Convertit le rectangle GeoJSON retourné par Folium
    en coordonnées géographiques et dimensions en mètres.

    Limite du pipeline V2 :
        2560 m × 2560 m

    Le modèle travaille sur :
        256 × 256 pixels
        10 m/pixel
    """

    if not drawing:
        return None

    geometry = drawing.get("geometry")

    if not geometry:
        return None

    coordinates = geometry.get("coordinates")

    if not coordinates:
        return None

    points = coordinates[0]

    if len(points) < 4:
        return None

    # GeoJSON :
    # [longitude, latitude]

    lons = [point[0] for point in points]
    lats = [point[1] for point in points]

    west = min(lons)
    east = max(lons)

    south = min(lats)
    north = max(lats)

    # --------------------------------------------------------
    # CENTRE
    # --------------------------------------------------------

    center_lat = (south + north) / 2.0
    center_lon = (west + east) / 2.0

    # --------------------------------------------------------
    # CONVERSION DEGRÉS -> MÈTRES
    # --------------------------------------------------------

    meters_per_degree_lat = 111320.0

    meters_per_degree_lon = (
        111320.0
        * math.cos(math.radians(center_lat))
    )

    height_m = (
        north - south
    ) * meters_per_degree_lat

    width_m = (
        east - west
    ) * meters_per_degree_lon

    # Conversion en entiers
    width_m = int(round(width_m))
    height_m = int(round(height_m))

    # --------------------------------------------------------
    # TAILLE CARRÉE POUR L'API V2
    # --------------------------------------------------------

    size_m = max(
        width_m,
        height_m
    )

    size_m = int(size_m)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    valid = (
        width_m > 0
        and height_m > 0
        and width_m <= ZONE_SIZE_METERS
        and height_m <= ZONE_SIZE_METERS
    )

    return {
        "latitude": center_lat,
        "longitude": center_lon,

        "width_m": width_m,
        "height_m": height_m,

        "size_m": size_m,

        "bounds": [
            [south, west],
            [north, east]
        ],

        "valid": valid
    }


# ============================================================
# HEADER
# ============================================================

st.title("🌳 Deforestation AI")

st.subheader(
    "Détection de la déforestation par imagerie satellite Sentinel-2"
)

st.markdown(
    """
    Cette application utilise automatiquement **Google Earth Engine**
    pour récupérer une image satellite de référence et une image récente,
    puis applique un modèle **U-Net** afin de détecter les changements
    pouvant correspondre à de la déforestation.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.write("**Source des images**")
    st.write("Google Earth Engine")

    st.write("**Satellite**")
    st.write("Sentinel-2")

    st.write("**Image de référence**")
    st.write("2020")

    st.write("**Image récente**")
    st.write(
        f"Dernière image disponible sur "
        f"{RECENT_DAYS} jours"
    )

    st.write("**Zone maximale**")
    st.write(
        f"{ZONE_SIZE_METERS} m × "
        f"{ZONE_SIZE_METERS} m"
    )

    st.write("**Résolution**")
    st.write("10 m / pixel")

    st.write("**Image modèle**")
    st.write("256 × 256 pixels")

    st.write("**Modèle**")
    st.write("U-Net")

    st.write("**Seuil de décision**")
    st.write(f"{THRESHOLD}")

    st.divider()

    st.info(
        f"""
        🛰️ Les images satellite sont récupérées
        automatiquement.

        🌍 Vous sélectionnez directement la zone
        sur la carte.

        📐 Taille maximale :
        **{ZONE_SIZE_METERS} × {ZONE_SIZE_METERS} m**

        🤖 Le modèle U-Net détecte ensuite
        les changements.
        """
    )


# ============================================================
# CARTE DE SÉLECTION
# ============================================================

st.header("🗺️ 1. Sélectionner une zone")

st.markdown(
    f"""
    Utilisez l'outil **rectangle ▭** dans la barre
    d'outils de la carte pour sélectionner la zone.

    ⚠️ **La taille maximale autorisée est
    {ZONE_SIZE_METERS} × {ZONE_SIZE_METERS} m.**

    Une sélection dépassant cette taille sera refusée.
    """
)


# ============================================================
# CRÉATION DE LA CARTE
# ============================================================

m = folium.Map(
    location=[
        DEFAULT_LAT,
        DEFAULT_LON
    ],
    zoom_start=12,
    control_scale=True,
    tiles=None
)


# ============================================================
# COUCHE SATELLITE
# ============================================================

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Esri World Imagery",
    name="🛰️ Satellite",
    overlay=False,
    control=True
).add_to(m)


# ============================================================
# COUCHE OPENSTREETMAP
# ============================================================

folium.TileLayer(
    tiles="OpenStreetMap",
    name="🗺️ OpenStreetMap",
    overlay=False,
    control=True
).add_to(m)


# ============================================================
# OUTIL DE DESSIN
# ============================================================

Draw(
    export=False,
    position="topleft",

    draw_options={
        "polyline": False,
        "polygon": False,
        "circle": False,
        "marker": False,
        "circlemarker": False,

        "rectangle": {
            "shapeOptions": {
                "weight": 2
            }
        }
    },

    edit_options={
        "edit": True,
        "remove": True
    }

).add_to(m)


# ============================================================
# CONTRÔLE DES COUCHES
# ============================================================

folium.LayerControl().add_to(m)


# ============================================================
# AFFICHAGE DE LA CARTE
# ============================================================

map_data = st_folium(
    m,
    width=None,
    height=600,
    key="selection_map"
)


# ============================================================
# RÉCUPÉRATION DE LA SÉLECTION
# ============================================================

if (
    map_data
    and map_data.get("last_active_drawing")
):

    zone = rectangle_to_zone(
        map_data["last_active_drawing"]
    )

    if zone:

        # ----------------------------------------------------
        # ZONE TROP GRANDE
        # ----------------------------------------------------

        if not zone["valid"]:

            st.session_state.selected_zone = None

            st.error(
                f"""
                ❌ **Zone trop grande**

                Votre sélection mesure :

                **{zone['width_m']} m × "
                f"{zone['height_m']} m**

                La taille maximale autorisée est :

                **{ZONE_SIZE_METERS} m × "
                f"{ZONE_SIZE_METERS} m**

                👉 Veuillez dessiner une zone plus petite.
                """
            )

        # ----------------------------------------------------
        # ZONE VALIDE
        # ----------------------------------------------------

        else:

            st.session_state.selected_zone = zone


# ============================================================
# AFFICHAGE DE LA ZONE SÉLECTIONNÉE
# ============================================================

if st.session_state.selected_zone:

    zone = st.session_state.selected_zone

    st.header("📐 2. Zone sélectionnée")


    # ========================================================
    # MÉTRIQUES
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Latitude",
            f"{zone['latitude']:.6f}"
        )


    with col2:

        st.metric(
            "Longitude",
            f"{zone['longitude']:.6f}"
        )


    with col3:

        st.metric(
            "Largeur",
            f"{zone['width_m']} m"
        )


    with col4:

        st.metric(
            "Hauteur",
            f"{zone['height_m']} m"
        )


    # ========================================================
    # INFORMATIONS SUR LA ZONE
    # ========================================================

    st.info(
        f"""
        📍 **Centre :**
        {zone['latitude']:.6f},
        {zone['longitude']:.6f}

        📐 **Rectangle sélectionné :**
        {zone['width_m']} × {zone['height_m']} m

        🤖 **Taille utilisée par l'API V2 :**
        {zone['size_m']} × {zone['size_m']} m

        🛰️ **Résolution :**
        10 m/pixel

        🔢 **Image du modèle :**
        256 × 256 pixels

        📐 **Limite maximale :**
        {ZONE_SIZE_METERS} × {ZONE_SIZE_METERS} m
        """
    )


    # ========================================================
    # AVERTISSEMENT SI RECTANGLE NON CARRÉ
    # ========================================================

    if abs(
        zone["width_m"]
        - zone["height_m"]
    ) > 100:

        st.warning(
            f"""
            ⚠️ **Le rectangle n'est pas parfaitement carré.**

            Votre sélection mesure :

            **{zone['width_m']} m × "
            f"{zone['height_m']} m**

            L'API V2 utilise une zone carrée correspondant
            au plus grand côté :

            **{zone['size_m']} × {zone['size_m']} m**

            centrée sur votre sélection.
            """
        )


    # ========================================================
    # BOUTON ANALYSE
    # ========================================================

    st.header("🤖 3. Lancer l'analyse")


    if st.button(
        "🔍 Analyser cette zone",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # VÉRIFICATION FINALE
        # ----------------------------------------------------

        if not zone["valid"]:

            st.error(
                "❌ Cette zone dépasse la taille maximale autorisée."
            )

            st.stop()


        # ----------------------------------------------------
        # VÉRIFICATION DES COORDONNÉES
        # ----------------------------------------------------

        latitude = float(
            zone["latitude"]
        )

        longitude = float(
            zone["longitude"]
        )

        if not -90 <= latitude <= 90:

            st.error(
                "❌ Latitude invalide."
            )

            st.stop()


        if not -180 <= longitude <= 180:

            st.error(
                "❌ Longitude invalide."
            )

            st.stop()


        # ----------------------------------------------------
        # PAYLOAD API
        # ----------------------------------------------------

        payload = {
            "latitude": latitude,
            "longitude": longitude,

            # IMPORTANT :
            # toujours un entier
            "size": int(
                zone["size_m"]
            ),

            "recent_days": int(
                RECENT_DAYS
            )
        }


        # ----------------------------------------------------
        # SÉCURITÉ API
        # ----------------------------------------------------

        if payload["size"] > ZONE_SIZE_METERS:

            st.error(
                f"""
                ❌ La taille demandée dépasse la limite.

                Taille :
                {payload['size']} m

                Maximum :
                {ZONE_SIZE_METERS} m
                """
            )

            st.stop()


        # ====================================================
        # APPEL API
        # ====================================================

        with st.spinner(
            "🛰️ Récupération des images Sentinel-2 "
            "et analyse par le modèle U-Net..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/analyze-v2",
                    json=payload,
                    timeout=300
                )


                # =================================================
                # ERREUR API
                # =================================================

                if response.status_code != 200:

                    st.error(
                        "❌ L'analyse n'a pas pu être réalisée."
                    )

                    st.error(
                        f"Erreur du serveur API : "
                        f"HTTP {response.status_code}"
                    )

                    try:

                        error_detail = (
                            response.json()
                        )

                        st.code(
                            str(error_detail)
                        )

                    except Exception:

                        st.code(
                            response.text
                        )


                # =================================================
                # SUCCÈS
                # =================================================

                else:

                    result = response.json()


                    # Sauvegarde dans session_state
                    st.session_state.analysis_result = (
                        result
                    )


                    st.success(
                        "✅ Analyse terminée avec succès."
                    )


            # =====================================================
            # TIMEOUT
            # =====================================================

            except requests.exceptions.Timeout:

                st.error(
                    """
                    ⏱️ **Le serveur a mis trop de temps
                    à répondre.**
                    """
                )

                st.info(
                    """
                    La récupération des images Sentinel-2
                    et l'inférence du modèle peuvent prendre
                    plusieurs minutes.
                    """
                )


            # =====================================================
            # ERREUR CONNEXION
            # =====================================================

            except requests.exceptions.ConnectionError:

                st.error(
                    """
                    ❌ **Impossible de contacter
                    l'API Cloud Run.**
                    """
                )

                st.info(
                    """
                    Vérifiez que l'API est disponible
                    et que son URL est correcte.
                    """
                )


            # =====================================================
            # AUTRE ERREUR HTTP
            # =====================================================

            except requests.exceptions.RequestException as e:

                st.error(
                    "❌ Erreur lors de la communication "
                    "avec l'API."
                )

                st.code(
                    str(e)
                )


            # =====================================================
            # ERREUR INATTENDUE
            # =====================================================

            except Exception as e:

                st.error(
                    "❌ Une erreur inattendue est survenue."
                )

                st.code(
                    str(e)
                )


# ============================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================

if st.session_state.analysis_result is not None:

    result = (
        st.session_state.analysis_result
    )


    # ========================================================
    # TITRE
    # ========================================================

    st.header(
        "📊 Résultats de l'analyse"
    )


    # ========================================================
    # CALCUL DU POURCENTAGE
    # ========================================================

    try:

        deforested_pixels = int(
            result["deforested_pixels"]
        )

        total_pixels = int(
            result["total_pixels"]
        )

        if total_pixels > 0:

            percentage = (
                deforested_pixels
                / total_pixels
                * 100
            )

        else:

            percentage = 0.0

    except (
        KeyError,
        TypeError,
        ValueError
    ):

        percentage = 0.0


    # ========================================================
    # MÉTRIQUES PRINCIPALES
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "🌳 Pixels déforestés",
            f"{deforested_pixels:,}"
        )


    with col2:

        st.metric(
            "🔢 Pixels analysés",
            f"{total_pixels:,}"
        )


    with col3:

        st.metric(
            "📊 Déforestation détectée",
            f"{percentage:.2f} %"
        )


    # ========================================================
    # INTERPRÉTATION
    # ========================================================

    st.subheader(
        "📈 Interprétation"
    )


    if percentage < 5:

        st.success(
            f"""
            🌱 **Faible proportion détectée : "
            f"{percentage:.2f} %.**
            """
        )

    elif percentage < 15:

        st.warning(
            f"""
            ⚠️ **Proportion modérée détectée : "
            f"{percentage:.2f} %.**
            """
        )

    else:

        st.error(
            f"""
            🚨 **Proportion élevée détectée : "
            f"{percentage:.2f} %.**
            """
        )


    st.caption(
        """
        Cette classification est basée uniquement
        sur la proportion de pixels détectés par
        le modèle. Elle ne constitue pas une
        validation terrain.
        """
    )


    # ========================================================
    # CARTES DE RÉSULTATS
    # ========================================================

    st.header(
        "🗺️ Résultats cartographiques"
    )


    map_col1, map_col2 = st.columns(2)


    # ========================================================
    # CARTE GÉOGRAPHIQUE
    # ========================================================

    with map_col1:

        st.subheader(
            "📍 Zone analysée"
        )


        result_map = folium.Map(
            location=[
                result["latitude"],
                result["longitude"]
            ],
            zoom_start=14,
            control_scale=True
        )


        # ----------------------------------------------------
        # MARQUEUR
        # ----------------------------------------------------

        folium.Marker(

            [
                result["latitude"],
                result["longitude"]
            ],

            popup=folium.Popup(
                f"""
                <b>Zone analysée</b><br>
                Latitude :
                {result['latitude']}<br>
                Longitude :
                {result['longitude']}<br>
                Taille :
                {result['size_meters']} m
                """,

                max_width=350
            ),

            tooltip="📍 Zone analysée"

        ).add_to(
            result_map
        )


        # ----------------------------------------------------
        # CERCLE DE LA ZONE
        # ----------------------------------------------------

        folium.Circle(

            location=[
                result["latitude"],
                result["longitude"]
            ],

            radius=(
                result["size_meters"]
                / 2
            ),

            popup=(
                f"Déforestation détectée : "
                f"{percentage:.2f} %"
            ),

            tooltip=(
                f"Déforestation : "
                f"{percentage:.2f} %"
            ),

            fill=True

        ).add_to(
            result_map
        )


        # ----------------------------------------------------
        # AFFICHAGE
        # ----------------------------------------------------

        st_folium(
            result_map,
            width=None,
            height=500
        )


    # ========================================================
    # MASQUE DE DÉFORESTATION
    # ========================================================

    with map_col2:

        st.subheader(
            "🌳 Masque de déforestation"
        )


        try:

            mask_url = (
                f"{API_URL}"
                f"{result['mask_url']}"
            )


            mask_response = requests.get(
                mask_url,
                timeout=60
            )


            if mask_response.status_code == 200:

                st.image(
                    mask_response.content,

                    caption=(
                        "Masque généré par "
                        "le modèle U-Net"
                    ),

                    use_container_width=True
                )

            else:

                st.error(
                    "❌ Impossible de récupérer "
                    "le masque."
                )


        except requests.exceptions.RequestException as e:

            st.error(
                "❌ Erreur lors du chargement "
                "du masque."
            )

            st.code(
                str(e)
            )


    # ========================================================
    # CARTE DE PROBABILITÉ
    # ========================================================

    st.subheader(
        "🧠 Carte de probabilité du modèle"
    )


    try:

        probability_url = (
            f"{API_URL}"
            f"{result['probability_url']}"
        )


        probability_response = requests.get(
            probability_url,
            timeout=60
        )


        if probability_response.status_code == 200:

            st.image(
                probability_response.content,

                caption=(
                    "Probabilité prédite par "
                    "le modèle U-Net"
                ),

                use_container_width=True
            )

        else:

            st.error(
                "❌ Impossible de récupérer "
                "la carte de probabilité."
            )


    except requests.exceptions.RequestException as e:

        st.error(
            """
            ❌ Erreur lors du chargement de
            la carte de probabilité.
            """
        )

        st.code(
            str(e)
        )


    # ========================================================
    # INFORMATIONS DU MODÈLE
    # ========================================================

    st.header(
        "🤖 Informations du modèle"
    )


    model_col1, model_col2 = st.columns(2)


    with model_col1:

        st.write(
            "**Architecture :** U-Net"
        )

        st.write(
            f"**Threshold :** "
            f"{result['threshold']}"
        )

        st.write(
            "**Device :** CPU"
        )


    with model_col2:

        st.write(
            f"**Taille de la zone :** "
            f"{result['size_meters']} × "
            f"{result['size_meters']} m"
        )

        st.write(
            f"**Pixels analysés :** "
            f"{result['total_pixels']:,}"
        )

        st.write(
            "**Résolution :** 10 m/pixel"
        )


    # ========================================================
    # INFORMATIONS TECHNIQUES
    # ========================================================

    st.header(
        "🔬 Informations techniques"
    )


    st.markdown(
        f"""
        **Job ID :**
        `{result['job_id']}`

        **Latitude :**
        {result['latitude']}

        **Longitude :**
        {result['longitude']}

        **Image de référence :**
        {result['reference_date']}

        **Image récente :**
        {result['recent_date']}

        **Seuil de segmentation :**
        {result['threshold']}

        **Pixels déforestés :**
        {result['deforested_pixels']:,}

        **Pixels analysés :**
        {result['total_pixels']:,}

        **Déforestation détectée :**
        {percentage:.2f} %
        """
    )


    # ========================================================
    # AVERTISSEMENT SCIENTIFIQUE
    # ========================================================

    st.warning(
        """
        ⚠️ **Interprétation du résultat**

        Le pourcentage affiché correspond à une
        détection automatique produite par le modèle
        U-Net.

        Il s'agit d'une estimation algorithmique à
        partir d'images Sentinel-2 et non d'une
        validation terrain.

        Le modèle utilisé a été entraîné sur des paires
        d'images historiques. L'utilisation d'une image
        récente peut donc nécessiter une validation
        complémentaire pour une interprétation
        scientifique.
        """
    )


# ============================================================
# MESSAGE INITIAL
# ============================================================

if (
    st.session_state.analysis_result
    is None
):

    st.info(
        f"""
        👆 Dessinez un rectangle sur la carte
        pour sélectionner une zone.

        📐 **Taille maximale :**
        {ZONE_SIZE_METERS} × {ZONE_SIZE_METERS} m

        🛰️ Les images Sentinel-2 seront récupérées
        automatiquement par Google Earth Engine.
        """
    )
