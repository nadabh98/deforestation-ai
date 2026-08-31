import streamlit as st
import requests
import folium

from streamlit_folium import st_folium


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "https://deforestation-api-702314968420.europe-west1.run.app"


# ============================================================
# CONFIGURATION PAGE
# ============================================================

st.set_page_config(
    page_title="Deforestation AI",
    page_icon="🌳",
    layout="wide"
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: bold;
    }

    .subtitle {
        font-size: 20px;
        color: #666;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITRE
# ============================================================

st.markdown(
    '<div class="main-title">🌳 Deforestation AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Détection de la déforestation par imagerie satellite Sentinel-2'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    """
    Cette application compare des images satellite Sentinel-2
    de **2020 et 2024** afin d'identifier les zones présentant
    des signes de déforestation.
    
    Le modèle utilisé est un réseau de segmentation **U-Net**.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌳 Deforestation AI")

st.sidebar.markdown(
    """
    ### Projet

    **ODD 15 — Vie terrestre**

    Analyse automatique de la déforestation
    à partir d'images satellites Sentinel-2.

    ---

    ### Modèle

    **U-Net**

    Epoch : **15**

    Threshold : **0.70**

    ---

    ### Données

    🛰️ Sentinel-2

    📅 2020 → 2024

    📐 Résolution : 256 × 256 pixels
    """
)


# ============================================================
# UPLOAD
# ============================================================

st.header("🛰️ Images satellite")

col1, col2 = st.columns(2)

with col1:

    image_2020 = st.file_uploader(
        "Image Sentinel-2 — 2020",
        type=["tif", "tiff"],
        key="image_2020"
    )

with col2:

    image_2024 = st.file_uploader(
        "Image Sentinel-2 — 2024",
        type=["tif", "tiff"],
        key="image_2024"
    )


# ============================================================
# PREVIEW
# ============================================================

if image_2020 is not None and image_2024 is not None:

    st.success(
        "✅ Les deux images satellite sont prêtes."
    )

    st.divider()

    # --------------------------------------------------------
    # Bouton analyse
    # --------------------------------------------------------

    if st.button(
        "🔍 Analyser la déforestation",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "🛰️ Analyse des images par le modèle U-Net..."
        ):

            try:

                files = {

                    "image_2020": (
                        image_2020.name,
                        image_2020.getvalue(),
                        "image/tiff"
                    ),

                    "image_2024": (
                        image_2024.name,
                        image_2024.getvalue(),
                        "image/tiff"
                    )
                }

                # ------------------------------------------------
                # Appel API
                # ------------------------------------------------

                response = requests.post(
                    f"{API_URL}/analyze",
                    files=files,
                    timeout=180
                )

                response.raise_for_status()

                result = response.json()

                # ------------------------------------------------
                # Sauvegarde du résultat
                # ------------------------------------------------

                st.session_state["result"] = result

            except requests.exceptions.RequestException as e:

                st.error(
                    f"❌ Erreur de communication avec l'API : {e}"
                )

            except Exception as e:

                st.error(
                    f"❌ Erreur : {e}"
                )


# ============================================================
# AFFICHAGE RESULTATS
# ============================================================

if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()

    st.header("📊 Résultats de l'analyse")


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🌳 Déforestation",
            f"{result['deforestation_percentage']:.2f} %"
        )

    with col2:

        st.metric(
            "🔴 Pixels déforestés",
            f"{result['deforested_pixels']:,}"
        )

    with col3:

        st.metric(
            "🛰️ Pixels analysés",
            f"{result['total_pixels']:,}"
        )

    with col4:

        st.metric(
            "🤖 Epoch",
            result["model_epoch"]
        )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    percentage = result["deforestation_percentage"]

    if percentage < 5:

        st.success(
            f"🟢 Zone relativement stable : "
            f"{percentage:.2f}% de la surface détectée comme déforestée."
        )

    elif percentage < 15:

        st.warning(
            f"🟠 Déforestation modérée détectée : "
            f"{percentage:.2f}% de la surface."
        )

    else:

        st.error(
            f"🔴 Niveau important de déforestation détecté : "
            f"{percentage:.2f}% de la surface."
        )


    # ========================================================
    # INFORMATIONS GEOSPATIALES
    # ========================================================

    st.subheader("🌍 Informations géospatiales")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📅 Image 2020")

        st.write(
            f"**CRS :** {result['crs_2020']}"
        )

        st.write(
            f"**Dimensions :** "
            f"{result['width']} × {result['height']}"
        )

        if "resolution_2020" in result:

            st.write(
                f"**Résolution :** "
                f"{result['resolution_2020'][0]:.2f} × "
                f"{result['resolution_2020'][1]:.2f}"
            )

    with col2:

        st.markdown("### 📅 Image 2024")

        st.write(
            f"**CRS :** {result['crs_2024']}"
        )

        st.write(
            f"**Dimensions :** "
            f"{result['width']} × {result['height']}"
        )

        if "resolution_2024" in result:

            st.write(
                f"**Résolution :** "
                f"{result['resolution_2024'][0]:.2f} × "
                f"{result['resolution_2024'][1]:.2f}"
            )


    # ========================================================
    # CARTE
    # ========================================================

    st.divider()

    st.header("🗺️ Localisation de la zone analysée")

    st.markdown(
        """
        La carte ci-dessous représente l'emprise géographique
        de l'image satellite analysée.
        """
    )


    # --------------------------------------------------------
    # Bounds
    # --------------------------------------------------------

    bounds = result.get("bounds_2020")

    if bounds:

        left = bounds["left"]
        bottom = bounds["bottom"]
        right = bounds["right"]
        top = bounds["top"]

        center_lat = (bottom + top) / 2
        center_lon = (left + right) / 2

        # ----------------------------------------------------
        # Création carte
        # ----------------------------------------------------

        m = folium.Map(
            location=[
                center_lat,
                center_lon
            ],
            zoom_start=10,
            control_scale=True
        )

        # ----------------------------------------------------
        # Rectangle zone analysée
        # ----------------------------------------------------

        folium.Rectangle(
            bounds=[
                [bottom, left],
                [top, right]
            ],
            tooltip="Zone analysée",
            popup=(
                f"Déforestation détectée : "
                f"{percentage:.2f}%"
            ),
            weight=3
        ).add_to(m)

        # ----------------------------------------------------
        # Marker
        # ----------------------------------------------------

        folium.Marker(
            [
                center_lat,
                center_lon
            ],
            tooltip="Zone analysée",
            popup=(
                f"""
                <b>Deforestation AI</b><br>
                Déforestation : {percentage:.2f}%<br>
                Pixels déforestés : {result['deforested_pixels']:,}
                """
            )
        ).add_to(m)

        # ----------------------------------------------------
        # Affichage
        # ----------------------------------------------------

        st_folium(
            m,
            width=None,
            height=500
        )

    else:

        st.warning(
            "Les informations géographiques ne sont pas disponibles."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌳 Deforestation AI — Projet ODD 15 | "
    "Sentinel-2 + U-Net + FastAPI + Streamlit + Cloud Run"
) 
