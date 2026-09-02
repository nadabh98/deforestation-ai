import streamlit as st
import requests
import folium

from streamlit_folium import st_folium


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "https://deforestation-api-702314968420.europe-west1.run.app"

# Zone d'étude : Manaus, Amazonas, Brésil
MANAUS_LAT = -3.1190
MANAUS_LON = -60.0217


# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Deforestation AI",
    page_icon="🌳",
    layout="wide"
)


# ============================================================
# TITRE
# ============================================================

st.title("🌳 Deforestation AI")

st.subheader(
    "Détection de la déforestation par imagerie satellite Sentinel-2"
)

st.markdown(
    """
    Cette application analyse l'évolution d'une zone entre
    **2020 et 2024** grâce à un modèle de segmentation
    **U-Net** entraîné sur des images Sentinel-2.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.write("**Zone d'étude**")
    st.write("Manaus, Amazonas, Brésil")

    st.write("**Période**")
    st.write("2020 → 2024")

    st.write("**Modèle**")
    st.write("U-Net")

    st.write("**Seuil de décision**")
    st.write("0.70")

    st.divider()

    st.info(
        """
        Le modèle réalise une segmentation pixel par pixel
        afin d'identifier les zones présentant une
        déforestation potentielle.
        """
    )


# ============================================================
# IMAGES SATELLITE
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
# ANALYSE
# ============================================================

if image_2020 is not None and image_2024 is not None:

    st.success(
        "✅ Les deux images satellite sont prêtes pour l'analyse."
    )

    if st.button(
        "🔍 Analyser la déforestation",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "🧠 Analyse par le modèle U-Net en cours..."
        ):

            try:

                # ------------------------------------------------
                # Préparation des fichiers
                # ------------------------------------------------

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
                # Appel API Cloud Run
                # ------------------------------------------------

                response = requests.post(
                    f"{API_URL}/analyze",
                    files=files,
                    timeout=180
                )

                if response.status_code != 200:
			 st.error(
        			f"Erreur API ({response.status_code}) : "
        			f"{response.text}"
   				 )
    			st.stop()

		result = response.json()

                # =================================================
                # RESULTATS
                # =================================================

                st.header("📊 Résultats de l'analyse")

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "🌳 Déforestation détectée",
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


                # =================================================
                # INTERPRETATION
                # =================================================

                percentage = result["deforestation_percentage"]

                st.subheader("📈 Interprétation")

                if percentage < 5:

                    st.success(
                        f"🌿 Faible niveau de déforestation détecté : "
                        f"{percentage:.2f} %."
                    )

                elif percentage < 15:

                    st.warning(
                        f"⚠️ Niveau modéré de déforestation détecté : "
                        f"{percentage:.2f} %."
                    )

                else:

                    st.error(
                        f"🚨 Niveau élevé de déforestation détecté : "
                        f"{percentage:.2f} %."
                    )


                # =================================================
                # CARTE
                # =================================================

                st.header("🗺️ Carte de la zone analysée")

                st.markdown(
                    """
                    La carte ci-dessous localise la zone d'étude
                    autour de **Manaus, Amazonas, Brésil**.
                    """
                )

                m = folium.Map(
                    location=[
                        MANAUS_LAT,
                        MANAUS_LON
                    ],
                    zoom_start=9,
                    control_scale=True
                )


                # ------------------------------------------------
                # Marqueur Manaus
                # ------------------------------------------------

                folium.Marker(

                    [
                        MANAUS_LAT,
                        MANAUS_LON
                    ],

                    popup=folium.Popup(
                        """
                        <b>Zone d'étude</b><br>
                        Manaus<br>
                        Amazonas, Brésil
                        """,
                        max_width=300
                    ),

                    tooltip="📍 Manaus"

                ).add_to(m)


                # ------------------------------------------------
                # Cercle représentant la zone étudiée
                # ------------------------------------------------

                folium.Circle(

                    location=[
                        MANAUS_LAT,
                        MANAUS_LON
                    ],

                    radius=25000,

                    popup=(
                        f"Déforestation détectée : "
                        f"{percentage:.2f} %"
                    ),

                    tooltip=(
                        f"Déforestation : "
                        f"{percentage:.2f} %"
                    ),

                    fill=True

                ).add_to(m)


                # ------------------------------------------------
                # Affichage
                # ------------------------------------------------

                st_folium(
                    m,
                    width=None,
                    height=500
                )


                # =================================================
                # INFORMATIONS MODELE
                # =================================================

                st.header("🤖 Informations du modèle")

                model_col1, model_col2 = st.columns(2)

                with model_col1:

                    st.write(
                        f"**Architecture :** U-Net"
                    )

                    st.write(
                        f"**Epoch :** {result['model_epoch']}"
                    )

                    st.write(
                        f"**Threshold :** {result['threshold']}"
                    )

                    st.write(
                        f"**Device :** CPU"
                    )

                with model_col2:

                    st.write(
                        f"**Taille image :** "
                        f"{result['width']} × "
                        f"{result['height']}"
                    )

                    st.write(
                        f"**CRS 2020 :** "
                        f"{result['crs_2020']}"
                    )

                    st.write(
                        f"**CRS 2024 :** "
                        f"{result['crs_2024']}"
                    )

                    st.write(
                        f"**Période analysée :** "
                        f"2020 → 2024"
                    )


                # =================================================
                # RESUME
                # =================================================

                st.header("📋 Résumé")

                st.markdown(
                    f"""
                    **Zone :** Manaus, Amazonas, Brésil

                    **Période :** 2020 → 2024

                    **Pixels analysés :**
                    {result['total_pixels']:,}

                    **Pixels identifiés comme déforestés :**
                    {result['deforested_pixels']:,}

                    **Pourcentage détecté :**
                    {percentage:.2f} %

                    **Modèle :** U-Net

                    **Epoch du modèle :**
                    {result['model_epoch']}

                    **Seuil :**
                    {result['threshold']}
                    """
                )


                # =================================================
                # AVERTISSEMENT
                # =================================================

                st.warning(
                    """
                    ⚠️ **Important :**

                    Le résultat correspond à une détection automatique
                    basée sur l'imagerie satellite et le modèle U-Net.
                    Il s'agit d'une estimation algorithmique et non
                    d'une validation terrain.
                    """
                )


            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ L'API a mis trop de temps à répondre. "
                    "Veuillez réessayer."
                )


            except requests.exceptions.RequestException as e:

                st.error(
                    f"❌ Erreur de communication avec l'API : {e}"
                )


            except Exception as e:

                st.error(
                    f"❌ Erreur inattendue : {e}"
                )


# ============================================================
# MESSAGE AVANT UPLOAD
# ============================================================

else:

    st.info(
        """
        👆 Veuillez importer les deux images Sentinel-2
        **2020** et **2024** afin de lancer l'analyse.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    🌳 Deforestation AI — Projet ODD 15  
    Détection de la déforestation par intelligence artificielle
    et imagerie satellite Sentinel-2.
    """
)
