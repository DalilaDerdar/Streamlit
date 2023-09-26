import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, learning_curve, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.over_sampling import SMOTE
import streamlit as st
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
import requests
import json


# Fonction pour afficher les graphiques
def plot_parallel_bars(client_data, similar_clients_data, top_features):
    # Créer des données pour le graphique
    categories = top_features
    
    # Valeurs du client
    client_values = client_data[top_features].values.tolist()[0]
    
    # Valeurs moyennes des clients similaires
    similar_values = similar_clients_data[top_features].mean().tolist()

    # Créer le graphique
    fig = go.Figure(data=[
        go.Bar(name='Client', x=categories, y=client_values, marker_color='#CC00FF'),
        go.Bar(name='Clients Similaires (Moyenne)', x=categories, y=similar_values, marker_color='#0033FF')
    ])
    
    # Disposition du graphique
    fig.update_layout(
        title="Comparaison du facteur clef, entre le client et les clients similaires",
        barmode='group',
        yaxis_title="Valeur",
        xaxis_title="Facteurs clefs",
        template="plotly_dark"  # Pour un thème moderne sombre
    )
    
    return fig

# Charger le modèle formé
@st.cache_data
def load_model():
    model = joblib.load('best_model.pkl')
    print(f'{model=}')
    return model

#Predire avec API Flask
def predict_with_api(features, route):
    url_base = "https://dalid.azurewebsites.net/" # "http://localhost:5000/"  #vérifier l'url VS url de flask_app.py
    url = url_base + route
    response = requests.post(url, json=json.dumps(features.to_dict()))  

    # Vérifier le statut de la réponse
    if response.status_code != 200:
        st.error(f"Erreur avec le statut {response.status_code}: {response.text}")
        return None  # ou retournez une valeur par défaut ou une erreur spécifique

    result = response.json()
    return result
    


def load_data():
    X = joblib.load('X.pkl')
    X_test = joblib.load('X_test.pkl')
    return X, X_test

X, X_test = load_data()


# Application principale
def main():
    st.markdown("<h1 style='text-align: center; color: #800020 ;'>PRET A DEPENSER</h1>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    
    # Charger le modèle
    model = load_model()

    # Charger les données d'échantillon pour la démo
    import pandas as pd
    sample_df = pd.read_csv('data/sample_df.csv')
    X_sample = sample_df.drop(columns=["TARGET"])
    print(f'{X_sample=}')
    print('\n\n\n')
    print(f'{type(X_sample)=}') 
    # Créer une sidebar
    index_selected = st.sidebar.selectbox("""Choisissez le dossier client à tester en sélectionnant 
                                          son numéro de dossier:""", X_sample.index)
    data_to_predict = X_sample.loc[[index_selected]]

    
    st.markdown("""<h2 style='text-align: left; color: #5a5a5a;'>Résultat d'analyse du dossier client n°{}</h2>""".format(index_selected), 
                unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)


    if st.sidebar.button('Lancer la prédiction'):
        selected_row = X_sample.loc[index_selected]
        selected_features = selected_row

        prediction = predict_with_api(selected_features, 'predict')

        if prediction == 0:
            result_text = "accepté"
        else:
            result_text = "refusé"
        
        st.markdown(f"""
            <div style="background-color: {'green' if prediction == 0 else 'red'}; 
                         padding: 10px 15px; border-radius: 5px; width: 50%; margin: 0 auto;">
                <h4 style="color: white; text-align: center;">Crédit {result_text.capitalize()}</h4>
            </div>
        """, unsafe_allow_html=True)

        if prediction == 0:
            st.write("Votre client semble avoir les éléments requis pour rembourser son crédit durablement. "
                     "Nous conseillons l’obtention du prêt.")
        else:
            st.write("Votre client ne semble pas avoir les éléments nécessaires pour rembourser son crédit durablement. "
                     "Nous ne conseillons pas l’obtention du prêt.")

        st.markdown("<br>", unsafe_allow_html=True)
        

    else:
        st.write("Veuillez cliquer sur le bouton pour obtenir la prédiction.")


    # Option "Information du client"
    if st.sidebar.checkbox("Informations du client"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: left; color: #800020 ;'>Informations du client</h3>", unsafe_allow_html=True)

        with st.expander("Cliquez pour afficher les détails"):
            st.write("""Vous trouverez les informations du client qui ont permis de faire l'analyse de son dossier. 
                     Une vue globale pour permet de voir toutes les informations qui ont été prises en compte. 
                     Vous pouvez également sélectionner les colonnes qui vous intéressent pour un affichage détaillé.""")
        st.write(data_to_predict)
        st.markdown("<br>", unsafe_allow_html=True)

    # Sélection de colonnes spécifiques pour un affichage détaillé
        columns_without_index = [col for col in sample_df.columns if col != 'index']  # Remplacez 'index' par le nom réel de votre colonne d'index, si différent
        specific_columns = st.multiselect("Choisissez les colonnes pour un affichage détaillé:", columns_without_index)
        if specific_columns:
            st.markdown("<h4>Détails sélectionnés</h4>", unsafe_allow_html=True)
            st.write(data_to_predict[specific_columns])
        st.markdown("<br>"*3, unsafe_allow_html=True)

    # Option "Facteurs clefs du client"
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if st.sidebar.checkbox("Facteurs clefs du client"):
        st.markdown("<h3 style='text-align: left; color: #800020 ;'>Facteurs clefs du client</h3>", unsafe_allow_html=True)
        with st.expander("Cliquez pour afficher les détails"):
            st.markdown("""<b>Comment lire ce graphique:</b><br>
Ce graphique illustre l'importance des différents facteurs pris en compte lors de l'évaluation du dossier du client. Les barres peuvent avoir des couleurs allant du bleu au rouge, et leur position sur l'axe horizontal indique leur niveau d'influence sur la décision finale.<br>

<b>Axe Vertical:</b><br>
Chaque barre du graphique représente un facteur spécifique ou une caractéristique du client. Ces facteurs sont triés par ordre d'importance, les facteurs les plus influents étant situés en haut du graphique.<br>

<b>Axe Horizontal:</b><br>
L'axe horizontal représente le niveau d'importance de chaque facteur. Une barre qui s'étend vers la droite indique une influence positive sur la prédiction d'un non remboursement par le client (augmente la probabilité de refuser le prêt), tandis qu'une barre s'étendant vers la gauche indique une influence négative sur la prédiction d'un non remboursement par le client (diminue la probabilité de refuser le prêt).<br>

<b>Couleur des Barres:</b><br>
La couleur des barres offre une indication visuelle supplémentaire du niveau d'influence : les barres rouges représentent une influence positive importante, tandis que les barres bleues représentent une influence négative importante. Plus la couleur est intense, plus l'effet est important.<br>
""", unsafe_allow_html=True)

        # Charger les données
        X_sample, _ = load_data()

        # Récupérer les données du client sélectionné
        data_to_predict = X_sample.loc[[index_selected]]
        print(f'{data_to_predict=}')
        print('\n')
        print(f'{type(data_to_predict)=}')
        print('\n')
        print(f'{(data_to_predict.shape)}')
        print('\n\n\n')

        # Appeler la fonction pour obtenir la prédiction et les SHAP values
        features_importance = predict_with_api(data_to_predict, 'explain')
        print(f'{type(features_importance)=}')
        print(f'{len(features_importance)=}')
        print('\n\n\n')
        

        # Applatie la liste de listes en une seule liste
        features_importance_flat = [item for sublist in features_importance for item in sublist]

        # Créé un DataFrame avec les noms des features et les valeurs d'importance
        import pandas as pd
        features_df = pd.DataFrame({'feature': data_to_predict.columns, 'importance': features_importance_flat})

        # Supprime la ligne correspondant à la feature 'index' (ajustez 'index_feature_name' selon le nom correct)
        index_feature_name = 'index'  
        features_df = features_df[features_df['feature'] != index_feature_name]

        # Trie le DataFrame par les valeurs d'importance en ordre décroissant
        features_df = features_df.sort_values(by='importance', ascending=False)

        # Sélectionnez les 3 caractéristiques les plus positives et les 3 les plus négatives
        top_10_positive = features_df.head(10)
        bottom_10_negative = features_df.tail(10)

        # Combine les deux sous-ensembles de données pour obtenir un total de 6 caractéristiques
        filtered_features_df = pd.concat([top_10_positive, bottom_10_negative], axis=0)

        # Triez-les par importance pour l'affichage
        filtered_features_df = filtered_features_df.sort_values(by='importance', ascending=True)

        
        # Créé un bar plot avec les SHAP values
        import plotly.express as px
        fig = px.bar(x=filtered_features_df['importance'], y=filtered_features_df['feature'], orientation='h', color=filtered_features_df['importance'], color_continuous_scale='bluered')
        # fig = px.bar(x=features_df['importance'], y=features_df['feature'], orientation='h', color=features_df['importance'], color_continuous_scale='bluered')
        fig.update_layout(
        xaxis_title="<b>Niveau d'Importance</b>", 
        yaxis_title="<b>Facteurs clefs</b>",
        height=800,
        width=1000
        )
        st.plotly_chart(fig)


    # Option "Information sur les facteurs clefs généraux"
    import plotly.express as px

    if st.sidebar.checkbox("Information sur les facteurs clefs généraux"):
        st.markdown("<h3 style='text-align: left; color: #800020 ;'>Information sur les facteurs clefs généraux</h3>", unsafe_allow_html=True)
        with st.expander("Cliquez pour afficher les détails"):
            st.write(""" Cette section présente les facteurs clés qui ont le plus grand impact sur les décisions de prêt.
                    Chaque point du graphique représente une caractéristique de vos données. Plus le point est à droite,
                    plus cette caractéristique a un impact fort sur la prédiction de capacité de remboursement, et plus il est à gauche, plus elle a un impact faible.
                     Vous pouvez choisir le nombre de facteurs que vous souhaitez afficher.""")
        importances = model.named_steps['classifier'].feature_importances_
        feature_importances = pd.DataFrame({'Feature': X_sample.columns, 'Importance': importances})
        feature_importances = feature_importances.sort_values(by='Importance', ascending=True)  # Inversez ici pour le tri ascendant

        # Ajoute une réglette pour choisir le nombre de caractéristiques à afficher
        num_features = st.slider('Nombre de facteurs clefs à afficher:', min_value=5, max_value=15, value=10, step=5)
        top_features = feature_importances[-num_features:]  # Prend les derniers éléments au lieu des premiers
        
        
    
        fig = px.bar(top_features, 
                 x='Importance', 
                 y='Feature', 
                 orientation='h',
                 labels={'Feature':'Features', 'Importance':'Importance'},
                 
                 title=f'Top {num_features} des facteurs clefs généraux',
                 color='Importance',
                 color_continuous_scale= 'bluered')  

        fig.update_layout(
            xaxis_title="<b>Niveau d'Importance</b>", 
            yaxis_title="<b>Facteurs clefs</b>"
)

        st.plotly_chart(fig)

    # Option "Comparaison des informations du client"
    if st.sidebar.checkbox("Comparer les informations du client"):
        st.markdown("<h3 style='text-align: left; color: #800020 ;'>Comparaison des informations du client</h3>", unsafe_allow_html=True)
        with st.expander("Cliquez pour afficher les détails"):
            st.write(""" Dans cette section, vous pouvez comparer les données de votre client avec les données de l'ensemble des clients ou avec 
                 les données des clients similaires. Pour chacune des comparaisons, les données des autres clients s'affichent sous forme de statistiques descriptives : 
                count pour compter le nombre de clients, mean pour la moyenne, std pour l'écart-type, min pour la valeur minimale, 25% pour le premier quartile, 
                50% pour la médiane, 75% pour le troisième quartile et max pour la valeur maximale.""")
        
        # Choix du groupe de comparaison
        compare_with = st.radio("Comparer avec :", ["Ensemble des clients", "Clients similaires"])
        
        # Caractéristiques les plus importantes
        importances = model.named_steps['classifier'].feature_importances_
        feature_importances = pd.DataFrame({'Feature': X_sample.columns, 'Importance': importances})
        feature_importances_sorted = feature_importances.sort_values(by='Importance', ascending=False)
        top_features = feature_importances_sorted['Feature'][:10].tolist()  # Top 10 caractéristiques les plus importantes

        # Filtre les données selon les caractéristiques les plus importantes
        X_filtered = X[top_features]
        data_to_predict_filtered = data_to_predict[top_features]

        # Affiche les statistiques descriptives
        if compare_with == "Ensemble des clients":
            st.write("Comparaison avec l'ensemble des clients:")
            desc = X_filtered.describe()
            st.write(desc)
            
            # Informations spécifiques du client par rapport à l'ensemble
            client_info = data_to_predict_filtered.describe()
            st.write(f"Informations pour le client n°{index_selected}:")
            st.write(data_to_predict_filtered)
            
        else:  # Clients similaires
            # Définie les clients similaires (10 plus proches)
            imputer = SimpleImputer(strategy='most_frequent')
            X_filled = pd.DataFrame(imputer.fit_transform(X_filtered), columns=X_filtered.columns)

            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(X_filled, imputer.transform(data_to_predict_filtered))
            closest_indices = np.argsort(distances, axis=0)[:10]
            closest_clients = X_filled.iloc[closest_indices.flatten()]
            
            st.write("Comparaison avec les clients similaires:")
            desc_similar = closest_clients.describe()
            st.write(desc_similar)
            
            # Informations spécifiques du client par rapport aux clients similaires
            client_info = data_to_predict_filtered.describe()
            st.write(f"Informations pour le client n°{index_selected}:")
            st.write(data_to_predict_filtered)

            st.markdown("<br>"*3, unsafe_allow_html=True)

            #Graphique de comparaison
            # Choix de la caractéristique à comparer
            st.markdown("<h4 style='text-align: left; color: black ;'>Comparaison des facteurs clefs</h4>", unsafe_allow_html=True)
            with st.expander("Cliquez pour afficher les détails"):
                st.write(""" Choisissez et affichez sous forme de graphique, un facteur clef à comparer entre les données des clients similaires et celles du client sélectionné.""")
            feature_to_compare = st.selectbox("Choisissez le facteur clef :", top_features)
            fig = plot_parallel_bars(data_to_predict, closest_clients, [feature_to_compare])
            st.plotly_chart(fig)
    

if __name__ == '__main__':
    main()


