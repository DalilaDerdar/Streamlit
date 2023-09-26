import streamlit as st


def main():
    
    st.markdown("<h1 style='text-align: center; color: #800020 ;'>PRET A DEPENSER</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Outil de décision d’octroi de crédit</h1>", unsafe_allow_html=True)
    st.markdown('<br>'*3, unsafe_allow_html=True)

    
    st.markdown("<h3 style='text-align: left; color: #5a5a5a;'>Présentation</h3>", unsafe_allow_html=True)
    st.write(""" Cette application est un outil de décision d’octroi de crédit. Il vous aidera dans votre analyse du dossier client dans le cadre 
             d’une demande de prêt. Il vous permettra de définir si oui ou non votre client est éligible à un prêt. 
             Cet outil vous donnera également des informations sur le ou les facteurs qui ont permis d’aboutir à cette décision.
              Ces éléments ainsi que les explications qui suivent pourront être partagés avec votre client pour plus de transparence.""")
    st.markdown('<br>'*2, unsafe_allow_html=True)

    
    st.markdown("<h3 style='text-align: left; color: #5a5a5a;'>Comment fonctionne-t-il?</h3>", unsafe_allow_html=True)
    st.write("""L’outil de décision d'octroi de crédit est basé sur un modèle de machine learning. 
             Nous avons conçu un système d’intelligence artificielle qui analyse différentes informations que le client a accepté de partager avec nous. 
             Cette analyse permet de prédire sa capacité à rembourser son crédit en détectant des facteurs clefs plus faibles 
             dans son dossier que dans d’autres dossiers clients qui remboursent leur crédit.""")
    st.markdown('<br>'*2, unsafe_allow_html=True)


    st.markdown("<h3 style='text-align: left; color: #5a5a5a;'>Comment utiliser l’outil?</h3>", unsafe_allow_html=True)
    st.markdown(""" 
    A la fin de cette explication, cliquez sur l'onglet 'Analyse', dans la barre latérale.<br>
    <br>
    <b>Etape 1</b><br>
    Choisissez le dossier client à analyser dans le menu déroulant.<br>
    <br>
    <b>Etape 2</b><br>
    Lisez le résultat de l’analyse.<br>
    <br>
    <b>Etape 3</b><br>
    Lisez ensuite les éléments du dossier client qui ont permis d’aboutir au résultat d’obtention ou non obtention du prêt. Partagez la décision et ces éléments avec votre client. N’hésitez pas à lui partager les graphiques.
    """, unsafe_allow_html=True)

    st.markdown('<br>' * 9, unsafe_allow_html=True)
    


    st.markdown('###### Application développée par [Dalila Derdar](https://www.linkedin.com/in/daliladerdar)')
if __name__ == '__main__':
    main()