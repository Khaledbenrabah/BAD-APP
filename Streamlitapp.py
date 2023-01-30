import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import streamlit_authenticator as stauth
from PIL import Image
import json
import os
import requests
from streamlit_lottie import st_lottie
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import explained_variance_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import os

#st.markdown("### This is a markdown")
# les couleurs des messages-----------------------------------------------
#st.success("Success")
#st.info("Information")
#st.warning("Warning")
#st.error("Error")


image = Image.open(r'C:\Users\BENRABAH.KHALED\01_pandas\data\Logo_bad_v2.PNG')
image_GCC = Image.open(r'C:\Users\BENRABAH.KHALED\01_pandas\data\logo_gcc.PNG')
col1, col2, col3 = st.sidebar.columns([0.2, 10, 0.2])
#col2.st.image(image, width = 150, caption='Direction Technique GCC ')
col2.image(image, width = 1, use_column_width=True)
#--------------------------------------------------------------------------


names = ['khaled ben rabah','Walid Saiem']
usernames = ['khaled','walid']
passwords = ['123','123']
hashed_passwords = stauth.hasher(passwords).generate()

authenticator = stauth.authenticate(names,usernames,hashed_passwords,
    'some_cookie_name','some_signature_key',cookie_expiry_days=10000)

name, authentication_status = authenticator.login('Login  ','main')
if authentication_status:
    st.write('Bienvenue *%s*' % (name))
    st.title('                  Version Demo                   ' )



elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:

    col4, col5, col6 = st.columns([0.02, 10, 0.02])
    col5.image(image_GCC, width=1, use_column_width=True)


if st.session_state['authentication_status']:
    #logo au niveau side bar

    #st.sidebar.image(r'C:\Users\BENRABAH.KHALED\01_pandas\data\Logo_bad', use_column_width = True)

    # Create a page dropdown
    page = st.selectbox("Choisissez votre page", [ "Menu principal","                    E+                 ","                    C-                 ","       BIM      ","       Données     ","RE2020"," Estimation des déchets  "," Travaux   ","FDES"])


    if page == "Menu principal":
        #st.write('You selected:')


        def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()


        lottie_hello = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_9wpyhdzo.json")

        st_lottie(
            lottie_hello,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",  # medium ; high
            # canvas
            height=None,
            width=None,
            key=None,
        )


    # Display details of page 1
    elif page =="                    E+                 ":
        col7, col8, col9 = st.columns([0.2, 10, 0.2])
        col10, col11, col12 = st.columns([3, 3, 3])
        col8.title('PREDICTION DE LA CONSOMMATION ENERGIE PRIMAIRE kWh ep./m².an  ')
        df = pd.read_excel(r'C:\Users\BENRABAH.KHALED\01_pandas\data\Base_de_donnees_E+C-2.xlsx')


        #'nb_logements', 'nb_occupant', 'duree_chantier','maquette_numerique',

        df = df[['parois_vitrees_uw','departement','s_surf_capt_pv','nb_batiments','usage_principal', 'zone_climatique', 'altitude', 'type_travaux',

       'elements_prefabriques', 'nb_niv_ssol', 'nb_niv_surface',
       'type_fondation', 'planchers_bas_nature', 'type_plancher',
       'planchers_bas_uparoi', 'type_structure_principale',
       'zone_sismique', 'sdp','cep_projet','materiau_principal', 'materiau_remplissage_facade',
       'parois_verticales_type_isolation', 'parois_verticales_uparoi',
       'type_toiture', 'planchers_haut_nature', 'planchers_haut_uparoi',
       'type_menuiserie', 'part_generiques', 'type_pm',
       'type_ventilation_principale', 'generateur_principal_ch',
       'generateur_principal_ecs', 'generateur_principal_fr',
       'vecteur_energie_principal_ch', 'vecteur_energie_principal_ecs',
       'vecteur_energie_principal_fr', 'emetteur_chaud']]
        df = df.dropna()
        Q1 = np.percentile(df['cep_projet'], 25,
                           interpolation='midpoint')

        Q3 = np.percentile(df['cep_projet'], 75,
                           interpolation='midpoint')
        IQR = Q3 - Q1
        mask_outlier_sup = df['cep_projet'] < (Q3 + 1.5 * IQR)
        df = df[mask_outlier_sup]
        mask_outlier_min = df['cep_projet'] > (Q1 - 1.5 * IQR)
        df = df[mask_outlier_min]

        #y_cepmax = df['cep_max']
        #df=df.drop('cep_max',axis=1)

        #y_cepmax = df[['cep_max'].copy()

        #del df['cep_max']

        from sklearn.compose import make_column_selector

        colonne_objet = make_column_selector(dtype_exclude=np.number)
        colonne_num = make_column_selector(dtype_include=np.number)

        df[colonne_objet] = df[colonne_objet].astype('str')

        y = df['cep_projet'].copy()
        del df['cep_projet']
        X = df.copy()
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42)
        #X_train_cepmax, X_test_cepmax, y_train_cepmax, y_test_cepmax = train_test_split(
           # X, y_cepmax, random_state=42)
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.impute import SimpleImputer
        from sklearn.compose import make_column_transformer
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn import preprocessing
        from sklearn.preprocessing import MaxAbsScaler
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import RobustScaler
        from catboost import CatBoostRegressor
        from sklearn.preprocessing import OneHotEncoder

        # encoder
        encoder0 = OneHotEncoder(handle_unknown="ignore")
        encoder1 = OrdinalEncoder(handle_unknown="ignore")

        # Scaler
        scaler0 = "passthrough"
        scaler1 = MaxAbsScaler()
        scaler2 = MinMaxScaler()
        scaler3 = StandardScaler()
        scaler4 = RobustScaler()

        # transformation
        categorical_encoder = encoder0

        categorical_cols = colonne_objet

        numerical_scaler = scaler0
        numerical_cols = colonne_num

        # Regressor

        regressor0 = LinearRegression()
        regressor = DecisionTreeRegressor(random_state=42)
        regressor1 = RandomForestRegressor(max_depth=31, max_features='auto', n_estimators=21, random_state=42)
        regressor2 = GradientBoostingRegressor(max_depth=10, max_features='sqrt', n_estimators=160,
                         random_state=42)
        regressor3 = ExtraTreesRegressor(n_estimators=500, max_features='auto', max_depth=255, random_state=42)
        regressor4 = CatBoostRegressor(n_estimators=132, max_depth=10, random_state=42)

        # selector

        selector = SelectKBest(f_regression, k='all')

        # création pipeline

        preprocessor = make_column_transformer(
            (categorical_encoder, categorical_cols),
            (numerical_scaler, numerical_cols))

        rf1 = make_pipeline(preprocessor, regressor2)
        #rf_cepmax = make_pipeline(preprocessor, regressor2)
        rf1.fit(X ,y)
        #rf_cepmax.fit(X_train_cepmax,y_train_cepmax)
        #y_pred_train1 = rf1.predict(X_train)
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import r2_score
        from sklearn.metrics import median_absolute_error
        #st.write("PRF train MAE: %0.3f" % mean_absolute_error(y_train, y_pred_train1))
        #st.write("PRF train MSE: %0.3f" % mean_squared_error(y_train, y_pred_train1))
        #st.write("PRF train RMSE: %0.3f" % mean_squared_error(y_train, y_pred_train1, squared=False))
        #st.write("PRF train R² : %0.3f" % r2_score(y_train, y_pred_train1))
        #st.write("PRF train Median AR : %0.3f" % median_absolute_error(y_train, y_pred_train1))
        #y_pred_test1 = rf1.predict(X_test)
        #st.write("PRF test MAE: %0.3f" % mean_absolute_error(y_test, y_pred_test1))
        #st.write("PRF test MSE: %0.3f" % mean_squared_error(y_test, y_pred_test1))
        #st.write("PRF test RMSE: %0.3f" % mean_squared_error(y_test, y_pred_test1, squared=False))
        #st.write("PRF test R² : %0.3f" % r2_score(y_test, y_pred_test1))
        #st.write("PRF test Median AR : %0.3f" % median_absolute_error(y_test, y_pred_test1))





        # les données :
        def user_input():
            #parois_vitrees_uw = st.sidebar.number_input('Performance des menuiseries extérierus ')
            Zone_Climatique = st.sidebar.selectbox('zone_climatique', df['zone_climatique'].unique())
            Zone_sismique = st.sidebar.selectbox('zone_sismique', df['zone_sismique'].unique())
            Altitude = st.sidebar.selectbox('altitude', df['altitude'].unique())
            Département = st.sidebar.selectbox('departement', sorted(df['departement'].unique()))
            usage_principal = st.sidebar.selectbox('usage_principal', df['usage_principal'].unique())
            Nombre_des_batiments = st.sidebar.selectbox('nb_batiments', sorted(df['nb_batiments'].unique()))
            type_travaux = st.sidebar.selectbox('type_travaux', df['type_travaux'].unique())
            #nb_logements = st.sidebar.selectbox('nb_logements', sorted(df['nb_logements'].unique()))
            #nb_occupant = st.sidebar.selectbox('nb_occupant', df['nb_occupant'].unique())
            #duree_chantier = st.sidebar.selectbox('duree_chantier', df['duree_chantier'].unique())
            #maquette_numerique = st.sidebar.selectbox('maquette_numerique', df['maquette_numerique'].unique())
            elements_prefabriques = st.sidebar.selectbox('elements_prefabriques', df['elements_prefabriques'].unique())
            nb_niv_ssol = st.sidebar.selectbox('nb_niv_ssol', df['nb_niv_ssol'].unique())
            nb_niv_surface = st.sidebar.selectbox('nb_niv_surface', df['nb_niv_surface'].unique())
            type_fondation = st.sidebar.selectbox('type_fondation', df['type_fondation'].unique())
            planchers_bas_nature = st.sidebar.selectbox('planchers_bas_nature', df['planchers_bas_nature'].unique())
            type_plancher = st.sidebar.selectbox('type_plancher', df['type_plancher'].unique())
            type_structure_principale = st.sidebar.selectbox('type_structure_principale',
                                                             df['type_structure_principale'].unique())
            materiau_principal = st.sidebar.selectbox('materiau_principal', df['materiau_principal'].unique())
            materiau_remplissage_facade = st.sidebar.selectbox('materiau_remplissage_facade',
                                                               df['materiau_remplissage_facade'].unique())
            parois_verticales_type_isolation = st.sidebar.selectbox('parois_verticales_type_isolation',
                                                                    df['parois_verticales_type_isolation'].unique())

            type_toiture = st.sidebar.selectbox('type_toiture', df['type_toiture'].unique())
            planchers_haut_nature = st.sidebar.selectbox('planchers_haut_nature', df['planchers_haut_nature'].unique())
            sdp =st.sidebar.number_input('surface du planchers m²',1000)
                #st.sidebar.selectbox('sdp', df['sdp'].unique())
# MEX------------------------------------------------
            type_menuiserie = st.sidebar.selectbox('type_menuiserie', df['type_menuiserie'].unique())
            type_pm = st.sidebar.selectbox('type_pm', df['type_pm'].unique())
            parois_vitrees_uw = st.sidebar.slider('Uw des vitrage', df['parois_vitrees_uw'].min(), df['parois_vitrees_uw'].max(), 1.3)

# PLACNHER BAS
            planchers_bas_uparoi = st.sidebar.slider('planchers_bas_uparoi', df['planchers_bas_uparoi'].min(), df['planchers_bas_uparoi'].max(), 0.25)

# PLACNHER HAUT
            planchers_haut_uparoi = st.sidebar.slider('planchers_haut_uparoi', df['planchers_haut_uparoi'].min(), df['planchers_haut_uparoi'].max(), 0.25)

# Paroi verticale
            parois_verticales_uparoi = st.sidebar.slider('parois_verticales_uparoi', df['parois_verticales_uparoi'].min(), df['parois_verticales_uparoi'].max(), 0.25)



            part_generiques = st.sidebar.selectbox('part_generiques', df['part_generiques'].unique())


            type_ventilation_principale = st.sidebar.selectbox('type_ventilation_principale',
                                                               df['type_ventilation_principale'].unique())
            generateur_principal_ch = st.sidebar.selectbox('generateur_principal_ch',
                                                           df['generateur_principal_ch'].unique())
            vecteur_energie_principal_ch = st.sidebar.selectbox('vecteur_energie_principal_ch',
                                                                df['vecteur_energie_principal_ch'].unique())
            emetteur_chaud = st.sidebar.selectbox('emetteur_chaud', df['emetteur_chaud'].unique())

            generateur_principal_ecs = st.sidebar.selectbox('generateur_principal_ecs',
                                                            df['generateur_principal_ecs'].unique())
            vecteur_energie_principal_ecs = st.sidebar.selectbox('vecteur_energie_principal_ecs',
                                                                 df['vecteur_energie_principal_ecs'].unique())
            generateur_principal_fr = st.sidebar.selectbox('generateur_principal_fr',
                                                           df['generateur_principal_fr'].unique())


            vecteur_energie_principal_fr = st.sidebar.selectbox('vecteur_energie_principal_fr',
                                                                df['vecteur_energie_principal_fr'].unique())

            s_surf_capt_pv = st.sidebar.slider('s_surf_capt_pv', 0,1200,0)



            #Nom = st.sidebar.selectbox('departement', df['departement'].unique())
            #ubat = st.sidebar.selectbox('ubat', df2['ubat'].unique())
            #type_pm = st.sidebar.selectbox('type_pm', df2['type_pm'].unique())
            #srt = st.sidebar.selectbox('SRT', df2['srt'].unique())
            #shab = st.sidebar.selectbox('SHAB', df2['shab'].unique())
            #planchers_haut_epaisseur_isolant = st.sidebar.selectbox('planchers_haut_epaisseur_isolant ?', df2['planchers_haut_epaisseur_isolant'].unique())
            #materiau_principal = st.sidebar.selectbox('materiau_principal', df2['materiau_principal'].unique())
            #Surface_RT = st.sidebar.selectbox('Surface_RT', df2['Surface RT [m²]'].unique())
            #vecteur_energie_principal_ch = st.sidebar.selectbox('vecteur_energie_principal_ch', df2['vecteur_energie_principal_ch'].unique())
            #Zone_climatique = st.sidebar.selectbox('Zone_climatique', df2['Zone climatique'].unique())
            #surt = st.sidebar.selectbox('surt', df2['surt'].unique())

            #prod_photovoltaique = st.sidebar.selectbox('prod_photovoltaique', df2['prod_photovoltaique'].unique())






            #sepal_lenght = st.sidebar.slider('Sepal long', 4.3, 7.9, 5.3)
            #sepal_width = st.sidebar.slider('Sepal largeury', 2.0, 4.4, 3.1)
            #petal_lenght = st.sidebar.slider('petal long', 1.0, 6.9, 5.3)
            #petal_width = st.sidebar.slider('petal largeur', 0.1, 2.5, 1.3)

            data = { 'parois_vitrees_uw':parois_vitrees_uw,
                     'departement':Département,
                     'nb_batiments':Nombre_des_batiments,
                     'zone_climatique':Zone_Climatique ,
                   'altitude':Altitude ,
                   'zone_sismique':Zone_sismique,
                   'sdp':sdp,
                    'type_travaux':type_travaux,
                  # 'nb_logements':nb_logements, 'nb_occupant':nb_occupant, 'duree_chantier':duree_chantier, 'maquette_numerique':maquette_numerique,
                   'elements_prefabriques':elements_prefabriques, 'nb_niv_ssol':nb_niv_ssol, 'nb_niv_surface':nb_niv_surface,
                   'type_fondation':type_fondation,
                     'planchers_bas_nature':planchers_bas_nature, 'type_plancher':type_plancher,
                   'planchers_bas_uparoi':planchers_bas_uparoi, 'type_structure_principale':type_structure_principale,
                     'materiau_principal': materiau_principal,
                     'materiau_remplissage_facade': materiau_remplissage_facade,
                     'parois_verticales_type_isolation': parois_verticales_type_isolation,
                     'parois_verticales_uparoi': parois_verticales_uparoi,
                     'type_toiture': type_toiture,
                     'planchers_haut_nature': planchers_haut_nature,
                     'planchers_haut_uparoi': planchers_haut_uparoi,
                     'type_menuiserie': type_menuiserie,
                     'part_generiques': part_generiques,

                     'type_pm': type_pm,
                     'type_ventilation_principale': type_ventilation_principale,
                     'generateur_principal_ch': generateur_principal_ch,
                     'generateur_principal_ecs': generateur_principal_ecs,
                     'generateur_principal_fr': generateur_principal_fr,
                     'vecteur_energie_principal_ch': vecteur_energie_principal_ch,
                     'vecteur_energie_principal_ecs': vecteur_energie_principal_ecs,
                     'vecteur_energie_principal_fr': vecteur_energie_principal_fr,

                     'emetteur_chaud': emetteur_chaud,


                     's_surf_capt_pv':s_surf_capt_pv,
                     #'prod_photovoltaique':prod_photovoltaique,
                     'usage_principal':usage_principal,


                    }
            batiment_parametres = pd.DataFrame(data, index=[0])
            return batiment_parametres
        # ----------------------------------------------------------evalution de la performance de modle
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import cross_validate

        #scores = cross_val_score(
        #    rf1, X_train, y_train, cv=5, scoring='neg_mean_squared_error'
        #)
        #st.write(scores.mean())
        #rmse_scores = np.sqrt(-scores)

        #st.write(
        #    f"RMSE: {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}"
        #)
        #st.write(y.mean(), y.max(), y.min())
        #st.write("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

        from sklearn.metrics import mean_absolute_error

        #Y_actual = y
        #Y_Predicted = rf1.predict(X)
        #mape = mean_absolute_error(Y_actual, Y_Predicted) * 100
        #st.write("MAPE",mape)
        # ----------------------------------------------------------evalution de la performance de modle

        variable_df = user_input()
        st.write(variable_df)





        prediction = rf1.predict(variable_df)
        #prediction_cepmax=rf_cepmax.predict(variable_df)
        X2=prediction[0]
        #X2_cepmax=prediction_cepmax[0]
        col11.title(round(X2,1))
        #st.title(round(X2_cepmax, 1))
        #X2DF=pd.DataFrame(X2,columns=['cep'])
        #st.bar_chart(X2DF)

        chart_data = pd.DataFrame(
            np.array((0,0,X2,0,0)),
            columns=["Cep kWh ep./m².an "])

        st.bar_chart(chart_data)








        if st.button('Export_resultat'):
            import pandas as pd
            from io import BytesIO
            from pyxlsb import open_workbook as open_xlsb
            import streamlit as st

            df_resultat = variable_df


            def to_excel(df):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                df.to_excel(writer, index=False, sheet_name='Sheet1')
                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                format1 = workbook.add_format({'num_format': '0.00'})
                worksheet.set_column('A:A', None, format1)
                writer.save()
                processed_data = output.getvalue()
                return processed_data


            df_xlsx = to_excel(df_resultat)
            st.download_button(label='📥 Download Current Result',
                               data=df_xlsx,
                               file_name='df_resultat.xlsx')







        else:
            st.write('**')
    elif page == "                    C-                 ":
        col7, col8, col9 = st.columns([0.2, 10, 0.2])
        col10, col11, col12 = st.columns([3, 3, 3])
        col8.title('PREDICTION de l Indicateur des émissions de Gaz à Effet de Serre sur l’ensemble du cycle de vie   ')
        df = pd.read_excel(r'C:\Users\BENRABAH.KHALED\01_pandas\data\Base_de_donnees_E+C-2.xlsx')

        # 'nb_logements', 'nb_occupant', 'duree_chantier','maquette_numerique',
        Target = 'eges'
        df = df[
            ['parois_vitrees_uw', 'departement', 's_surf_capt_pv', 'nb_batiments', 'usage_principal', 'zone_climatique',
             'altitude', 'type_travaux',

             'elements_prefabriques', 'nb_niv_ssol', 'nb_niv_surface',
             'type_fondation', 'planchers_bas_nature', 'type_plancher',
             'planchers_bas_uparoi', 'type_structure_principale',
             'zone_sismique', 'sdp', 'eges', 'materiau_principal', 'materiau_remplissage_facade',
             'parois_verticales_type_isolation', 'parois_verticales_uparoi',
             'type_toiture', 'planchers_haut_nature', 'planchers_haut_uparoi',
             'type_menuiserie', 'part_generiques', 'type_pm',
             'type_ventilation_principale', 'generateur_principal_ch',
             'generateur_principal_ecs', 'generateur_principal_fr',
             'vecteur_energie_principal_ch', 'vecteur_energie_principal_ecs',
             'vecteur_energie_principal_fr', 'emetteur_chaud']]
        df = df.dropna()
        Q1 = np.percentile(df[Target], 25,
                           interpolation='midpoint')

        Q3 = np.percentile(df[Target], 75,
                           interpolation='midpoint')
        IQR = Q3 - Q1
        mask_outlier_sup = df[Target] < (Q3 + 1.5 * IQR)
        df = df[mask_outlier_sup]
        mask_outlier_min = df[Target] > (Q1 - 1.5 * IQR)
        df = df[mask_outlier_min]

        # y_cepmax = df['cep_max']
        # df=df.drop('cep_max',axis=1)

        # y_cepmax = df[['cep_max'].copy()

        # del df['cep_max']

        from sklearn.compose import make_column_selector

        colonne_objet = make_column_selector(dtype_exclude=np.number)
        colonne_num = make_column_selector(dtype_include=np.number)

        df[colonne_objet] = df[colonne_objet].astype('str')

        y = df[Target].copy()
        del df[Target]
        X = df.copy()
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42)
        # X_train_cepmax, X_test_cepmax, y_train_cepmax, y_test_cepmax = train_test_split(
        # X, y_cepmax, random_state=42)
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.impute import SimpleImputer
        from sklearn.compose import make_column_transformer
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn import preprocessing
        from sklearn.preprocessing import MaxAbsScaler
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import RobustScaler
        from catboost import CatBoostRegressor
        from sklearn.preprocessing import OneHotEncoder

        # encoder
        encoder0 = OneHotEncoder(handle_unknown="ignore")
        encoder1 = OrdinalEncoder(handle_unknown="ignore")

        # Scaler
        scaler0 = "passthrough"
        scaler1 = MaxAbsScaler()
        scaler2 = MinMaxScaler()
        scaler3 = StandardScaler()
        scaler4 = RobustScaler()

        # transformation
        categorical_encoder = encoder0

        categorical_cols = colonne_objet

        numerical_scaler = scaler0
        numerical_cols = colonne_num

        # Regressor

        regressor0 = LinearRegression()
        regressor = DecisionTreeRegressor(random_state=42)
        regressor1 = RandomForestRegressor(max_depth=31, max_features='auto', n_estimators=21, random_state=42)
        regressor2 = GradientBoostingRegressor(max_depth=10, max_features='sqrt', n_estimators=160,
                                               random_state=42)
        regressor3 = ExtraTreesRegressor(n_estimators=500, max_features='auto', max_depth=255, random_state=42)
        regressor4 = CatBoostRegressor(n_estimators=132, max_depth=10, random_state=42)

        # selector

        selector = SelectKBest(f_regression, k='all')

        # création pipeline

        preprocessor = make_column_transformer(
            (categorical_encoder, categorical_cols),
            (numerical_scaler, numerical_cols))

        rf1 = make_pipeline(preprocessor, regressor2)
        # rf_cepmax = make_pipeline(preprocessor, regressor2)
        rf1.fit(X, y)
        # rf_cepmax.fit(X_train_cepmax,y_train_cepmax)
        # y_pred_train1 = rf1.predict(X_train)
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import r2_score
        from sklearn.metrics import median_absolute_error


        # st.write("PRF train MAE: %0.3f" % mean_absolute_error(y_train, y_pred_train1))
        # st.write("PRF train MSE: %0.3f" % mean_squared_error(y_train, y_pred_train1))
        # st.write("PRF train RMSE: %0.3f" % mean_squared_error(y_train, y_pred_train1, squared=False))
        # st.write("PRF train R² : %0.3f" % r2_score(y_train, y_pred_train1))
        # st.write("PRF train Median AR : %0.3f" % median_absolute_error(y_train, y_pred_train1))
        # y_pred_test1 = rf1.predict(X_test)
        # st.write("PRF test MAE: %0.3f" % mean_absolute_error(y_test, y_pred_test1))
        # st.write("PRF test MSE: %0.3f" % mean_squared_error(y_test, y_pred_test1))
        # st.write("PRF test RMSE: %0.3f" % mean_squared_error(y_test, y_pred_test1, squared=False))
        # st.write("PRF test R² : %0.3f" % r2_score(y_test, y_pred_test1))
        # st.write("PRF test Median AR : %0.3f" % median_absolute_error(y_test, y_pred_test1))

        # les données :
        def user_input():
            # parois_vitrees_uw = st.sidebar.number_input('Performance des menuiseries extérierus ')
            Zone_Climatique = st.sidebar.selectbox('zone_climatique', df['zone_climatique'].unique())
            Zone_sismique = st.sidebar.selectbox('zone_sismique', df['zone_sismique'].unique())
            Altitude = st.sidebar.selectbox('altitude', df['altitude'].unique())
            Département = st.sidebar.selectbox('departement', sorted(df['departement'].unique()))
            usage_principal = st.sidebar.selectbox('usage_principal', df['usage_principal'].unique())
            Nombre_des_batiments = st.sidebar.selectbox('nb_batiments', sorted(df['nb_batiments'].unique()))
            type_travaux = st.sidebar.selectbox('type_travaux', df['type_travaux'].unique())
            # nb_logements = st.sidebar.selectbox('nb_logements', sorted(df['nb_logements'].unique()))
            # nb_occupant = st.sidebar.selectbox('nb_occupant', df['nb_occupant'].unique())
            # duree_chantier = st.sidebar.selectbox('duree_chantier', df['duree_chantier'].unique())
            # maquette_numerique = st.sidebar.selectbox('maquette_numerique', df['maquette_numerique'].unique())
            elements_prefabriques = st.sidebar.selectbox('elements_prefabriques', df['elements_prefabriques'].unique())
            nb_niv_ssol = st.sidebar.selectbox('nb_niv_ssol', df['nb_niv_ssol'].unique())
            nb_niv_surface = st.sidebar.selectbox('nb_niv_surface', df['nb_niv_surface'].unique())
            type_fondation = st.sidebar.selectbox('type_fondation', df['type_fondation'].unique())
            planchers_bas_nature = st.sidebar.selectbox('planchers_bas_nature', df['planchers_bas_nature'].unique())
            type_plancher = st.sidebar.selectbox('type_plancher', df['type_plancher'].unique())
            type_structure_principale = st.sidebar.selectbox('type_structure_principale',
                                                             df['type_structure_principale'].unique())
            materiau_principal = st.sidebar.selectbox('materiau_principal', df['materiau_principal'].unique())
            materiau_remplissage_facade = st.sidebar.selectbox('materiau_remplissage_facade',
                                                               df['materiau_remplissage_facade'].unique())
            parois_verticales_type_isolation = st.sidebar.selectbox('parois_verticales_type_isolation',
                                                                    df['parois_verticales_type_isolation'].unique())

            type_toiture = st.sidebar.selectbox('type_toiture', df['type_toiture'].unique())
            planchers_haut_nature = st.sidebar.selectbox('planchers_haut_nature', df['planchers_haut_nature'].unique())
            sdp = st.sidebar.number_input('surface du planchers m²', 1000)
            # st.sidebar.selectbox('sdp', df['sdp'].unique())
            # MEX------------------------------------------------
            type_menuiserie = st.sidebar.selectbox('type_menuiserie', df['type_menuiserie'].unique())
            type_pm = st.sidebar.selectbox('type_pm', df['type_pm'].unique())
            parois_vitrees_uw = st.sidebar.slider('Uw des vitrage', df['parois_vitrees_uw'].min(),
                                                  df['parois_vitrees_uw'].max(), 1.3)

            # PLACNHER BAS
            planchers_bas_uparoi = st.sidebar.slider('planchers_bas_uparoi', df['planchers_bas_uparoi'].min(),
                                                     df['planchers_bas_uparoi'].max(), 0.25)

            # PLACNHER HAUT
            planchers_haut_uparoi = st.sidebar.slider('planchers_haut_uparoi', df['planchers_haut_uparoi'].min(),
                                                      df['planchers_haut_uparoi'].max(), 0.25)

            # Paroi verticale
            parois_verticales_uparoi = st.sidebar.slider('parois_verticales_uparoi',
                                                         df['parois_verticales_uparoi'].min(),
                                                         df['parois_verticales_uparoi'].max(), 0.25)

            part_generiques = st.sidebar.selectbox('part_generiques', df['part_generiques'].unique())

            type_ventilation_principale = st.sidebar.selectbox('type_ventilation_principale',
                                                               df['type_ventilation_principale'].unique())
            generateur_principal_ch = st.sidebar.selectbox('generateur_principal_ch',
                                                           df['generateur_principal_ch'].unique())
            vecteur_energie_principal_ch = st.sidebar.selectbox('vecteur_energie_principal_ch',
                                                                df['vecteur_energie_principal_ch'].unique())
            emetteur_chaud = st.sidebar.selectbox('emetteur_chaud', df['emetteur_chaud'].unique())

            generateur_principal_ecs = st.sidebar.selectbox('generateur_principal_ecs',
                                                            df['generateur_principal_ecs'].unique())
            vecteur_energie_principal_ecs = st.sidebar.selectbox('vecteur_energie_principal_ecs',
                                                                 df['vecteur_energie_principal_ecs'].unique())
            generateur_principal_fr = st.sidebar.selectbox('generateur_principal_fr',
                                                           df['generateur_principal_fr'].unique())

            vecteur_energie_principal_fr = st.sidebar.selectbox('vecteur_energie_principal_fr',
                                                                df['vecteur_energie_principal_fr'].unique())

            s_surf_capt_pv = st.sidebar.slider('s_surf_capt_pv', 0, 1200, 0)

            # Nom = st.sidebar.selectbox('departement', df['departement'].unique())
            # ubat = st.sidebar.selectbox('ubat', df2['ubat'].unique())
            # type_pm = st.sidebar.selectbox('type_pm', df2['type_pm'].unique())
            # srt = st.sidebar.selectbox('SRT', df2['srt'].unique())
            # shab = st.sidebar.selectbox('SHAB', df2['shab'].unique())
            # planchers_haut_epaisseur_isolant = st.sidebar.selectbox('planchers_haut_epaisseur_isolant ?', df2['planchers_haut_epaisseur_isolant'].unique())
            # materiau_principal = st.sidebar.selectbox('materiau_principal', df2['materiau_principal'].unique())
            # Surface_RT = st.sidebar.selectbox('Surface_RT', df2['Surface RT [m²]'].unique())
            # vecteur_energie_principal_ch = st.sidebar.selectbox('vecteur_energie_principal_ch', df2['vecteur_energie_principal_ch'].unique())
            # Zone_climatique = st.sidebar.selectbox('Zone_climatique', df2['Zone climatique'].unique())
            # surt = st.sidebar.selectbox('surt', df2['surt'].unique())

            # prod_photovoltaique = st.sidebar.selectbox('prod_photovoltaique', df2['prod_photovoltaique'].unique())

            # sepal_lenght = st.sidebar.slider('Sepal long', 4.3, 7.9, 5.3)
            # sepal_width = st.sidebar.slider('Sepal largeury', 2.0, 4.4, 3.1)
            # petal_lenght = st.sidebar.slider('petal long', 1.0, 6.9, 5.3)
            # petal_width = st.sidebar.slider('petal largeur', 0.1, 2.5, 1.3)

            data = {'parois_vitrees_uw': parois_vitrees_uw,
                    'departement': Département,
                    'nb_batiments': Nombre_des_batiments,
                    'zone_climatique': Zone_Climatique,
                    'altitude': Altitude,
                    'zone_sismique': Zone_sismique,
                    'sdp': sdp,
                    'type_travaux': type_travaux,
                    # 'nb_logements':nb_logements, 'nb_occupant':nb_occupant, 'duree_chantier':duree_chantier, 'maquette_numerique':maquette_numerique,
                    'elements_prefabriques': elements_prefabriques, 'nb_niv_ssol': nb_niv_ssol,
                    'nb_niv_surface': nb_niv_surface,
                    'type_fondation': type_fondation,
                    'planchers_bas_nature': planchers_bas_nature, 'type_plancher': type_plancher,
                    'planchers_bas_uparoi': planchers_bas_uparoi,
                    'type_structure_principale': type_structure_principale,
                    'materiau_principal': materiau_principal,
                    'materiau_remplissage_facade': materiau_remplissage_facade,
                    'parois_verticales_type_isolation': parois_verticales_type_isolation,
                    'parois_verticales_uparoi': parois_verticales_uparoi,
                    'type_toiture': type_toiture,
                    'planchers_haut_nature': planchers_haut_nature,
                    'planchers_haut_uparoi': planchers_haut_uparoi,
                    'type_menuiserie': type_menuiserie,
                    'part_generiques': part_generiques,

                    'type_pm': type_pm,
                    'type_ventilation_principale': type_ventilation_principale,
                    'generateur_principal_ch': generateur_principal_ch,
                    'generateur_principal_ecs': generateur_principal_ecs,
                    'generateur_principal_fr': generateur_principal_fr,
                    'vecteur_energie_principal_ch': vecteur_energie_principal_ch,
                    'vecteur_energie_principal_ecs': vecteur_energie_principal_ecs,
                    'vecteur_energie_principal_fr': vecteur_energie_principal_fr,

                    'emetteur_chaud': emetteur_chaud,

                    's_surf_capt_pv': s_surf_capt_pv,
                    # 'prod_photovoltaique':prod_photovoltaique,
                    'usage_principal': usage_principal,

                    }
            batiment_parametres = pd.DataFrame(data, index=[0])
            return batiment_parametres


        # ----------------------------------------------------------evalution de la performance de modle
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import cross_validate

        # scores = cross_val_score(
        #    rf1, X_train, y_train, cv=5, scoring='neg_mean_squared_error'
        # )
        # st.write(scores.mean())
        # rmse_scores = np.sqrt(-scores)

        # st.write(
        #    f"RMSE: {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}"
        # )
        # st.write(y.mean(), y.max(), y.min())
        # st.write("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

        from sklearn.metrics import mean_absolute_error

        # Y_actual = y
        # Y_Predicted = rf1.predict(X)
        # mape = mean_absolute_error(Y_actual, Y_Predicted) * 100
        # st.write("MAPE",mape)
        # ----------------------------------------------------------evalution de la performance de modle

        variable_df = user_input()
        st.write(variable_df)

        prediction = rf1.predict(variable_df)
        # prediction_cepmax=rf_cepmax.predict(variable_df)
        X2 = prediction[0]
        # X2_cepmax=prediction_cepmax[0]
        col11.title(round(X2, 1),"kg eq.CO2/m²SDP")
        # st.title(round(X2_cepmax, 1))
        # X2DF=pd.DataFrame(X2,columns=['cep'])
        # st.bar_chart(X2DF)

        chart_data = pd.DataFrame(
            np.array((0, 0, X2, 0, 0)),
            columns=["Eges keqCO2/m²sdp "])

        st.bar_chart(chart_data)



    elif page=="       BIM      ":
        import pandas as pd
        from io import BytesIO
        from pyxlsb import open_workbook as open_xlsb
        import streamlit as st
        dfbim=pd.DataFrame()


        def to_excel(df):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            format1 = workbook.add_format({'num_format': '0.00'})
            worksheet.set_column('A:A', None, format1)
            writer.save()
            processed_data = output.getvalue()
            return processed_data


        df_xlsx = to_excel(dfbim)
        st.download_button(label='📥 Download Current Result',
                           data=df_xlsx,
                           file_name='df_test.xlsx')



    elif page=="       Données     ":
        st.write("hello")
        import time
        import requests

        import streamlit as st
        from streamlit_lottie import st_lottie
        from streamlit_lottie import st_lottie_spinner


        def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()


        lottie_url_hello = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
        lottie_url_download = "https://assets4.lottiefiles.com/private_files/lf30_t26law.json"
        lottie_hello = load_lottieurl(lottie_url_hello)
        lottie_download = load_lottieurl(lottie_url_download)

        st_lottie(lottie_hello, key="hello")

        if st.button("Download"):
            with st_lottie_spinner(lottie_download, key="download"):
                time.sleep(5)
            st.balloons()





#-----------------------------------------------------------------------------------------
elif st.session_state['authentication_status'] == False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] == None:
    st.warning('Please enter your username and password')


