import streamlit as st
import base64
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, CatBoost
import joblib
from preprocessing import features


@st.cache(suppress_st_warning=True)
def scoring(test_df, path_to_model, new_cols2, tresh, target_col='pred2_bin'):
    df_test = test_df[['id', 'time', 'x']].copy()
    data2 = pd.DataFrame()
    for ids in list(test_df.id.unique()):
        df = test_df[test_df.id == ids]
        data2 = data2.append(features(df.copy()))
    cb = CatBoostClassifier()
    cb.load_model(path_to_model)
    test_df['probability'] = cb.predict_proba(Pool(data2[new_cols2]))[:,1].astype(float)
    test_df[target_col] = (test_df['probability'] > tresh).astype(int)
    df_test = df_test.merge(test_df[['id', 'time', 'x', 'probability', target_col]], 
                          on =['id', 'time', 'x'], how='left')
    df_test.loc[(df_test[target_col].isnull()), target_col] = 0
    return df_test

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() 
    href = f'<a href="data:file/csv;base64,{b64}" download="Cardio_scoring_model_results.csv">Скачать результаты в формате csv</a>'
    return href

def slider_feats(train, feat, target_col_name):
    try:
        grouped = train.groupby(feat)[target_col_name].mean().to_frame(target_col_name).sort_values(by=target_col_name, ascending=False)
        values = list(grouped.index) 
    except:
        values = sorted(train[feat].unique())
    
    st.write('Результат работы модели для различных ID')
    ids = st.selectbox(f'Выберите {feat} (сортировка в порядке уменьшения числа выявленных аномалий)', values)
    df_samp = train[train['id']==ids].copy()
    df_samp.set_index('time', inplace=True)
    df_samp['Аномалия ритма сердца'] = df_samp['x'] * df_samp[target_col_name].replace(0, np.nan)
    try:
        st.line_chart(df_samp[['x', 'Аномалия ритма сердца']])
    except:
        pass

st.set_page_config("Fit_Predict Cardiospike demo")
st.image("https://i.ibb.co/Vwhhs7J/image.png", width=150)

if st.sidebar.button('Очистить кэш'):
    st.caching.clear_cache()

new_cols2 = joblib.load('models/features')
tresh = 0.393
data_path = 'data/test.csv'
target_col_name = 'prediction'

st.markdown('## Детектор ковидных аномалий на ритмограмме')

options = st.selectbox('Какие данные скорить?',
         ['Тестовый датасет', 'Загрузить новые данные'], index=1)

if options == 'Тестовый датасет':
    df = pd.read_csv('data/test.csv')
    df.sort_values(by=['id', 'time'], inplace=True)
    res = scoring(df, 'models/best_model', new_cols2, tresh, target_col = target_col_name)
    st.markdown('### Скоринг завершён успешно!')

    st.markdown(get_table_download_link(res), unsafe_allow_html=True)
    st.write('Доля класса 1 = ', round(100*res[target_col_name].mean(), 1), ' %')
    slider_feats(res, 'id', target_col_name)
else:
    file_buffer = st.file_uploader(label = 'Выберите датасет')
    if file_buffer:
        try:
            df = pd.read_csv(file_buffer, encoding=None)
        except:
            st.write('Файл некорректен!')
        assert df.shape[1] == 3 or df.shape[1] == 4
        st.markdown('#### Файл корректный!')  
        st.write('Пример данных из файла:')
        st.dataframe(df.sample(3))  
        res = scoring(df, 'models/best_model', new_cols2, tresh, target_col = target_col_name)
        st.markdown('### Скоринг завершён успешно!')

        st.markdown(get_table_download_link(res), unsafe_allow_html=True)
		
	
        st.write('Доля класса 1 = ', round(100*res[target_col_name].mean(), 1), ' %')
        
        slider_feats(res, 'id', target_col_name)
        
        
if st.sidebar.button('Анализ важности переменных модели'):
    st.markdown('#### SHAP важности признаков модели')  
    st.image("https://i.ibb.co/dJBVY5s/importances.png", width=500)
    
if st.sidebar.button('Анализ качества модели'):
    st.markdown('#### Точность модели на train-val выборках:')  
    st.image("https://i.ibb.co/YdmqfKn/catboost.png", width=500)
    st.write("Maximum value F1 = 0.8421 when treshold = 0.393")
    st.image("https://i.ibb.co/RD5DNfG/treshold.png", width=500)
