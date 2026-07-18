import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
import logging

# Silenciar logs internos do Prophet que causam conflito no Streamlit Cloud
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

st.set_page_config(layout="wide", page_title="Painel de Séries Temporais")

# 1. Carregar Dados
@st.cache_data
def carregar_dados():
    df = pd.read_csv('df_filtrado.csv')
    df['data'] = pd.to_datetime(df['data'])
    df = df.sort_values(by='data')
    df = df.drop_duplicates(subset=['data'])
    return df

# 2. NOVA ABORDAGEM: Fazer cache apenas da Tabela de Previsão
@st.cache_data
def gerar_previsao(df, dias_futuros=15):
    # Prepara os dados pro Prophet
    df_prophet = df[['data', 'quantidade']].copy()
    df_prophet.columns = ['ds', 'y']
    
    # Treina o modelo isoladamente
    modelo = Prophet()
    modelo.fit(df_prophet)
    
    # Gera as datas futuras e prevê
    future_dates = modelo.make_future_dataframe(periods=dias_futuros)
    forecast = modelo.predict(future_dates)
    
    # Retorna o DataFrame final
    return forecast

def plotar_serie(df, medias_moveis, data_inicio, data_fim):
    df_filtrado = df[(df['data'] >= data_inicio) & (df['data'] <= data_fim)].copy()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtrado['data'], y=df_filtrado['quantidade'], mode='lines+markers', name='Série Histórica'))

    for media in medias_moveis:
        if media == 'Média Móvel de 7 dias':
            df_filtrado['media_movel_7'] = df_filtrado['quantidade'].rolling(window=7).mean()
            fig.add_trace(go.Scatter(x=df_filtrado['data'], y=df_filtrado['media_movel_7'], mode='lines', name='Média 7 dias'))
        elif media == 'Média Móvel de 15 dias':
            df_filtrado['media_movel_15'] = df_filtrado['quantidade'].rolling(window=15).mean()
            fig.add_trace(go.Scatter(x=df_filtrado['data'], y=df_filtrado['media_movel_15'], mode='lines', name='Média 15 dias'))
        elif media == 'Média Móvel de 30 dias':
            df_filtrado['media_movel_30'] = df_filtrado['quantidade'].rolling(window=30).mean()
            fig.add_trace(go.Scatter(x=df_filtrado['data'], y=df_filtrado['media_movel_30'], mode='lines', name='Média 30 dias'))

    fig.update_layout(title='Sadia', xaxis_title='Data', yaxis_title='Quantidade')
    return fig

def main():
    st.title('Painel de Séries Temporais')
    st.markdown('Produtos - BRF')

    st.markdown(
        """
        <style>
            .css-3mn07m { background-color: #f0f0f0; color: black; }
            .css-1bglu7e { background-color: #191970; color: white; }
        </style>
        """, unsafe_allow_html=True
    )

    df_filtrado = carregar_dados()

    filtro_medias_moveis = st.sidebar.checkbox('Filtrar Médias Móveis')
    datas_disponiveis = df_filtrado['data'].dt.date.astype(str).tolist()
    
    indice_data_inicio = st.sidebar.slider('Data de início', 0, len(datas_disponiveis) - 1, 0)
    indice_data_fim = st.sidebar.slider('Data de fim', 0, len(datas_disponiveis) - 1, len(datas_disponiveis) - 1)

    data_inicio = pd.to_datetime(datas_disponiveis[indice_data_inicio])
    data_fim = pd.to_datetime(datas_disponiveis[indice_data_fim])

    medias_moveis = []
    if filtro_medias_moveis:
        st.sidebar.header('Adicionar Médias Móveis')
        if st.sidebar.checkbox('Média Móvel de 7 dias'): medias_moveis.append('Média Móvel de 7 dias')
        if st.sidebar.checkbox('Média Móvel de 15 dias'): medias_moveis.append('Média Móvel de 15 dias')
        if st.sidebar.checkbox('Média Móvel de 30 dias'): medias_moveis.append('Média Móvel de 30 dias')

    # Gráfico histórico normal
    fig = plotar_serie(df_filtrado, medias_moveis, data_inicio, data_fim)
    st.plotly_chart(fig, use_container_width=True)

    # Botão de Previsão
    if st.button('Plotar Previsão para os Próximos 15 Dias'):
        with st.spinner("Treinando modelo e gerando previsão..."):
            forecast = gerar_previsao(df_filtrado, dias_futuros=15)

            fig_forecast = go.Figure()
            # Dados originais
            fig_forecast.add_trace(go.Scatter(x=df_filtrado['data'], y=df_filtrado['quantidade'], mode='lines+markers', name='Série Histórica'))
            
            # Linha de previsão do Prophet
            fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsão do Modelo', line=dict(color='orange')))

            fig_forecast.update_layout(title='Série Histórica vs. Previsão para os Próximos 15 Dias', xaxis_title='Data', yaxis_title='Quantidade')
            
            st.plotly_chart(fig_forecast, use_container_width=True)

if __name__ == "__main__":
    main()
