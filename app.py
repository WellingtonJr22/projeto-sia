import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# 1. Configuração da página DEVE ser o primeiro comando do Streamlit
st.set_page_config(layout="wide", page_title="Painel de Séries Temporais")

# 2. CACHE NOS DADOS: Evita ler o CSV toda vez que você mexer no slider
@st.cache_data
def carregar_dados():
    df = pd.read_csv('df_filtrado.csv')
    df['data'] = pd.to_datetime(df['data'])
    # Garantir que os dados estão ordenados por data para o slider funcionar
    df = df.sort_values(by='data')
    df = df.drop_duplicates(subset=['data'])
    return df

# 3. CACHE NO MODELO: Treina o Prophet apenas uma vez quando o app inicia
@st.cache_resource
def treinar_prophet(df):
    df_prophet = df[['data', 'quantidade']].copy()
    df_prophet.columns = ['ds', 'y']
    modelo = Prophet()
    modelo.fit(df_prophet)
    return modelo

# Função para plotar série histórica, médias móveis e previsão
def plotar_serie(df, medias_moveis, data_inicio, data_fim):
    # O .copy() aqui é vital para evitar o erro 'SettingWithCopyWarning' do Pandas
    df_filtrado = df[(df['data'] >= data_inicio) & (df['data'] <= data_fim)].copy()
    
    fig = go.Figure()
    
    # Adicionar série histórica
    fig.add_trace(go.Scatter(x=df_filtrado['data'], y=df_filtrado['quantidade'], mode='lines+markers', name='Série Histórica'))

    # Adicionar médias móveis selecionadas
    for media in medias_moveis:
        if media == 'Média Móvel de 7 dias':
            df_filtrado['media_movel_7'] = df_filtrado['quantidade'].rolling(window=7).mean()
            fig.add_trace(go.Scatter(x=df_filtrado['data'], y=df_filtrado['media_movel_7'], mode='lines', name='Média Móvel de 7 dias'))
        elif media == 'Média Móvel de 15 dias':
            df_filtrado['media_movel_15'] = df_filtrado['quantidade'].rolling(window=15).mean()
            fig.add_trace(go.Scatter(x=df_filtrado['data'], y=df_filtrado['media_movel_15'], mode='lines', name='Média Móvel de 15 dias'))
        elif media == 'Média Móvel de 30 dias':
            df_filtrado['media_movel_30'] = df_filtrado['quantidade'].rolling(window=30).mean()
            fig.add_trace(go.Scatter(x=df_filtrado['data'], y=df_filtrado['media_movel_30'], mode='lines', name='Média Móvel de 30 dias'))

    # Configurar layout do gráfico
    fig.update_layout(title='Sadia', xaxis_title='Data', yaxis_title='Quantidade')
    return fig

def main():
    st.title('Painel de Séries Temporais')
    st.markdown('Produtos - BRF')

    # Adicionar CSS para personalização
    st.markdown(
        """
        <style>
            .css-3mn07m {
                background-color: #f0f0f0;
                color: black;
            }
            .css-1bglu7e {
                background-color: #191970;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Carregar dados e treinar o modelo usando o Cache
    df_filtrado = carregar_dados()
    model_prophet = treinar_prophet(df_filtrado)

    # Adicionar barra lateral para seleção de médias móveis e sliders de datas
    filtro_medias_moveis = st.sidebar.checkbox('Filtrar Médias Móveis')
    
    # Converter as datas em uma lista de strings (pegando apenas a data, sem horas)
    datas_disponiveis = df_filtrado['data'].dt.date.astype(str).tolist()
    
    indice_data_inicio = st.sidebar.slider('Data de início', 0, len(datas_disponiveis) - 1, 0)
    indice_data_fim = st.sidebar.slider('Data de fim', 0, len(datas_disponiveis) - 1, len(datas_disponiveis) - 1)

    data_inicio = pd.to_datetime(datas_disponiveis[indice_data_inicio])
    data_fim = pd.to_datetime(datas_disponiveis[indice_data_fim])

    # Criar lista com médias móveis selecionadas
    medias_moveis = []
    if filtro_medias_moveis:
        st.sidebar.header('Adicionar Médias Móveis')
        if st.sidebar.checkbox('Média Móvel de 7 dias'):
            medias_moveis.append('Média Móvel de 7 dias')
        if st.sidebar.checkbox('Média Móvel de 15 dias'):
            medias_moveis.append('Média Móvel de 15 dias')
        if st.sidebar.checkbox('Média Móvel de 30 dias'):
            medias_moveis.append('Média Móvel de 30 dias')

    # Exibir gráfico da série histórica
    fig = plotar_serie(df_filtrado, medias_moveis, data_inicio, data_fim)
    st.plotly_chart(fig, use_container_width=True)

    # Botão para plotar previsão dos próximos 15 dias
    if st.button('Plotar Previsão para os Próximos 15 Dias'):
        # Criar um DataFrame com as datas dos próximos 15 dias para a previsão
        future_dates = pd.date_range(start=data_fim + pd.DateOffset(days=1), periods=15)
        future_dates_df = pd.DataFrame({'ds': future_dates})

        # Fazer a previsão para os próximos 15 dias
        forecast = model_prophet.predict(future_dates_df)

        # Adicionar previsão para os próximos 15 dias ao gráfico
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=df_filtrado['data'], y=df_filtrado['quantidade'], mode='lines+markers', name='Série Histórica'))
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsão (15 dias)'))

        fig_forecast.update_layout(title='Série Histórica vs. Previsão para os Próximos 15 Dias', xaxis_title='Data', yaxis_title='Quantidade')
        st.plotly_chart(fig_forecast, use_container_width=True)

if __name__ == "__main__":
    main()
