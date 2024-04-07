import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# Carregar seus dados (ajustar o caminho conforme necessário)
df_filtrado = pd.read_csv('df_filtrado.csv')
df_filtrado['data'] = pd.to_datetime(df_filtrado['data'])  # Certifique-se de que a coluna de data está no formato correto
df_filtrado = df_filtrado.drop_duplicates(subset=['data'])

# Renomear as colunas para "ds" e "y"
df_filtrado_prophet = df_filtrado[['data', 'quantidade']].copy()
df_filtrado_prophet.columns = ['ds', 'y']

# Remover duplicatas no mesmo dia
df_filtrado_prophet = df_filtrado_prophet.drop_duplicates(subset=['ds'])



# Inicializar o modelo Prophet
model_prophet = Prophet()

# Ajustar o modelo aos dados históricos
model_prophet.fit(df_filtrado_prophet)

# Função para plotar série histórica, médias móveis e previsão
def plotar_serie(df, medias_moveis, data_inicio, data_fim):
    df_filtrado = df[(df['data'] >= data_inicio) & (df['data'] <= data_fim)]
    
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
    st.set_page_config(layout="wide")
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

    # Adicionar barra lateral para seleção de médias móveis e sliders de datas
    filtro_medias_moveis = st.sidebar.checkbox('Filtrar Médias Móveis')
    
    # Converter as datas em uma lista de strings
    datas_disponiveis = df_filtrado['data'].astype(str).tolist()
    
    indice_data_inicio = st.sidebar.slider('Data de início', 0, len(datas_disponiveis) - 1, 0)
    indice_data_fim = st.sidebar.slider('Data de fim', 0, len(datas_disponiveis) - 1, len(datas_disponiveis) - 1)

    data_inicio = pd.to_datetime(datas_disponiveis[indice_data_inicio])
    data_fim = pd.to_datetime(datas_disponiveis[indice_data_fim])

    if filtro_medias_moveis:
        st.sidebar.header('Adicionar Médias Móveis')
        media_movel_7 = st.sidebar.checkbox('Média Móvel de 7 dias')
        media_movel_15 = st.sidebar.checkbox('Média Móvel de 15 dias')
        media_movel_30 = st.sidebar.checkbox('Média Móvel de 30 dias')
        
        # Criar lista com médias móveis selecionadas
        medias_moveis = []
        if media_movel_7:
            medias_moveis.append('Média Móvel de 7 dias')
        if media_movel_15:
            medias_moveis.append('Média Móvel de 15 dias')
        if media_movel_30:
            medias_moveis.append('Média Móvel de 30 dias')

        # Exibir gráfico com médias móveis selecionadas
        fig = plotar_serie(df_filtrado, medias_moveis, data_inicio, data_fim)
        st.plotly_chart(fig)
    else:
        # Exibir gráfico sem médias móveis
        fig = plotar_serie(df_filtrado, [], data_inicio, data_fim)
        st.plotly_chart(fig)

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
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsão'))

        fig_forecast.update_layout(title='Série Histórica vs. Previsão para os Próximos 15 Dias', xaxis_title='Data', yaxis_title='Quantidade')
        st.plotly_chart(fig_forecast)

if __name__ == "__main__":
    main()
