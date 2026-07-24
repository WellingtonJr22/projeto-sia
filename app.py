import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(
    layout="wide",
    page_title="Painel de Séries Temporais"
)

# ===========================
# CARREGAR DADOS
# ===========================

@st.cache_data
def carregar_dados():

    df = pd.read_csv("df_filtrado.csv")

    df["data"] = pd.to_datetime(df["data"])

    df = df.sort_values("data")

    df = df.drop_duplicates(subset="data")

    df = df.dropna(subset=["quantidade"])

    return df


# ===========================
# PREVISÃO
# ===========================

@st.cache_data
def gerar_previsao(df, dias_futuros=15):

    serie = df["quantidade"]

    try:

        modelo = ExponentialSmoothing(
            serie,
            trend="add",
            seasonal=None
        )

        ajuste = modelo.fit()

        previsao = ajuste.forecast(dias_futuros)

    except:

        modelo = ExponentialSmoothing(
            serie,
            trend=None,
            seasonal=None
        )

        ajuste = modelo.fit()

        previsao = ajuste.forecast(dias_futuros)

    datas_futuras = pd.date_range(
        start=df["data"].max() + pd.Timedelta(days=1),
        periods=dias_futuros,
        freq="D"
    )

    forecast = pd.DataFrame({
        "ds": datas_futuras,
        "yhat": previsao.values
    })

    return forecast


# ===========================
# HISTÓRICO
# ===========================

def plotar_serie(df, medias_moveis, data_inicio, data_fim):

    df = df[
        (df["data"] >= data_inicio)
        &
        (df["data"] <= data_fim)
    ].copy()

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=df["data"],
            y=df["quantidade"],

            mode="lines+markers",

            name="Histórico"

        )

    )

    if "7" in medias_moveis:

        df["mm7"] = df["quantidade"].rolling(7).mean()

        fig.add_trace(

            go.Scatter(

                x=df["data"],
                y=df["mm7"],

                mode="lines",

                name="MM 7"

            )

        )

    if "15" in medias_moveis:

        df["mm15"] = df["quantidade"].rolling(15).mean()

        fig.add_trace(

            go.Scatter(

                x=df["data"],
                y=df["mm15"],

                mode="lines",

                name="MM 15"

            )

        )

    if "30" in medias_moveis:

        df["mm30"] = df["quantidade"].rolling(30).mean()

        fig.add_trace(

            go.Scatter(

                x=df["data"],
                y=df["mm30"],

                mode="lines",

                name="MM 30"

            )

        )

    fig.update_layout(

        title="Série Histórica",

        xaxis_title="Data",

        yaxis_title="Quantidade"

    )

    return fig

# ===========================
# MAIN
# ===========================

def main():

    st.title("📈 Painel de Séries Temporais")

    st.markdown("### Produtos - BRF")

    df = carregar_dados()

    # ------------------------
    # SIDEBAR
    # ------------------------

    st.sidebar.header("Filtros")

    datas = df["data"].dt.date

    data_inicio = st.sidebar.date_input(
        "Data Inicial",
        value=datas.min(),
        min_value=datas.min(),
        max_value=datas.max()
    )

    data_fim = st.sidebar.date_input(
        "Data Final",
        value=datas.max(),
        min_value=datas.min(),
        max_value=datas.max()
    )

    medias = []

    st.sidebar.markdown("---")
    st.sidebar.subheader("Médias Móveis")

    if st.sidebar.checkbox("7 dias"):
        medias.append("7")

    if st.sidebar.checkbox("15 dias"):
        medias.append("15")

    if st.sidebar.checkbox("30 dias"):
        medias.append("30")

    data_inicio = pd.to_datetime(data_inicio)
    data_fim = pd.to_datetime(data_fim)

    # ------------------------
    # DADOS FILTRADOS
    # ------------------------

    df_filtrado = df[
        (df["data"] >= data_inicio)
        &
        (df["data"] <= data_fim)
    ].copy()

    if df_filtrado.empty:

        st.warning("Nenhum registro encontrado para o período selecionado.")

        return

    # ------------------------
    # KPIs
    # ------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Quantidade Total",
        f"{df_filtrado['quantidade'].sum():,.0f}"
    )

    c2.metric(
        "Média",
        f"{df_filtrado['quantidade'].mean():,.2f}"
    )

    c3.metric(
        "Máximo",
        f"{df_filtrado['quantidade'].max():,.0f}"
    )

    st.divider()

    # ------------------------
    # HISTÓRICO
    # ------------------------

    fig = plotar_serie(
        df_filtrado,
        medias,
        data_inicio,
        data_fim
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ------------------------
    # PREVISÃO
    # ------------------------

    if st.button(
        "🔮 Plotar previsão dos próximos 15 dias",
        use_container_width=True
    ):

        with st.spinner("Calculando previsão..."):

            forecast = gerar_previsao(
                df_filtrado,
                dias_futuros=15
            )


            # ==========================
            # GRÁFICO HISTÓRICO + PREVISÃO
            # ==========================

            fig_previsao = go.Figure()

            # Histórico
            fig_previsao.add_trace(
                go.Scatter(
                    x=df_filtrado["data"],
                    y=df_filtrado["quantidade"],
                    mode="lines+markers",
                    name="Histórico",
                    line=dict(color="royalblue")
                )
            )

            # Previsão
            fig_previsao.add_trace(
                go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat"],
                    mode="lines+markers",
                    name="Previsão",
                    line=dict(color="orange", dash="dash")
                )
            )

            fig_previsao.update_layout(
                title="Histórico + Previsão dos Próximos 15 Dias",
                xaxis_title="Data",
                yaxis_title="Quantidade",
                hovermode="x unified",
                template="plotly_white",
                height=600
            )

            st.plotly_chart(
                fig_previsao,
                use_container_width=True
            )

            # ==========================
            # TABELA DA PREVISÃO
            # ==========================

            tabela = forecast.copy()

            tabela["ds"] = tabela["ds"].dt.strftime("%d/%m/%Y")
            tabela["yhat"] = tabela["yhat"].round(0).astype(int)

            st.subheader("Previsão dos próximos 15 dias")

            st.dataframe(
                tabela.rename(
                    columns={
                        "ds": "Data",
                        "yhat": "Quantidade Prevista"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

            # ==========================
            # DOWNLOAD CSV
            # ==========================

            csv = tabela.rename(
                columns={
                    "ds": "Data",
                    "yhat": "Quantidade Prevista"
                }
            ).to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Baixar previsão em CSV",
                csv,
                file_name="previsao_15_dias.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ------------------------
    # RODAPÉ
    # ------------------------

    st.divider()

    st.caption(
        "Painel desenvolvido em Streamlit • Modelo de previsão utilizando Exponential Smoothing."
    )


# ===========================
# EXECUÇÃO
# ===========================

if __name__ == "__main__":
    main()
