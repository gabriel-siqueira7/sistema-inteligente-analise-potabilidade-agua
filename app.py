import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Qualidade & Potabilidade da Água", layout="wide")

st.title("Sistema Inteligente de Análise de Água")
st.markdown("""
Esta plataforma combina **Análise Estatística Avançada** e **Inteligência Artificial** para avaliar a segurança 
e potabilidade de amostras de água destinadas ao consumo humano.
""")

# Definição padronizada das 9 colunas de recursos (features)
COLUNAS_FEATURES = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

# 2. CARREGAMENTO E LIMPEZA DOS DADOS
@st.cache_data
def carregar_e_tratar_dados():
    try:
        df = pd.read_csv("water_potability.csv")
    except FileNotFoundError:
        np.random.seed(42)
        n = 1000
        dados_mock = {
            'ph': np.random.normal(7.0, 1.5, n),
            'Hardness': np.random.normal(196, 35, n),
            'Solids': np.random.normal(22000, 8000, n),
            'Chloramines': np.random.normal(7.1, 1.5, n),
            'Sulfate': np.random.normal(333, 41, n),
            'Conductivity': np.random.normal(426, 80, n),
            'Organic_carbon': np.random.normal(14.2, 3.3, n),
            'Trihalomethanes': np.random.normal(66.3, 16, n),
            'Turbidity': np.random.normal(3.9, 0.7, n),
            'Potability': np.random.choice([0, 1], size=n, p=[0.6, 0.4])
        }
        df = pd.DataFrame(dados_mock)
        for col in ['ph', 'Sulfate', 'Trihalomethanes']:
            df.loc[df.sample(frac=0.1).index, col] = np.nan

    for col in ['ph', 'Sulfate', 'Trihalomethanes']:
        df[col] = df[col].fillna(df.groupby('Potability')[col].transform('median'))
        
    return df

df = carregar_e_tratar_dados()
df['Situação'] = df['Potability'].map({0: 'Não Potável ❌', 1: 'Potável ✅ Safe'})

# 3. CRIAÇÃO DAS ABAS DE NAVEGAÇÃO
aba_estatistica, aba_simulador = st.tabs(["Análise Estatística (Histórico)", "Simulador Preditivo (Machine Learning)"])

# ==========================================
# ABA 1: ANÁLISE ESTATÍSTICA (PASSADO)
# ==========================================
with aba_estatistica:
    st.header("Análise Exploratória de Dados (EDA)")
    st.markdown("Explore o comportamento físico-químico das amostras históricas registradas em laboratório.")
    
    # KPIs Gerais dentro da aba
    total_amostras = len(df)
    taxa_potavel = (df['Potability'].sum() / total_amostras) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total de Amostras Catalogadas", value=f"{total_amostras:,}")
    with col2:
        st.metric(label="Total de Amostras Seguras", value=f"{df['Potability'].sum():,}")
    with col3:
        st.metric(label="Índice de Potabilidade Geral", value=f"{taxa_potavel:.1f}%")

    st.markdown("---")
    
    # Configuração dos filtros locais para esta aba
    st.subheader("Filtros e Gráficos Interativos")
    parametro = st.selectbox(
        "Selecione um parâmetro laboratorial para detalhar a distribuição estatística:",
        COLUNAS_FEATURES
    )
    
    col_esq, col_dir = st.columns(2)
    
    with col_esq:
        fig_box = px.box(df, x="Situação", y=parametro, color="Situação",
                         title=f"Distribuição Estatística e Outliers de: {parametro.capitalize()}",
                         color_discrete_map={'Não Potável ❌': '#EF553B', 'Potável ✅ Safe': '#636EFA'})
        st.plotly_chart(fig_box, width="stretch")
        
    with col_dir:
        corr = df[COLUNAS_FEATURES + ['Potability']].corr()
        fig_heatmap = px.imshow(corr, text_auto=".2f", aspect="auto",
                                title="Matriz de Afinidade (Correlação de Pearson)",
                                color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_heatmap, width="stretch")

# ==========================================
# ABA 2: SIMULADOR PREDITIVO (FUTURO)
# ==========================================
with aba_simulador:
    st.header("Previsão de Potabilidade em Tempo Real")
    st.markdown("Insira os parâmetros de uma nova amostra coletada em campo para obter o diagnóstico da inteligência artificial.")
    
    # Treinamento do Modelo (Executado de forma otimizada via cache)
    @st.cache_resource
    def treinar_modelo_atualizado(data):
        X = data[COLUNAS_FEATURES]
        y = data['Potability']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)
        acc = accuracy_score(y_test, modelo.predict(X_test))
        return modelo, acc

    modelo_rf, acuracia = treinar_modelo_atualizado(df)
    
    st.info(f"**Status do Algoritmo:** Modelo `Random Forest Classifier` carregado com acurácia de validação de **{acuracia*100:.1f}%**.")
    
    st.markdown("### Parâmetros da Nova Amostra")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        v_ph = st.slider("Nível de pH (Equilíbrio Ácido/Base)", 0.0, 14.0, float(df['ph'].mean()))
        v_hardness = st.slider("Dureza da Água (mg/L de CaCO₃)", 40.0, 320.0, float(df['Hardness'].mean()))
        v_solids = st.slider("Sólidos Totais Dissolvidos - TDS (ppm)", 320.0, 61000.0, float(df['Solids'].mean()))
    with c2:
        v_chloramines = st.slider("Concentração de Cloraminas (ppm)", 0.3, 13.0, float(df['Chloramines'].mean()))
        v_sulfate = st.slider("Concentração de Sulfato (mg/L)", 120.0, 480.0, float(df['Sulfate'].mean()))
        v_cond = st.slider("Condutividade Elétrica (μS/cm)", 180.0, 750.0, float(df['Conductivity'].mean()))
    with c3:
        v_carbon = st.slider("Carbono Orgânico Total - TOC (ppm)", 2.0, 28.0, float(df['Organic_carbon'].mean()))
        v_trihalo = st.slider("Trihalometanos (μg/L)", 0.0, 124.0, float(df['Trihalomethanes'].mean()))
        v_turb = st.slider("Grau de Turbidez (NTU)", 1.4, 7.0, float(df['Turbidity'].mean()))

    st.markdown("---")
    
    if st.button("Calcular Índice de Segurança da Água", type="primary"):
        entrada = np.array([[v_ph, v_hardness, v_solids, v_chloramines, v_sulfate, v_cond, v_carbon, v_trihalo, v_turb]])
        predicao = modelo_rf.predict(entrada)
        probabilidade = modelo_rf.predict_proba(entrada)[0][predicao[0]] * 100
        
        if predicao[0] == 1:
            st.success(f"### Resultado: ÁGUA POTÁVEL!\nO modelo analisou as propriedades moleculares e possui **{probabilidade:.1f}%** de certeza de que esta água está segura para o consumo doméstico.")
        else:
            st.error(f"### Resultado: ÁGUA IMPRÓPRIA PARA CONSUMO!\nO modelo identificou anomalias físico-químicas severas fora dos padrões de saneamento com **{probabilidade:.1f}%** de certeza.")



# FIM DO SCRIPT