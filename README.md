# Sistema Inteligente de Análise e Potabilidade da Água

[![Streamlit App](https://streamlit.io)](https://potabilidadeagua.streamlit.app)

Este projeto consiste em um dashboard analítico e preditivo focado em Saneamento Ambiental e Saúde Pública. A aplicação combina **Análise Exploratória de Dados (EDA)** com algoritmos de **Machine Learning** para avaliar propriedades físico-químicas laboratoriais e prever se uma amostra de água está segura ou imprópria para o consumo humano.

---

## Funcionalidades e Estrutura da Aplicação

O sistema foi modularizado em duas abas principais de navegação para separar o contexto analítico do preditivo:

### 1. Análise Estatística (Dados Históricos)
 **KPIs em Tempo Real:** Monitoramento do volume de amostras, taxas gerais de potabilidade e volumetria de segurança.
 **Análise de Distribuição (Boxplot Interativo):** Visualização dinâmica de parâmetros químicos divididos por classe de potabilidade, permitindo a identificação visual de *outliers* (valores discrepantes) e assimetrias nas medições.
 **Matriz de Correlação (Pearson):** Mapa de calor interativo que exibe a força de afinidade linear entre diferentes compostos químicos e variáveis físicas da água.

### 2. Simulador Preditivo (Machine Learning)
 **Modelo Ativo:** Treinamento em tempo real do algoritmo de classificação **Random Forest Classifier** com separação rigorosa de dados de treino e validação (80/20).
 **Inferência Interativa:** Interface com controles deslizantes (*sliders*) que simulam uma nova análise laboratorial em tempo real, fornecendo o veredito de segurança acompanhado da probabilidade de certeza do algoritmo.

---

## Diferenciais Técnicos e Engenharia de Dados

Para valorizar este projeto no portfólio, foram aplicadas boas práticas recomendadas para profissionais de dados:
 **Tratamento Avançado de Nulos:** Em vez de descartar dados e reduzir o dataset (`dropna`), foi aplicada uma imputação estatística preenchendo os valores ausentes com a **mediana de cada variável agrupada pela classe de destino (Potabilidade)**. Isso preserva a integridade estatística do modelo.
 **Arquitetura Modular:** O design do dashboard foi projetado para ser independente da origem dos dados. Qualquer base pública tabular de saneamento pode ser integrada à aplicação bastando aplicar um mapeamento prévio de colunas (*pipeline de ETL*).
 **Interface Centrada no Usuário (UI/UX):** Layout construído em tons de branco e azul para refletir a temática de recursos hídricos, utilizando vermelho estritamente como gatilho visual de alerta para amostras contaminadas.

---

## Tecnologias Utilizadas

*   **Linguagem:** Python 3.13
*   **Interface Web:** Streamlit
*   **Manipulação de Dados:** Pandas & Numpy
*   **Visualização Gráfica:** Plotly Express
*   **Inteligência Artificial:** Scikit-Learn (`Random Forest Classifier`, `train_test_split`, `accuracy_score`)

Desenvolvido por Gabriel Siqueira para fins de demonstração técnica de portfólio.
