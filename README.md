# Avaliação N3 - Ciência de Dados: Previsão de Churn de Clientes

**Alunos:** Israel Schroeder Moreira, Marco Leone Merini e Filipe Luiz Orlamunder

**Data:** 01/12/2025  

**Disciplina:** Ciência de Dados

---

## Visão Geral

Este projeto tem como objetivo desenvolver um modelo preditivo para identificar clientes com alta probabilidade de cancelamento de serviço (Churn) em uma empresa de telecomunicações. O projeto segue o **ciclo completo de Ciência de Dados**, desde a definição do problema até o deploy do modelo.

---

## Estrutura do Projeto

```
/
├── notebooks/
│   └── N3_Avaliacao.ipynb          # Notebook principal com toda a análise e modelagem
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset utilizado (7.043 clientes)
├── scripts/
│   └── predict_churn.py            # Script auxiliar para previsões em produção
├── modelo_final.pkl                # Modelo treinado e serializado (Decision Tree)
├── requirements.txt                # Dependências do projeto
└── README.md                       # Este arquivo
```

---

## Parte 1: O Problema de Negócio (1,0 ponto)

### 1.1 Domínio do Problema
O setor de telecomunicações enfrenta alta competitividade e a perda de clientes (Churn) impacta diretamente a receita. Estudos indicam que **adquirir um novo cliente custa 5-7x mais** do que manter um existente. Portanto, identificar clientes propensos ao cancelamento é crucial para ações proativas de retenção.

### 1.2 Pergunta de Negócio
**"Quais clientes têm maior probabilidade de cancelar o serviço e quais fatores mais influenciam essa decisão?"**

### 1.3 Objetivo do Modelo
Construir um modelo de **Classificação Binária** capaz de prever se um cliente vai cancelar o serviço (`Yes`/1) ou não (`No`/0) com base em seus dados cadastrais e de consumo.

---

## Parte 2: Pipeline de Dados e Arquitetura (1,0 ponto)

### 2.1 Origem e Repositório de Dados
- **Fonte:** Dataset público "Telco Customer Churn" (IBM/Kaggle)
- **Volume:** 7.043 clientes, 21 variáveis
- **Arquitetura:** Data Lake simplificado (Raw → Processed)

### 2.2 Pipeline de Dados (Fluxograma)

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  INGESTÃO   │───▶│   LIMPEZA   │───▶│     EDA     │───▶│ PREPARAÇÃO  │
│             │    │             │    │             │    │             │
│ • CSV       │    │ • Tipos     │    │ • Distrib.  │    │ • Encoding  │
│ • Kaggle    │    │ • NaN       │    │ • Correl.   │    │ • Split     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 2.3 Etapas de ETL
1. **Ingestão:** Leitura do CSV
2. **Limpeza:** Conversão de `TotalCharges`, remoção de 11 nulos
3. **EDA:** Identificou desbalanceamento (73% vs 27%) e features importantes
4. **Preparação:** One-Hot Encoding, split estratificado 80/20

---

## Parte 3: Modelagem e Avaliação (6,0 pontos)

### 3.1 Modelos Treinados
| # | Modelo | Descrição |
|---|--------|-----------|
| 1 | Regressão Logística | Baseline, interpretável |
| 2 | Árvore de Decisão | Alta interpretabilidade |
| 3 | Random Forest | Ensemble robusto |

### 3.2 Métricas Utilizadas

| Métrica | Significado | Relevância para Churn |
|---------|-------------|----------------------|
| **Acurácia** | % total de acertos | Visão geral |
| **Precisão** | TP/(TP+FP) | Evita falsos alarmes |
| **Recall** | TP/(TP+FN) | **CRÍTICO** - captura clientes em risco |
| **F1-Score** | Média harmônica | Equilíbrio |

### 3.3 Resultados Comparativos

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|--------|----------|----------|--------|----------|
| Regressão Logística | 80.4% | 64.8% | 57.2% | 60.8% |
| **Árvore de Decisão** | 77.8% | 58.1% | **59.6%** | 58.8% |
| Random Forest | 78.7% | 62.2% | 51.1% | 56.1% |

**Modelo Escolhido:** Árvore de Decisão (melhor Recall - identifica mais clientes em risco)

---

## Parte 4: Deploy (2,0 pontos)

### 4.1 Salvamento do Modelo
```python
import joblib
joblib.dump(best_model, 'modelo_final.pkl')
```

### 4.2 Uso do Modelo
```python
model = joblib.load('modelo_final.pkl')
prediction = model.predict(novo_cliente)
probabilities = model.predict_proba(novo_cliente)
```

### 4.3 Exemplo de Resultado
- **Previsão:** Cliente NÃO vai cancelar (Churn = 0)
- **Probabilidade:** 99% de permanecer, 1% de cancelar
- **Ação:** Baixo risco, não priorizar retenção

---

## Como Executar

1. Clone este repositório:
   ```bash
   git clone <url-do-repositorio>
   cd cienciadedados
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute o notebook:
   ```bash
   jupyter notebook notebooks/N3_Avaliacao.ipynb
   ```

4. Para previsões em produção:
   ```bash
   python scripts/predict_churn.py
   ```

---

## Conclusão

Este projeto demonstrou o ciclo completo de Ciência de Dados:
- ✅ Definição clara do problema de negócio
- ✅ Pipeline de dados documentado
- ✅ Comparação de 3 modelos com 4 métricas
- ✅ Deploy funcional com salvamento e reutilização

O modelo de **Árvore de Decisão** foi escolhido por maximizar o **Recall**, permitindo identificar o maior número possível de clientes em risco de cancelamento para ações proativas de retenção.
