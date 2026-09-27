# ltg-matrix-758211
# 🚀 LTG-Matrix-758211

O **LTG-Matrix-758211** é um projeto de Inteligência Artificial e Deep Learning desenvolvido em **Python 3.12** utilizando o framework **PyTorch**. 

O objetivo deste projeto é construir, treinar e avaliar uma rede neural artificial completa do zero — sem depender de APIs externas de modelos pré-treinados —, aplicando conceitos fundamentais de regressão, cálculo de perda (*loss*) e otimização por *backpropagation*.

---

## 🛠️ Tecnologias e Ferramentas

* **Linguagem:** Python 3.12+
* **Framework Principal:** PyTorch (`torch`)
* **Processamento de Dados:** NumPy
* **Editor:** VS Code
* **Versionamento:** Git & GitHub

---

## 📐 Arquitetura da Rede Neural (`LTGMatrixNet`)

O modelo segue uma estrutura sequencial *feedforward*:

* **Camada de Entrada (Input Layer):** Receptora de vetores numéricos de dimensão 2.
* **Camada Oculta (Hidden Layer):** 8 neurónios com função de ativação **ReLU** (Rectified Linear Unit) para introduzir não-linearidade.
* **Camada de Saída (Output Layer):** 1 neurónio com saída contínua (Regressão).
* **Otimizador:** Adam (`lr=0.01`)
* **Função de Perda:** Erro Quadrático Médio (`MSELoss`)

---

## 📊 Desempenho e Resultados

Durante as 1000 épocas de treino, o modelo reduziu o erro (*loss*) de forma contínua até atingir alta precisão:

```text
Época [200/1000]  - Loss: 0.008893
Época [400/1000]  - Loss: 0.002858
Época [600/1000]  - Loss: 0.000601
Época [800/1000]  - Loss: 0.000090
Época [1000/1000] - Loss: 0.000016
