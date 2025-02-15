# 📌 LSTM Stock Price Prediction API

## 📖 Sobre o Projeto

Este projeto desenvolve um modelo de **Redes Neurais Long Short-Term Memory (LSTM)** para prever o preço de fechamento de ações de uma empresa com base nos preços históricos. Além disso, disponibiliza uma **API REST** usando **FastAPI**, permitindo que os usuários solicitem previsões para datas futuras.

## 📌 Funcionalidades

✅ Coleta automática de dados históricos da ação escolhida via Yahoo Finance (`yfinance`).  
✅ Normalização e pré-processamento dos dados para treinamento do modelo LSTM.  
✅ Construção e treinamento do modelo LSTM usando **TensorFlow/Keras**.  
✅ Implementação de uma **API RESTful** com **FastAPI**, onde o usuário pode fornecer um símbolo de ação e uma data futura para prever o preço.  
✅ Deploy gratuito usando **Render** para acesso público à API.

## 📁 Estrutura do Projeto

```
📂 lstm-stock-prediction
│── 📜 lstm_stock_model.h5 # Último modelo treinado
│── 📜 main.py             # API FastAPI para servir o modelo
│── 📜 model_training.py   # Script de treinamento do modelo LSTM
│── 📜 requirements.txt    # Dependências do projeto
│── 📜 Dockerfile          # Configuração para deploy com Docker
│── 📜 README.md           # Documentação do projeto
```

## 🔧 Instalação e Execução

### 1️⃣ **Clonar o Repositório**

```bash
git clone https://github.com/seu-usuario/lstm-stock-prediction.git
cd lstm-stock-prediction
```

### 2️⃣ **Criar e Ativar um Ambiente Virtual**

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 3️⃣ **Instalar Dependências**

```bash
pip install -r requirements.txt
```

### 4️⃣ **Treinar o Modelo**

```bash
python model_training.py
```

Isso criará um arquivo `lstm_stock_model.h5`, que será usado para previsões na API.

### 5️⃣ **Rodar a API Localmente**

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Agora, a API estará disponível em `http://127.0.0.1:8000`.

### 6️⃣ **Testar a API**

**Usando o navegador:**

- Acesse `http://127.0.0.1:8000/docs` para testar via Swagger.

**Via terminal (cURL):**

```bash
curl -X 'POST' 'http://127.0.0.1:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{"symbol": "AAPL", "date": "2024-07-30"}'
```

**Via Python:**

```python
import requests
url = "http://127.0.0.1:8000/predict/"
data = {"symbol": "AAPL", "date": "2024-07-30"}
response = requests.post(url, json=data)
print(response.json())
```

## 🚀 **Deploy Gratuito no Render**

Para hospedar a API, siga estes passos:

1. Suba o código para um repositório no GitHub.
2. Acesse [https://render.com](https://render.com) e crie uma conta.
3. Crie um novo serviço **Web Service** e conecte seu repositório.
4. Configure os comandos:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api:app --host 0.0.0.0 --port 10000`
5. Aguarde o deploy e acesse a API pelo link gerado.

## 📌 **Tecnologias Utilizadas**

- **Python 3.10+**
- **TensorFlow/Keras** (Deep Learning)
- **Yahoo Finance (yfinance)** (Coleta de dados)
- **FastAPI** (Criação da API)
- **Uvicorn** (Servidor ASGI)
- **scikit-learn** (Pré-processamento e métricas)
- **Render** (Hospedagem gratuita da API)

## 📜 **Licença**

Este projeto é de código aberto e pode ser usado para fins educacionais e acadêmicos. 🎓🚀

---

👨‍💻 **Desenvolvido por:** \
📌 GitHub: [danilommarano](https://github.com/danilommarano)  
📧 Contato: danilo.m.marano@gmail.com
