# Deployment guide

## Local Streamlit demo

```bash
python -m pip install -r requirements.txt
streamlit run deployment/streamlit_app.py
```

Open the local URL shown in the terminal, enter customer details, and click **Predict churn risk**.

## FastAPI service

```bash
uvicorn deployment.api:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation. Test `POST /predict` using `sample_request.json`.

## Docker

```bash
docker build -t churn-risk-studio .
docker run -p 8501:8501 churn-risk-studio
```

## Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Sign in at Streamlit Community Cloud.
3. Choose the repository and set the main file to `deployment/streamlit_app.py`.
4. Deploy and test several valid and invalid inputs.

Official deployment reference: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Production checklist

- Pin and reproduce the model environment.
- Load only trusted joblib artifacts.
- Add authentication, request logging, rate limits, and HTTPS.
- Monitor input drift, prediction rate, delayed labels, subgroup metrics, latency, and errors.
- Define retraining and rollback triggers.
