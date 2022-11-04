from fastapi import FastAPI
from pydantic import BaseModel, create_model, BaseConfig
import joblib
import pandas as pd
import uvicorn
import json

app = FastAPI()

features_dict = joblib.load('p7_features_type.joblib')
model = joblib.load('p7_pipeline.joblib')

BaseConfig.arbitrary_types_allowed = True

ClientModel = create_model(
    'ClientClassifierModel',
    **features_dict,
    __base__=BaseModel
)

@app.post('/predict/')
def create_client(client: ClientModel):
    client_data = client.dict()
    client_df = pd.DataFrame([client_data])

    return json.dumps(model.predict_proba(client_df)[:, 1].tolist())


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
