import csv
import io
import os

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from starlette.responses import FileResponse

from app.models import MedicalCost
from app.repository import MedicalCostRepository, get_repository
from app.database import init_db
from app.requests import PredictionRequest

app = FastAPI()

app.add_event_handler("startup", init_db)

PLOT_DIR = "plot"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

COLUMN_ORDER = ['age', 'gender', 'bmi', 'children', 'smoker', 'region']

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...), repo: MedicalCostRepository = Depends(get_repository)):
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    content = await file.read()
    csv_reader = csv.DictReader(io.StringIO(content.decode('utf-8')))

    file_record = repo.create_file(file.filename)

    for row in csv_reader:
        medical_cost = MedicalCost(
            age=int(row['age']),
            gender=row['gender'],
            bmi=float(row['bmi']),
            children=int(row['children']),
            smoker=row['smoker'].lower() == 'yes',
            region=row['region'],
            cost=float(row['cost']),
            file_id=file_record.id,
        )
        repo.create(medical_cost)

    return {
        "message": "CSV data successfully uploaded",
        "file_id": file_record.id,
    }


@app.get("/train/{file_id}")
async def train(file_id: int, repo: MedicalCostRepository = Depends(get_repository)):
    medical_costs = repo.retrieve_by_file_id(file_id)

    if not medical_costs:
        raise HTTPException(status_code=404, detail="No medical costs found for the given file ID.")

    preprocessor, x, y = await create_preprocessor(medical_costs)

    model, x_test, x_train, y_test, y_train = await create_and_train_model(file_id, preprocessor, x, y)

    y_pred = model.predict(x_test)

    await save_plot(file_id, y_pred, y_test)

    mse, named_pvalues, r2 = await get_statistics(model, preprocessor, x_test, x_train, y_pred, y_test, y_train)

    return {
        "mean_squared_error": mse,
        "r2_score": r2,
        "p_values": named_pvalues
    }


async def create_and_train_model(file_id, preprocessor, x, y):
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)
    model_path = os.path.join(MODEL_DIR, f"model_{file_id}.pkl")
    joblib.dump(model, model_path)
    return model, x_test, x_train, y_test, y_train


async def create_preprocessor(medical_costs):
    df = pd.DataFrame([cost.__dict__ for cost in medical_costs])
    df = df.dropna()
    if df.empty:
        raise HTTPException(status_code=400, detail="DataFrame is empty after cleaning.")
    categorical_features = ['gender', 'smoker', 'region']
    numerical_features = ['age', 'bmi', 'children']
    x = df[numerical_features + categorical_features][COLUMN_ORDER]
    y = df['cost']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor, x, y


async def get_statistics(model, preprocessor, x_test, x_train, y_pred, y_test, y_train):
    mse = mean_squared_error(y_test, y_pred)
    r2 = model.score(x_test, y_test)
    feature_names = preprocessor.get_feature_names_out()
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_train_with_const = sm.add_constant(pd.DataFrame(x_train_transformed, columns=feature_names))
    x_train_with_const.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    model_with_const = sm.OLS(y_train, x_train_with_const).fit()
    named_pvalues = pd.Series(model_with_const.pvalues, index=x_train_with_const.columns)
    return mse, named_pvalues, r2


async def save_plot(file_id, y_pred, y_test):
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(y_test)), y_test, label='Actual Cost', color='blue', linestyle='-', marker='o')
    plt.plot(np.arange(len(y_pred)), y_pred, label='Predicted Cost', color='orange', linestyle='--', marker='x')
    plt.xlabel('Index')
    plt.ylabel('Cost')
    plt.title('Actual vs Predicted Cost')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/plot_{file_id}.png')
    plt.close()


@app.post("/predict")
async def predict_cost(prediction_request: PredictionRequest):
    file_id = prediction_request.file_id
    subject = prediction_request.subject

    model_path = os.path.join(MODEL_DIR, f"model_{file_id}.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found for the given file ID.")

    model = joblib.load(model_path)

    input_data = pd.DataFrame([{
        'age': subject.age,
        'bmi': subject.bmi,
        'children': subject.children,
        'gender': subject.gender,
        'smoker': subject.smoker == "yes",
        'region': subject.region
    }])[COLUMN_ORDER]

    predicted_cost = model.predict(input_data)[0]

    return {
        "predicted_cost": "${:,.2f}".format(predicted_cost),
    }


@app.get("/report/{file_id}")
async def report(file_id: int):
    plot_file = f"plot/plot_{file_id}.png"

    # Serve the plot image
    return FileResponse(plot_file, media_type="image/png")

