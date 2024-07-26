from pydantic import BaseModel


class Subject(BaseModel):
    age: int
    gender: str
    bmi: float
    children: int
    smoker: str
    region: str


class PredictionRequest(BaseModel):
    file_id: int
    subject: Subject
