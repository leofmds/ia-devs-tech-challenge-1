from sqlalchemy import Column, String, Integer, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class MedicalCost(Base):
    __tablename__ = 'medical_cost'

    id = Column(Integer, primary_key=True)
    age = Column(Integer)
    gender = Column(String)
    bmi = Column(Float)
    children = Column(Integer)
    smoker = Column(Boolean)
    region = Column(String)
    cost = Column(Float)
    file_id = Column(Integer, ForeignKey('medical_cost_file.id'))

    file = relationship('MedicalCostFile', back_populates='costs')


class MedicalCostFile(Base):
    __tablename__ = 'medical_cost_file'

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)

    costs = relationship('MedicalCost', back_populates='file')
