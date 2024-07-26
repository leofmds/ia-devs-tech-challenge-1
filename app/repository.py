from app.database import Session
from app.models import MedicalCost, MedicalCostFile


class MedicalCostRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_file(self, filename: str) -> MedicalCostFile:
        file_record = MedicalCostFile(filename=filename)
        self.session.add(file_record)
        self.session.commit()
        self.session.refresh(file_record)
        return file_record

    def create(self, medical_cost: MedicalCost) -> None:
        self.session.add(medical_cost)
        self.session.commit()

    def retrieve_all(self):
        return self.session.query(MedicalCost).all()

    def retrieve_by_file_id(self, file_id: int):
        return self.session.query(MedicalCost).filter(MedicalCost.file_id == file_id).all()


def get_repository():
    db: Session = Session()
    try:
        repo = MedicalCostRepository(db)
        yield repo
    finally:
        db.close()
