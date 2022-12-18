from .Base import BaseModel
from peewee import *


class Wine(BaseModel):
    id: int = PrimaryKeyField(null=False)
    fixed_acidity: float = FloatField()
    volatile_acidity: float = FloatField()
    citric_acid: float = FloatField()
    residual_sugar: float = FloatField()
    chlorides: float = FloatField()
    free_sulfur_dioxide: int = IntegerField()
    total_sulfur_dioxide: int = IntegerField()
    density: float = FloatField()
    pH: float = FloatField()
    sulphates: float = FloatField()
    alcohol: float = FloatField()

    class Meta:
        db_table = 'wine'

    async def create_wine(self, fixed_acidity: float, volatile_acidity: float, citric_acid: float,
                    residual_sugar: float, chlorides: float, free_sulfur_dioxide: int, total_sulfur_dioxide: int,
                    density: float, ph: float, sulphates: float, alcohol: float):
        wine_object = Wine(
            fixed_acidity=fixed_acidity,
            volatile_acidity=volatile_acidity,
            citric_acid=citric_acid,
            residual_sugar=residual_sugar,
            chlorides=chlorides,
            free_sulfur_dioxide=free_sulfur_dioxide,
            total_sulfur_dioxide=total_sulfur_dioxide,
            density=density,
            pH=ph,
            sulphates=sulphates,
            alcohol=alcohol
        )
        wine_object.save()
        return wine_object

    def get_wine(self, uid: int):
        return Wine.filter(Wine.id == uid).first()
    def get_all_wine(self):
        return list(Wine.select())

    def list_wine(self, skip: int = 0, limit: int = 100):
        return list(Wine.select().offset(skip).limit(limit))

    def delete_wine(self, uid: int):
        return Wine.delete().where(Wine.uid == uid).execute()
