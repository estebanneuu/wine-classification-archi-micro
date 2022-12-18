

class Wine():
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
