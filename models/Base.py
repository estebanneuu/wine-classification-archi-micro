from peewee import *
from database import connection


class BaseModel(Model):
    class Meta:
        database = connection
