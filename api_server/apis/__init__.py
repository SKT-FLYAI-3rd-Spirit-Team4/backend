from enum import Enum

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import HTTPException
from pymongo import MongoClient
import os


class MeloDB:
    def __init__(self):
        DB_ID = os.environ.get('DB_ID')
        DB_PASSWORD = os.environ.get('DB_PASSWORD')
        client = MongoClient(f'mongodb://{DB_ID}:{DB_PASSWORD}@mongo', 27017)
        self.melo_db = client['melovision']
        self.melo_users = self.melo_db['users']
        self.melo_babies = self.melo_db['babies']
        self.melo_diaries = self.melo_db['diaries']
        self.melo_letters = self.melo_db['letters']
        self.melo_chats = self.melo_db['chats']
        self.melo_images = self.melo_db['images']
        self.melo_music = self.melo_db['music']


class Sex(str, Enum):
    male = 'male'
    female = 'female'


class Genre(str, Enum):
    classic = 'classic'
    jazz = 'jazz'
    pop = 'pop'
    rock = 'rock'
    hiphop = 'hiphop'


class Instrument(str, Enum):
    piano = 'piano'
    guitar = 'guitar'
    drum = 'drum'
    organ = 'organ'
    clarinet = 'clarinet'


class Speed(str, Enum):
    slow = 'slow'
    medium = 'medium'
    fast = 'fast'


class Duration(str, Enum):
    ten_seconds = '10'
    thirty_seconds = '30'
    one_minute = '60'
    one_minute_thirty_seconds = '90'
    two_minutes = '120'


def object_id_to_str(documents):
    result = []
    for document in documents:
        document['_id'] = str(document['_id'])
        result.append(document)

    return result


def str_to_object_id(string):
    try:
        return ObjectId(string)
    except InvalidId:
        raise HTTPException(status_code=400, detail=f"Invalid ObjectId ({string})")
    # TODO: string 반환 안하게 바꾸기
