#from decouple import config
import os
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


class Config:
    pass

class DevelopmentConfig(Config):
    DEBUG = True
    SECRET_KEY = 'TPmi4aLWRbyVq8zu9v82dWYW1'
    SESSION_TYPE = 'filesystem'
    UPLOAD_FOLDER = os.path.join(APP_ROOT, 'temp')


class ProductionConfig(Config):
    DEBUG = True
    #SQLALCHEMY_DATABASE_URI = config('DATABASE_URL', default='localhost')
    #SQLALCHEMY_TRACK_MODIFICATIONS = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig
}