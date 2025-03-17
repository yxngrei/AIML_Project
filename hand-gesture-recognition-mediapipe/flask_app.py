from flask import Flask
from app import main_track

application = Flask(__name__)

@application.route('/')
def home():
    main_track()

if __name__ == '__main__':
    application.run(debug=True)