# contains Flask API's 

from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields

app = Flask(__name__)
