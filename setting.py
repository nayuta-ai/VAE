import os
from os.path import dirname, join

from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)

API_KEY = os.environ.get("API_KEY")
WORKSPACE = os.environ.get("WORKSPACE")
