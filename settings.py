import os
from os.path import join, dirname
from dotenv import load_dotenv


def init():
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
    print("Initialized...")
    # OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

