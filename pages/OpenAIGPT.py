import os
import json
from datetime import datetime
import streamlit as st
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.document_loaders import WebBaseLoader
from openai import OpenAI, AuthenticationError
