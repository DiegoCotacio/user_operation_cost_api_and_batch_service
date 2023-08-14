# bigquery_connection.py
from google.cloud import bigquery
from google.oauth2 import service_account
import os

def connect_to_bigquery():

    credentials = service_account.Credentials.from_service_account_file('src/connections/protean-fabric-386717-d6a21dd66382.json')

    # Connect to the BigQuery API using the credentials
    client = bigquery.Client(credentials=credentials)
    
    return client

