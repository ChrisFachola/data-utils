# bigquery_utils.py

import os

from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account


def get_bigquery_client(credentials_path: str = None) -> bigquery.Client:
    """
    Creates a BigQuery client using credentials from environment variables
    or a specified credentials file path.

    Args:
        credentials_path (str, optional): Path to the service account credentials JSON file.
            If not provided, will look for GOOGLE_APPLICATION_CREDENTIALS in environment variables.

    Returns:
        bigquery.Client: Authenticated BigQuery client

    Raises:
        ValueError: If credentials path is not found or environment variable is not set
    """
    load_dotenv()  # Load environment variables from .env file

    # If no specific path is provided, try to get it from environment variable
    if credentials_path is None:
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not credentials_path:
        raise ValueError(
            "No credentials provided. Either pass credentials_path or set "
            "GOOGLE_APPLICATION_CREDENTIALS environment variable."
        )

    if not os.path.exists(credentials_path):
        raise ValueError(f"Credentials file not found at: {credentials_path}")

    # Create credentials object
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    # Return authenticated client
    return bigquery.Client(credentials=credentials)
