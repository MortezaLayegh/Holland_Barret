import os
import urllib.request as request
import zipfile
from mlProject import logger
from mlProject.utils.common import get_size
from mlProject.entity.config_entity import DataIngestionConfig
from pathlib import Path


class DataIngestion:
    """
    A class to handle data ingestion tasks.

    Attributes:
        config (DataIngestionConfig): Configuration object containing data ingestion settings.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initializes DataIngestion class with provided configuration.

        Args:
            config (DataIngestionConfig): Configuration object containing data ingestion settings.
        """
        self.config = config

    def download_file(self):
        """
        Downloads a file from a given URL to a local directory.

        If the file already exists locally, it logs the file size.

        Returns:
            None
        """
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        Extracts a zip file into the specified directory.

        Creates the directory if it doesn't exist.

        Returns:
            None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
