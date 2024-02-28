import os
from mlProject import logger
import pandas as pd
from mlProject.entity.config_entity import DataValidationConfig

class DataValidation:
    """
    A class to perform data validation tasks.

    Attributes:
        config (DataValidationConfig): Configuration object containing data validation settings.
    """

    def __init__(self, config: DataValidationConfig):
        """
        Initializes DataValidation class with provided configuration.

        Args:
            config (DataValidationConfig): Configuration object containing data validation settings.
        """
        self.config = config

    def validate_all_columns(self) -> bool:
        """
        Validates if all expected columns are present in the dataset.

        Returns:
            bool: True if all columns are present, False otherwise.
        """
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            logger.error(f"An error occurred during data validation: {str(e)}")
            raise e
