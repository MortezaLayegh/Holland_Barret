from src.mlProject import logger 
from src.mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = 'data ingestion training pipeline'

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e