from os import path, remove
import logging
import logging.config
import json


if path.isfile('python_logging.log'):
    remove('python_logging.log')

with open('advanced01_logging_configuration.json', 'r') as logging_configuration_file:
    config_dict = json.load(logging_configuration_file)

# with open('python_logging_configuration.json', 'r') as logging_configuration_file:
#     config_dict = json.load(logging_configuration_file)


logging.config.dictConfig(config_dict)
# logger.setLevel(logging.INFO)
# logger_handler = logging.FileHandler('python_logging.log')
# logger_handler.setLevel(logging.INFO)
# logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s ')
# logger_handler.setFormatter(logger_formatter)
# logger.addHandler(logger_handler)

logger = logging.getLogger(__name__)
logger.info("Completed configuring logger()!")
