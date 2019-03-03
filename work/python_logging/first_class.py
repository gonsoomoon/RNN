import logging

class FirstClass(object):
    def __init__(self):
        self.current_number = 0
        self.logger = logging.getLogger(__name__)


    def increment_number(self):
        self.current_number += 1
        self.logger.warning('Incrementing number!')
        self.logger.info("Still incrementing number!!")


    def decrement_number(self):
        self.current_number -= 1

    def clear_number(self):
        self.current_number = 0
        self.logger.warning('Clearing number!')
        self.logger.info("Still clearing number!!")

    def contain_varaibles(self):
        name = 'John'
        self.logger.info("%s raised an error", name)

    def show_captue_exception(self):
        try:
            open('/path/to/does/not/exist', 'rb')
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as e:
            self.logger.error("Failed to open file", exc_info=True)
