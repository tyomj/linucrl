import logging.config
import os
import yaml

class Logger:
    """
    **File/console logger**

    - Usage::

        # Creation
        logger = Logger(root_path='%path_to_the_project%', name='EdaLogger', filename='eda.txt')

        # Different levels of importance
        logger.debug('something happened')
        logger.info('something happened')
        logger.warning('something happened')
        logger.error('something happened')
        logger.exception('something happened')
        logger.critical('something happened')
    """

    _logger: logging

    def __init__(self, root_path: str, name: str, filename: str, path: str = None):
        """
        Logger class which is used within the project.

        :param root_path: initializes paths to `root`
        :param name: a functionality name you would like to add in the log line
        :param filename: in which these particular logs should be writen
        :param path: by default logs file is created in '%project$/logs' directory. You may specify a different path to log your experiment.
        The parameter may take both full and relative paths. The relative path should be defined related to the project folder.
        """
        super().__init__()

        self._src_path = root_path
        self._logs_path = os.path.abspath(os.path.join(self._src_path, './logs'))
        self._config_path = os.path.abspath(os.path.join(self._src_path, './config'))

        if path:
            if os.path.isabs(path):
                self._logs_path = path
            else:
                self._logs_path = os.path.abspath(os.path.join(self._src_path, path))

            if not os.path.isdir(self._logs_path):
                os.makedirs(self._logs_path)

            print('New log path is specified {}'.format(self._logs_path))

        try:
            with open(os.path.join(self._config_path, 'logging_config.yml'), 'r') as f:
                config = yaml.safe_load(f.read())
                if filename:
                    config['handlers']['file']['filename'] = os.path.join(self._logs_path, filename)
                logging.config.dictConfig(config)
                if not name:
                    name = __name__
                self._logger = logging.getLogger(name)
        except FileNotFoundError as f:
            print("Unable to open file {}".format(os.path.join(self._config_path, 'logging_config.yml')))
            logging.config.dictConfig(self.__get_default_config__())
            self._logger = logging.getLogger(name)

    def debug(self, msg, args=[], kwargs={}):
        try:
            self._logger.disabled = False
            self._logger.debug(msg.format(args, kwargs))
        except Exception as ex:
            print("Exception: {}".format(ex))
            print(msg.format(args, kwargs))

    def info(self, msg, args=[], kwargs={}):
        try:
            self._logger.disabled = False
            self._logger.info(msg.format(args, kwargs))
        except Exception as ex:
            print("Exception: {}".format(ex))
            print(msg.format(args, kwargs))

    def warning(self, msg, args=[], kwargs={}):
        try:
            self._logger.disabled = False
            self._logger.warning(msg.format(args, kwargs))
        except Exception as ex:
            print("Exception: {}".format(ex))
            print(msg.format(args, kwargs))

    def error(self, msg, args=[], kwargs={}):
        try:
            self._logger.disabled = False
            self._logger.error(msg.format(args, kwargs))
        except Exception as ex:
            print("Exception: {}".format(ex))
            print(msg.format(args, kwargs))

    def exception(self, msg, exc_info, args=[], kwargs={}):
        try:
            self._logger.disabled = False
            self._logger.exception(msg.format(args, kwargs), exc_info)
        except Exception as ex:
            print("Exception: {}".format(ex))
            print(msg.format(args, kwargs))

    def critical(self, msg, args=[], kwargs={}):
        try:
            self._logger.disabled = False
            self._logger.critical(msg.format(args, kwargs))
        except Exception as ex:
            print("Exception: {}".format(ex))
            print(msg.format(args, kwargs))

    @staticmethod
    def __get_default_config__() -> dict:
        result = dict({'version': 1})
        result['formatters'] = {}
        result['formatters']['simple'] = {}
        result['formatters']['simple']['format'] = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        result['handlers'] = {}
        result['handlers']['console'] = {}
        result['handlers']['console']['class'] = "logging.StreamHandler"
        result['handlers']['console']['level'] = logging.DEBUG
        result['handlers']['console']['formatter'] = 'simple'
        result['handlers']['console']['stream'] = 'ext://sys.stdout'
        result['loggers'] = {}
        result['loggers']['sampleLogger'] = {}
        result['loggers']['sampleLogger']['level'] = logging.DEBUG
        result['loggers']['sampleLogger']['handlers'] = ['console']
        result['loggers']['sampleLogger']['propagate'] = False
        result['root'] = {}
        result['root']['level'] = logging.DEBUG
        result['root']['handlers'] = ['console']
        return result