import wget
import logging
import sys
import os


class web:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.home_dir = os.path.expanduser('~') + os.path.sep + '.visionlib'
        pass

    def download_file(self, url, file_name):
        full_path = self.home_dir + os.path.sep + file_name

        if os.path.exists(self.home_dir):
            pass
        else:
            os.mkdir(self.home_dir)
        if os.path.exists(full_path):
            return full_path

        else:
            try:
                logging.info("Downloading {0} ".format(file_name))
                wget.download(url, full_path)
                return full_path
            except Exception as e:
                logging.fatal("Something went wrong during download")
                logging.fatal("Try again later")
                sys.exit(1)
