from pkg_resources import resource_filename, Requirement
import logging
import json
import wget
import pafy
import os

class web:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.home_dir = os.path.expanduser("~") + os.path.sep + ".visionlib"
        self.json_data_path = resource_filename(
            Requirement.parse("visionlib"),
            "visionlib" + os.path.sep + "utils" + os.path.sep + "urls.json",
        )

    def get_json(self, name):
        with open(self.json_data_path) as data_file:
            data = json.load(data_file)
        yolo_models = ['Yolov3', 'Tiny-yolov3']
        for dat in data:
            if name in yolo_models and name == dat:

                model_url = data[dat]["model_url"]
                model_name = data[dat]["model_name"]
                cfg_url = data[dat]["cfg_url"]
                cfg_name = data[dat]["cfg_name"]
                label_name = data[dat]["label_name"]
                label_url = data[dat]["label_url"]
                dest_path = data[dat]["dest_path"]
                return (model_url, cfg_url), (model_name, cfg_name), (label_name, label_url), dest_path

            elif name == 'Gender' and name == dat:

                model_url = data[dat]["model_url"]
                model_name = data[dat]["model_name"]
                cfg_url = data[dat]["cfg_url"]
                cfg_name = data[dat]["cfg_name"]
                dest_path = data[dat]["dest_path"]
                return (model_url, cfg_url), (model_name, cfg_name), dest_path

            elif name == 'Keypoint' and name == dat:

                model_url = data[dat]["model_url"]
                model_name = data[dat]["model_name"]
                dest_path = data[dat]["dest_path"]
                return (model_url, ), (model_name, ), dest_path

    def download_file(self, name, labels=False):
        if labels is False:
            urls, names, dest_path = self.get_json(name)
            labels = None
        else:
            urls, names, labels, dest_path = self.get_json(name)

        paths = []

        for url, name in zip(urls, names):
            fpath = self.home_dir + os.path.sep + dest_path + os.path.sep + name
            spath = self.home_dir + os.path.sep + dest_path

            if os.path.exists(spath):
                if os.path.exists(fpath):
                    paths.append(fpath)
                    continue
            else:
                os.mkdir(spath)

            try:
                logging.info("Downloading {0} ".format(name))
                wget.download(url, fpath)
                paths.append(fpath)

            except Exception as e:
                raise Exception(e)

        if labels is not None:

            lpath = self.home_dir + os.path.sep + dest_path + os.path.sep + labels[0]

            if os.path.exists(lpath):
                paths.append(lpath)
            else:
                try:
                    logging.info("Downloading {0} ".format(labels[1]))
                    wget.download(labels[1], lpath)
                    paths.append(lpath)

                except Exception as e:
                    raise Exception(e)

        return paths

    def load_video(self, url=None):
        if url is not None:
            content = pafy.new(url)
            video = content.getbestvideo(preftype='webm')
            return video.url
        else:
            raise Exception("No url given")
