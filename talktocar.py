from os import path as osp

import cv2
import json
import os.path as osp
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from typing import List, Tuple, Dict
import os

class Command:
    def __init__(
        self,
        t2c,
        scene_token: str,
        frame_token: str,
        box: Box,
        command: str,
        command_token: str,
        referred_box_2d: list,
        box_token: str = None,
        slim_dataset: bool = False,
    ):
        """
        :param frame_token:
        :param scene_token:
        :param box:
        """
        self.scene_token = scene_token
        self.frame_token = frame_token
        self.box = box
        self.text = command
        self.box_token = box_token
        self.command_token = command_token
        self.t2c = t2c
        self.slim_dataset = slim_dataset
        self.referred_box_2d = referred_box_2d
        if not slim_dataset:
            self.sd_rec = self.t2c.get("sample_data", self.frame_token)
            _, _, self.camera_intrinsic = self.t2c.get_sample_data(self.sd_rec["token"])

    def __repr__(self):
        """
        Get the textual representation of the command
        :return: str
        """
        return f"""Scene token: {self.scene_token},\nFrame token: {self.frame_token},\nBox:         {self.box}"""

    def to_json(self):
        """
        :return: a dictionary containing the information of the commmand
        """
        js = {
            "scene_token": self.scene_token,
            "frame_token": self.frame_token,
            "box": self.box,
            "text": self.text,
            "box_token": self.box_token,
        }
        return json.dumps(js)

    def get_command_token(self):
        """
        : return: the command token
        """
        return self.command_token

    def get_2d_bbox(self):
        """
        Converts a 3D bounding box to a 2D bounding box
        :return: A list of a bounding box as [x1, y1, width, height]
        """

        ## Below is some old code that we used to convert 3D boxes to 2d
        ## We leave it here for future reference.
        # # Get translated corners
        # b = np.zeros((900, 1600, 3))
        #
        # self.box.render_cv2(
        #     b,
        #     view=self.camera_intrinsic,
        #     normalize=True,
        #     colors=((0, 0, 255), (0, 0, 255), (0, 0, 255)),
        # )
        # y, x = np.nonzero(b[:, :, 0])
        #
        # x1, y1, x2, y2 = map(int, (x.min(), y.min(), x.max(), y.max()))
        # x1 = max(0, x1)
        # y1 = max(0, y1)
        # x2 = min(1600, x2)
        # y2 = min(900, y2)
        # return [x1, y1, x2 - x1, y2 - y1]

        return self.referred_box_2d

    def get_image_path(self):
        """
        :return:
        """
        # Get records from DB
        sd_rec = self.t2c.get("sample_data", self.frame_token)

        # Get data from DB
        im_path, _, _ = self.t2c.get_sample_data(sd_rec["token"])
        return im_path

class Talk2CarBase:
    def __init__(self, split, commands_root, slim):
        self.version = split
        self.commands_root = commands_root
        self.commands = []
        self.lookup = {}
        self.max_command_length = 0
        self.slim = slim

        (
            self.commands,
            self.lookup,
            self.max_command_length,
        ) = self.load_commands()

    def __len__(self):
        return len(self.commands)

    def change_version(self, new_version):
        self.version = new_version
        self.commands, self.lookup, self.max_command_length = self.load_commands()

    def __load_t2c_table__(self, table_name):
        with open(
            osp.join(
                osp.join(self.commands_root, "commands"),
                "{}_commands.json".format(table_name),
            )
        ) as f:
            table = json.load(f)
        return table

    def load_commands(self) -> [List, Dict]:
        """
        :return:
        """
        t2c = self.__load_t2c_table__(self.version)
        cmnds = t2c["commands"]
        self.scene_tokens = t2c["scene_tokens"]
        ret = []
        lookup = {}
        max_length = 0

        # Get data from DB
        for c in cmnds:
            scene_token = c["scene_token"]
            if self.version == "train" or self.version == "val":
                box = Box(
                    c["translation"],
                    c["size"],
                    Quaternion(c["rotation"]),
                    name=c["obj_name"],
                )
            else:
                box = None
            command_token = c["command_token"]
            command_text = c["command"]

            cmd = Command(
                self,
                scene_token,
                c["sample_token"],
                box,
                command_text,
                command_token,
                referred_box_2d=c.get("2d_box", None),
                box_token=c.get("box_token", None),
                slim_dataset=self.slim
            )
            ret.append(cmd)

            if len(command_text) > max_length:
                max_length = len(command_text)

            if scene_token not in lookup:
                lookup[scene_token] = []

            lookup[scene_token].append(cmd)

        return ret, lookup, max_length

    def get_commands(self) -> List:
        """
        :return: returns all existing commands
        """
        if self.commands:
            return self.commands
        else:
            raise Exception(
                "Commands are empty are you sure you supplied a valid version? Supplied version: {}".format(
                    self.version
                )
            )

class Talk2Car(NuScenes, Talk2CarBase):
    img_mean = [0.3950, 0.4004, 0.3906]
    img_std = [0.2115, 0.2068, 0.2164]

    def __init__(
        self,
        split: str = "train",
        root: str = "../datasets/nuScenes/data/sets/nuscenes",
        commands_root = None,
        verbose: bool = False,
    ):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param split: Version to load (e.g. "v1.0-trainval", ...).
        :param root: Path to the tables and data.
        :param commands_root: Path to the command data. If None, will use the path given to the root param.
        :param verbose: Whether to print status messages during load.
        """
        commands_root = commands_root if commands_root else root

        NuScenes.__init__(version="v1.0-trainval", dataroot=root, verbose=verbose)
        Talk2CarBase.__init__(split=split, commands_root=commands_root, slim=False)
        # print("Did you update the commands.json in the nuscenes folder with the new version?")
        # print("If so, continue.")
        # Load commands
        self.scene_tokens = None

class Talk2CarSlim(Talk2CarBase):
    img_mean = [0.3950, 0.4004, 0.3906]
    img_std = [0.2115, 0.2068, 0.2164]

    def __init__(
        self,
        split: str = "train",
        root: str = "../datasets/nuScenes/data/sets/nuscenes",
        commands_root = None,
        verbose: bool = False,
    ):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param split: Version to load (e.g. "v1.0-trainval", ...).
        :param root: Path to the tables and data.
        :param commands_root: Path to the command data. If None, will use the path given to the root param.
        :param verbose: Whether to print status messages during load.
        """
        commands_root = commands_root if commands_root else root
        super().__init__(split=split, commands_root=commands_root, slim=True)
        # print("Did you update the commands.json in the nuscenes folder with the new version?")
        # print("If so, continue.")
        # Load commands
        self.scene_tokens = None

def get_talk2car_class(root, split, command_path=None, slim=True, verbose=False):
    if slim:
        return Talk2CarSlim(root=root, split=split, verbose=verbose, commands_root=command_path)
    else:
        return Talk2Car(root=root, split=split, verbose=verbose, commands_root=command_path)

if __name__ == "__main__":
    ds = get_talk2car_class("./data", split="val")
    print("ds")
