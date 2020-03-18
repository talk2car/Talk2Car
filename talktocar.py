import cv2
import json
import os.path as osp
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from typing import List, Tuple, Dict


class Command:
    def __init__(
        self,
        t2c,
        scene_token: str,
        frame_token: str,
        box: Box,
        command: str,
        command_token: str,
        box_token: str = None,
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
        :return: A tuple of a bounding box as (x1, y1, width, height)
        """

        # Get translated corners
        b = np.zeros((900, 1600, 3))

        self.box.render_cv2(
            b,
            view=self.camera_intrinsic,
            normalize=True,
            colors=((0, 0, 255), (0, 0, 255), (0, 0, 255)),
        )
        y, x = np.nonzero(b[:, :, 0])

        x1, y1, x2, y2 = map(int, (x.min(), y.min(), x.max(), y.max()))
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(1600, x2)
        y2 = min(900, y2)
        return (x1, y1, x2 - x1, y2 - y1)

    def get_image_path(self):
        """
        :return:
        """
        # Get records from DB
        sd_rec = self.t2c.get("sample_data", self.frame_token)

        # Get data from DB
        im_path, _, _ = self.t2c.get_sample_data(sd_rec["token"])
        return im_path


class Talk2Car(NuScenes):
    img_mean = [0.3950, 0.4004, 0.3906]
    img_std = [0.2115, 0.2068, 0.2164]

    def __init__(
        self,
        version: str = "train",
        dataroot: str = "../datasets/nuScenes/data/sets/nuscenes",
        verbose: bool = False,
    ):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "v1.0-trainval", ...).
        :param dataroot: Path to the tables and data.
        :param verbose: Whether to print status messages during load.
        """
        super().__init__(version="v1.0-trainval", dataroot=dataroot, verbose=verbose)
        # print("Did you update the commands.json in the nuscenes folder with the new version?")
        # print("If so, continue.")
        self.version = version
        # Load commands
        self.scene_tokens = None
        if version:
            (
                self.commands,
                self.lookup,
                self.max_command_length,
            ) = self.__load_commands__()
        else:
            self.commands = []
            self.lookup = []
            self.max_command_length = 0

    def __len__(self):
        return len(self.commands)

    def __load_t2c_table__(self, table_name):
        with open(
            osp.join(
                osp.join(self.dataroot, "commands"),
                "{}_commands.json".format(table_name),
            )
        ) as f:
            table = json.load(f)
        return table

    def __load_commands__(self) -> [List, Dict]:
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
                box_token=c.get("box_token", None),
            )
            ret.append(cmd)

            if len(command_text) > max_length:
                max_length = len(command_text)

            if scene_token not in lookup:
                lookup[scene_token] = []

            lookup[scene_token].append(cmd)

        return ret, lookup, max_length

    def change_version(self, new_version):
        self.version = new_version
        self.commands, self.lookup, self.max_command_length = self.__load_commands__()

    def list_commands(self) -> None:
        """
        :return:
        """
        for command in self.commands:
            print(command)

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

    def get_commands_for_scene(self, scene_token) -> List:
        """
        :param scene_token: The scene we want the commands for
        :return: a list of all existing commands for a certain scene. Empty if no commands for a scene exists.
        """
        return self.lookup.get(scene_token, [])

    def get_imgs_with_bboxes(
        self, only_key_frames=True, transform_2d_bbox=False
    ) -> List:
        """
        :param only_key_frames: If you want to only get the key frames from a video
        :param transform_2d_bbox: Transform the 3D bounding boxes into 2D of the form (x,y,w,h)
                                    with x and y being the lower left corner of the bounding box
        :return: a list with elements of the format (impath, boxes, camera_intrinsic).
                 The boxes are transformed to 2D if the parameter transform_2d_bbox is set to true else
                 these boxes will be in 3D
        """
        imgs = []
        for scene_token in self.scene_tokens:
            scene_obj = self.get("scene", scene_token)
            next_token = scene_obj["first_sample_token"]
            while next_token:
                curr_sample = self.get("sample", next_token)
                next_token = curr_sample["next"]
                fc_rec = curr_sample["data"]["CAM_FRONT"]
                # Get records from DB
                sd_rec = self.get("sample_data", fc_rec)
                if not sd_rec["is_key_frame"] and only_key_frames:
                    continue
                # Get data from DB
                impath, boxes, camera_intrinsic = self.get_sample_data(sd_rec["token"])
                if transform_2d_bbox:
                    boxes = [
                        self._transform_3d_to_2d_bbox(bbox, camera_intrinsic)
                        for bbox in boxes
                    ]

                imgs.append((impath, boxes, camera_intrinsic))

        return imgs

    def store_command(
        self, command: Command, file_name, imsize: Tuple[float, float] = (640, 360)
    ):
        """
        :param command:
        :param file_name:
        :param imsize:
        :return:
        """
        # Get records from DB
        sd_rec = self.get("sample_data", command.frame_token)

        # Get data from DB
        impath, boxes, camera_intrinsic = self.get_sample_data(sd_rec["token"])

        im = cv2.imread(impath)
        c = self.explorer.get_color(command.box.name)
        command.box.render_cv2(
            im, view=camera_intrinsic, normalize=True, colors=(c, c, c)
        )

        # Render
        im = cv2.resize(im, imsize)
        cv2.imwrite(f"{file_name}", im)
        print(command.text)
