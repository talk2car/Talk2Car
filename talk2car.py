import json
import os.path as osp
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
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
        referred_box_2d: list,
        t2c_image_name: str,
        box_token: str = None,
        slim_dataset: bool = False,
        color: str = None,
        location: str = None,
        action: str = None,
        referring_expression: str = None,
        destination_data: dict = None,
    ):
        """
        :param frame_token:
        :param scene_token:
        :param box:
        """
        self.scene_token = scene_token
        self.frame_token = frame_token
        self.box = box
        self.command = command
        self.box_token = box_token
        self.command_token = command_token
        self.t2c = t2c
        self.slim_dataset = slim_dataset
        self.referred_box_2d = referred_box_2d
        self.t2c_image_name = t2c_image_name
        self.color = color
        self.location = location
        self.action = action
        self.referring_expression = referring_expression
        self.destination_data = destination_data
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
            "text": self.command,
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


class Destination:

    def __init__(self, destinations, egobbox_top=None,
                 all_detections_top=None, detected_object_classes=None,
                 all_detections_front=None, predicted_referred_obj_index=None,
                 detection_scores=None, gt_referred_obj_top=None, **kwargs):

        self.destinations = destinations
        self.egobbox_top = egobbox_top
        self.all_detections_top = all_detections_top
        self.detected_object_classes = detected_object_classes
        self.all_detections_front = all_detections_front
        self.predicted_referred_obj_index = predicted_referred_obj_index
        self.detection_scores = detection_scores
        self.gt_referred_obj_top = gt_referred_obj_top

class Trajectory(Destination):
    def __init__(self, trajectories, **kwargs):
        self.trajectories = trajectories
        super().__init__(**kwargs)

class Talk2CarBase:
    img_mean = [0.3950, 0.4004, 0.3906]
    img_std = [0.2115, 0.2068, 0.2164]

    def __init__(self, split, commands_root, slim, load_talk2car_expr=False, load_talk2car_destination=False,
                 load_talk2car_trajectory=False):
        self.version = split
        self.commands_root = commands_root
        self.commands = []
        self.lookup = {}
        self.max_command_length = 0
        self.slim = slim
        self.load_talk2car_expr = load_talk2car_expr
        if load_talk2car_destination and load_talk2car_trajectory:
            print("[WARNING] Talk2CarBase: load_talk2car_destination and load_talk2car_trajectory are both True. "
                  "Setting load_talk2car_destination to False.")
        self.load_talk2car_destination = load_talk2car_destination if not load_talk2car_trajectory else False
        self.load_talk2car_trajectory = load_talk2car_trajectory

        if load_talk2car_expr:
            self.attr_data = self.load_talk2car_expr_data()

        if self.load_talk2car_destination:
            self.destination_data = self.load_talk2car_destination_data()

        if self.load_talk2car_trajectory:
            self.trajectory_data = self.load_talk2car_trajectory_data()

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
                t2c_image_name=c["t2c_img"],
                box_token=c.get("box_token", None),
                slim_dataset=self.slim
            )
            if self.load_talk2car_expr and self.version != "test":
                attr_data = self.attr_data[command_token]
                cmd.referring_expression = attr_data["description"]
                cmd.action = attr_data["action"]
                cmd.location = attr_data["location"]
                cmd.color = attr_data["color"]
            
            if self.load_talk2car_destination and self.version != "test":
                if command_token in self.destination_data:
                    cmd.destination = Destination(**self.destination_data[command_token])
                else:
                    cmd.destination = None

            if self.load_talk2car_trajectory and self.version != "test":
                if command_token in self.trajectory_data:
                    cmd.trajectory = Trajectory(**self.trajectory_data[command_token])
                else:
                    cmd.trajectory = None
                
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

    def load_talk2car_expr_data(self):
        with open(
                osp.join(
                    osp.join(self.commands_root, "commands"),
                    "talk2car_expr_{}.json".format(self.version),
                )
        ) as f:
            return json.load(f)

    def load_talk2car_destination_data(self):
        with open(
                osp.join(
                    osp.join(self.commands_root, "commands"),
                    "talk2car_destination_{}.json".format(self.version),
                )
        ) as f:
            return json.load(f)

    def load_talk2car_trajectory_data(self):
        with open(
                osp.join(
                    osp.join(self.commands_root, "commands"),
                    "talk2car_trajectory_{}.json".format(self.version),
                )
        ) as f:
            return json.load(f)

class Talk2Car(NuScenes, Talk2CarBase):

    def __init__(
        self,
        split,
        root,
        commands_root = None,
        verbose: bool = False,
        load_talk2car_expr: bool = False,
        load_talk2car_destination: bool = False,
        load_talk2car_trajectory: bool = False,
    ):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param split: Version to load (e.g. "v1.0-trainval", ...).
        :param root: Path to the tables and data.
        :param commands_root: Path to the command data. If None, will use the path given to the root param.
        :param verbose: Whether to print status messages during load.
        """
        commands_root = commands_root if commands_root else root

        NuScenes.__init__(version="v1.0-trainval", dataroot=root, verbose=verbose, self=self)
        Talk2CarBase.__init__(split=split, commands_root=commands_root, slim=False,
                              self=self, load_talk2car_expr=load_talk2car_expr,
                              load_talk2car_destination=load_talk2car_destination,
                              load_talk2car_trajectory=load_talk2car_trajectory)
        # print("Did you update the commands.json in the nuscenes folder with the new version?")
        # print("If so, continue.")
        # Load commands
        self.scene_tokens = None

class Talk2CarSlim(Talk2CarBase):

    def __init__(
        self,
        split,
        root,
        commands_root = None,
        verbose: bool = False,
        load_talk2car_expr: bool = False,
        load_talk2car_destination: bool = False,
        load_talk2car_trajectory: bool = False,
    ):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param split: Version to load (e.g. "v1.0-trainval", ...).
        :param root: Path to the talk2car jsons.
        :param commands_root: Path to the command data. If None, will use the path given to the root param.
        :param verbose: Whether to print status messages during load.
        """
        commands_root = commands_root if commands_root else root
        super().__init__(split=split, commands_root=commands_root, slim=True,
                         load_talk2car_expr=load_talk2car_expr, load_talk2car_destination=load_talk2car_destination,
                         load_talk2car_trajectory=load_talk2car_trajectory)
        # print("Did you update the commands.json in the nuscenes folder with the new version?")
        # print("If so, continue.")
        # Load commands
        self.scene_tokens = None

def get_talk2car_class(root, split, command_path=None, slim=True, verbose=False,
                       load_talk2car_expr=False, load_talk2car_destination=False, load_talk2car_trajectory=False):
    """
    Helper function to retrieve a slimmed down version of the Talk2Car dataset (recommended) or the full version including all nuScenes data.
    The latter one takes more time to load.

    :param root: Path to the data.
    :param split: Version to load (e.g. "train", ...).
    :param command_path: Path to the command data. If None, will use the path given to the root param.
    :param slim: Whether to load the slim version of the dataset.
    :param verbose: Whether to print status messages during load.
    :param load_talk2car_expr: Whether to load the Talk2Car expression data.
    :param load_talk2car_destination: Whether to load the Talk2Car destination data.
    :param load_talk2car_trajectory: Whether to load the Talk2Car trajectory data.
        Note: the Talk2Car-Trajectory dataset also contains the Talk2Car-Destination dataset so you do not have to set
        load_talk2car_destination to True if you set load_talk2car_trajectory to True.

    :return: The Talk2Car dataset.
    """

    if slim:
        return Talk2CarSlim(root=root, split=split, verbose=verbose,
                            commands_root=command_path, load_talk2car_expr=load_talk2car_expr,
                            load_talk2car_destination=load_talk2car_destination, load_talk2car_trajectory=load_talk2car_trajectory)
    else:
        return Talk2Car(root=root, split=split, verbose=verbose,
                        commands_root=command_path, load_talk2car_expr=load_talk2car_expr,
                        load_talk2car_destination=load_talk2car_destination, load_talk2car_trajectory=load_talk2car_trajectory)

def main():
    ds = get_talk2car_class("./data", split="val", load_talk2car_destination=True, load_talk2car_trajectory=True)
    print("#Commands for split {}: {}".format(ds.version, len(ds.commands)))

if __name__ == "__main__":
    main()