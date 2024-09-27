import math
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

# Setting up scale factors
METRES_M_SCALE = 10.0
MS_TO_MPH = 2.23694
VELOCITY_MS_SCALE = 5.0

# 뭔가 enum 형태로 표현하고 싶어서 만든거 같다
# Enumerating fields for the vector representations of the different objects
# Unspecified fields are not involved in the structured language generation
class VehicleField(IntEnum):
    ACTIVE = 0
    DYNAMIC = 1
    SPEED = 2
    X = 3
    Y = 4
    Z = 5
    DX = 6
    DY = 7
    PITCH = 8
    HALF_LENGTH = 9
    HALF_WIDTH = 10
    HALF_HEIGHT = 11
    UNSPECIFIED_12 = 12
    UNSPECIFIED_13 = 13
    UNSPECIFIED_14 = 14
    UNSPECIFIED_15 = 15
    UNSPECIFIED_16 = 16
    UNSPECIFIED_17 = 17
    UNSPECIFIED_18 = 18
    UNSPECIFIED_19 = 19
    UNSPECIFIED_20 = 20
    UNSPECIFIED_21 = 21
    UNSPECIFIED_22 = 22
    UNSPECIFIED_23 = 23
    UNSPECIFIED_24 = 24
    UNSPECIFIED_25 = 25
    UNSPECIFIED_26 = 26
    UNSPECIFIED_27 = 27
    UNSPECIFIED_28 = 28
    UNSPECIFIED_29 = 29
    UNSPECIFIED_30 = 30
    UNSPECIFIED_31 = 31
    UNSPECIFIED_32 = 32

# 사람 장애물에 대해서 enum으로 만들고자 한 듯
class PedestrianField(IntEnum):
    ACTIVE = 0
    SPEED = 1
    X = 2
    Y = 3
    Z = 4
    DX = 5
    DY = 6
    CROSSING = 8

# 길에 대한 정보도 enum으로 생성함
class RouteField(IntEnum):
    X = 0
    Y = 1
    Z = 2
    TANGENT_DX = 3
    TANGENT_DY = 4
    PITCH = 5
    SPEED_LIMIT = 6
    HAS_JUNCTION = 7
    ROAD_WIDTH0 = 8
    ROAD_WIDTH1 = 9
    HAS_TL = 10
    TL_GO = 11
    TL_GOTOSTOP = 12
    TL_STOP = 13
    TL_STOPTOGO = 14
    IS_GIVEWAY = 15
    IS_ROUNDABOUT = 16

# 자선에 대한 정보도 enum으로 생성함
class EgoField(IntEnum):
    ACCEL = 0
    SPEED = 1
    BRAKE_PRESSURE = 2
    STEERING_ANGLE = 3
    PITCH = 4
    HALF_LENGTH = 5
    HALF_WIDTH = 6
    HALF_HEIGHT = 7
    UNSPECIFIED_8 = 8
    CLASS_START = 9
    CLASS_END = 13
    DYNAMICS_START = 14
    DYNAMICS_END = 15
    PREV_ACTION_START = 16
    UNSPECIFIED_17 = 17
    PREV_ACTION_END = 18
    RAYS_LEFT_START = 19
    UNSPECIFIED_20 = 20
    UNSPECIFIED_21 = 21
    UNSPECIFIED_22 = 22
    UNSPECIFIED_23 = 23
    RAYS_LEFT_END = 24
    RAYS_RIGHT_START = 25
    UNSPECIFIED_26 = 26
    UNSPECIFIED_27 = 27
    UNSPECIFIED_28 = 28
    UNSPECIFIED_29 = 29
    RAYS_RIGHT_END = 30

# 관계성이 있는 차량에 대한 넘버링
class LiableVechicleField(IntEnum):
    VEHICLE_0 = 0
    VEHICLE_1 = 1
    VEHICLE_2 = 2
    VEHICLE_3 = 3
    VEHICLE_4 = 4
    VEHICLE_5 = 5
    VEHICLE_6 = 6
    VEHICLE_7 = 7
    VEHICLE_8 = 8
    VEHICLE_9 = 9
    VEHICLE_13 = 13
    VEHICLE_14 = 14
    VEHICLE_15 = 15
    VEHICLE_16 = 16
    VEHICLE_17 = 17
    VEHICLE_18 = 18
    VEHICLE_19 = 19
    VEHICLE_20 = 20
    VEHICLE_21 = 21
    VEHICLE_22 = 22
    VEHICLE_23 = 23
    VEHICLE_24 = 24
    VEHICLE_25 = 25
    VEHICLE_26 = 26
    VEHICLE_27 = 27
    VEHICLE_28 = 28
    VEHICLE_29 = 29

########################################################################################################################
#                                                   Utility Function
########################################################################################################################
# 주어진 vehicle 정보에 대해서 좌표를 추출하기 위한 index를 주려고 enum을 준거네
def xy_from_vehicle_desc(vehicle_array):
    x = vehicle_array[:, VehicleField.X]
    y = vehicle_array[:, VehicleField.Y]
    # scale 변환 수행
    return torch.vstack((x, y)).T * METRES_M_SCALE

#차량의 변환하는 각도에 대해서 계산하고자하는 함수
def traveling_angle_deg_from_vehicle_desc(vehicle_array):
    dx = vehicle_array[:, VehicleField.DX]
    dy = vehicle_array[:, VehicleField.DY]
    return direction_to_angle_deg(dx, dy)

# 속도에 대한 단위 변경
def speed_mph_from_vehicle_desc(vehicle_array):
    return vehicle_array[:, VehicleField.SPEED] * VELOCITY_MS_SCALE * MS_TO_MPH

def xy_from_pedestrian_desc(pedestrian_array):
    x = pedestrian_array[:, PedestrianField.X]
    y = pedestrian_array[:, PedestrianField.Y]
    return torch.vstack((x, y)).T * METRES_M_SCALE

# 사람에 대한 이동 방향에 대한 계산
def traveling_angle_deg_from_pedestrian_desc(pedestrian_array):
    dx = pedestrian_array[:, PedestrianField.DX]
    dy = pedestrian_array[:, PedestrianField.DY]
    return direction_to_angle_deg(dx, dy)

# 도로에 대한 거리 단위값 변경
def xy_from_route_desc(route_array):
    x = route_array[:, RouteField.X]
    y = route_array[:, RouteField.Y]
    return torch.vstack((x, y)).T * METRES_M_SCALE

# 도로에 대해서 각도에 대한 계산 수행
def route_angles_from_route_desc(route_array):
    dirx, diry = (
        route_array[:, RouteField.TANGENT_DX],
        route_array[:, RouteField.TANGENT_DY],
    )
    return direction_to_angle_deg(dirx, diry)

# 주어진 좌표계들이 fov이내에 존재하는지 판단하는 함수
def flags_in_fov(xy_coords, fov_degrees=60, max_distance=40):
    distances, angular = angles_deg_and_distances(xy_coords)
    # 유효한 좌표계인지
    # 각도가 fov이내인지
    # 거리가 유효 거리 이내인지
    return (
        (xy_coords[:, 0] > 0)
        & (-fov_degrees < angular)
        & (angular < fov_degrees)
        & (distances <= max_distance)
    )

# 각도와 거리를 계산함
def angles_deg_and_distances(xy_coords):
    distances = torch.linalg.norm(xy_coords, axis=1)
    angular = direction_to_angle_deg(xy_coords[:, 0], xy_coords[:, 1])
    return distances, angular

def direction_to_angle_deg(dirx, diry):
    return torch.atan2(-diry, dirx) * 180.0 / np.pi

# 각도값을 기준으로 sort 수행함
def sort_angular(distances, angular):
    angular_order = np.argsort(angular)
    return distances[angular_order], angular[angular_order]

# 활성화 되어 있는 자동차인 경우에 대해서, fov와 max 거리 이내인지 판단해서 장애물 정보들을 반환함
def vehicle_filter_flags(vehicle_descriptors):
    active_flags = vehicle_descriptors[:, VehicleField.ACTIVE] == 1
    fov_flags = flags_in_fov(xy_from_vehicle_desc(vehicle_descriptors), max_distance=40)
    return active_flags & fov_flags

def pedestrian_filter_flags(pedestrian_descriptors):
    active_flags = pedestrian_descriptors[:, PedestrianField.ACTIVE] == 1
    fov_flags = flags_in_fov(
        xy_from_pedestrian_desc(pedestrian_descriptors), max_distance=30
    )
    return active_flags & fov_flags

# 현재 위치로 부터 교차료 까지의 거리를 계산하는 것이다
def distance_to_junction(route_descriptor):
    is_junction = route_descriptor[:, RouteField.HAS_JUNCTION] > 0.0
    if is_junction[0]:
        return 0
    elif torch.all(~is_junction):
        return torch.inf
    else:
        distances, angular = angles_deg_and_distances(
            xy_from_route_desc(route_descriptor)
        )
        return torch.amin(distances[is_junction])

# traffic light에 대한 정보를 불러오는 것이다
# 거리와 각도를 계산하고
# 신호등의 상태를 확인함
def get_tl_state(route_descriptor):
    is_tl = route_descriptor[:, RouteField.HAS_TL] > 0.0
    if torch.all(~is_tl):
        return None, None

    distances, angular = angles_deg_and_distances(xy_from_route_desc(route_descriptor))

    tl_index = np.where(is_tl)[0][0]
    tl_flags = route_descriptor[tl_index, RouteField.TL_GO : RouteField.TL_GO + 4] > 0.0
    index_on = np.where(tl_flags)[0][0]

    return ["green", "yellow", "red", "red+yellow"][index_on], distances[tl_index]

# 각도에 대해서 문장형의 상태로 변환함
def object_direction(angle_deg):
    if abs(angle_deg) < 45:
        return "same direction as me"
    elif abs(abs(angle_deg) - 180) < 45:
        return "opposite direction from me"
    elif abs(angle_deg - 90) < 45:
        return "from left to right"
    elif abs(angle_deg + 90) < 45:
        return "from right to left"
    return f"{angle_deg} degrees"

def side(angle_deg):
    return "left" if angle_deg < 0 else "right"

# 종방향 입력값에 대해서,
# 악셀과 브레이크 정보를 계산하게 된다
# 이때 x는 -1부터 1까지의 정보를 가지게 되며
# -1은 브레이크를 최대로 밟은 상태, 1은 가속을 최대로 밟은 상태
def control_to_pedals(control_longitudinal):
    x = 2.0 * control_longitudinal - 1.0
    accelerator_pedal_pct = np.clip(x, 0.0, 1.0)
    brake_pressure_pct = np.clip(-x, 0.0, 1.0)
    return accelerator_pedal_pct, brake_pressure_pct

# 주어진 경로 데이터에 대해서, 회전 교차료를 탐지하는 역할
# 차량이 여러번 좌회전 우회전 하는지를 확인해서, 그게 회전 교차료 인지를 확인한다
# 각도의 차이를 더해서 회전 교차료임을 알아내게 되는 것이다
def determine_roundabout(route_descriptors):
    route_angles = route_angles_from_route_desc(route_descriptors)
    angle_diffs = torch.diff(route_angles)
    angle_diffs[angle_diffs > 180] -= 360
    angle_diffs[angle_diffs < -180] += 360
    total_turn_right = torch.sum(angle_diffs[angle_diffs > 0])
    total_turn_left = torch.sum(angle_diffs[angle_diffs < 0])
    return abs(total_turn_left) > 30 and abs(total_turn_right) > 30

########################################################################################################################
#                                                   Randomization utils
########################################################################################################################

class Randomizable:
    ENUM: Any = None
    FIELD_TYPES_RANGES: Dict[str, Tuple[Any, Any]] = {}
    FIELD_PROB: Dict[str, float] = {}

    @classmethod
    def randomize(cls, vector: np.array):
        randomize_enum(cls.ENUM, vector, cls.FIELD_TYPES_RANGES, cls.FIELD_PROB)

# enum_cls를 기반으로 해서 vector에 임의의 값을 할당하게 되는 것이다
def randomize_enum(enum_cls, vector: np.array, field_types_ranges, field_probs):
    for field_name in field_types_ranges:
        idx = getattr(enum_cls, field_name)
        field_type_range = field_types_ranges[field_name]
        field_prob = field_probs.get(field_name, 0.5)
        vector[idx] = random_value(field_type_range, field_prob)

# 주어진 필드와, 확률을 기반으로 해서 랜덤값을 생성하게 된다
# 불리언인지 플롯인지에 따라서 다른 방식으로 값을 생성하게 된다
def random_value(
    field_type_range: Tuple[type, Tuple[float, float]],
    field_prob: float = 0.5,
) -> Union[int, float]:
    field_type, field_range = field_type_range
    if field_type == bool:
        return 1 if random.random() < field_prob else 0
    if field_type == float:
        return random.uniform(*field_range)
    raise ValueError(f"Unsupported field type: {field_type}")

# 이것에서 보이는 것처럼 랜덤하게 값을 생성하게 된다
# 아마 observation값도 이런 형식으로 vector가 만들어진 것 같다
class VehicleFieldRandom(Randomizable):
    ENUM = VehicleField
    FIELD_TYPES_RANGES: Dict[str, Any] = {
        "ACTIVE": (bool, (0.0, 1.0)),
        "DYNAMIC": (bool, (0.0, 1.0)),
        "SPEED": (float, (-1.4e-09, 2.2e00)),
        "X": (float, (-9.9e00, 9.9e00)),
        "Y": (float, (-9.9e00, 9.9e00)),
        "Z": (float, (0.0, 0.0)),
        "DX": (float, (-1.0, 1.0)),
        "DY": (float, (-1.0, 1.0)),
        "PITCH": (float, (0.0, 0.0)),
        "HALF_LENGTH": (float, (0.0, 2.4e00)),
        "HALF_WIDTH": (float, (0.0, 9.6e-01)),
        "HALF_HEIGHT": (float, (0.0, 8.9e-01)),
        "UNSPECIFIED_12": (bool, (0.0, 1.0)),
        "UNSPECIFIED_13": (bool, (0.0, 1.0)),
        "UNSPECIFIED_14": (bool, (0.0, 1.0)),
        "UNSPECIFIED_15": (bool, (0.0, 1.0)),
        "UNSPECIFIED_16": (bool, (0.0, 1.0)),
        "UNSPECIFIED_17": (bool, (0.0, 1.0)),
        "UNSPECIFIED_18": (float, (-1.0e01, 1.0e01)),
        "UNSPECIFIED_19": (float, (-1.0e01, 1.0e01)),
        "UNSPECIFIED_20": (float, (-1.0e01, 1.0e01)),
        "UNSPECIFIED_21": (float, (-1.0e01, 1.0e01)),
        "UNSPECIFIED_22": (float, (-1.0e01, 1.0e01)),
        "UNSPECIFIED_23": (float, (-1.0e01, 1.0e01)),
        "UNSPECIFIED_24": (float, (-1.0e01, 1.0e01)),
        "UNSPECIFIED_25": (float, (-1.0e01, 1.0e01)),
        "UNSPECIFIED_26": (float, (-1.0e-09, 2.0e00)),
        "UNSPECIFIED_27": (float, (-1.0e-09, 2.0e00)),
        "UNSPECIFIED_28": (float, (-1.0e-10, 2.0e00)),
        "UNSPECIFIED_29": (float, (-2.0e-10, 4.0e00)),
        "UNSPECIFIED_30": (bool, (0.0, 1.0)),
        "UNSPECIFIED_31": (bool, (0.0, 1.0)),
        "UNSPECIFIED_32": (bool, (0.0, 1.0)),
    }

    FIELD_PROB: Dict[str, float] = {"ACTIVE": 0.8}

class PedestrianFieldRandom(Randomizable):
    ENUM = PedestrianField

    FIELD_TYPES_RANGES: Dict[str, Tuple[Any, Any]] = {
        "ACTIVE": (bool, (0.0, 1.0)),
        "SPEED": (float, (0.0, 1.7933)),
        "X": (float, (0.0, 4.9982)),
        "Y": (float, (-4.9940, 4.9829)),
        "Z": (float, (0.0, 0.1992)),
        "DX": (float, (-1.0, 1.0)),
        "DY": (float, (-1.0, 1.0)),
        "CROSSING": (bool, (0.0, 1.0)),
    }
    FIELD_PROB: Dict[str, float] = {"ACTIVE": 0.9}


class RouteFieldRandom:
    FIELD_TYPES_RANGES: Dict[str, Tuple[Any, Any]] = {
        "X": (float, (-4.2200, 6.0593)),
        "Y": (float, (-5.7505, 5.7946)),
        "Z": (float, (0.0, 0.0)),
        "TANGENT_DX": (float, (-1.0, 1.0)),
        "TANGENT_DY": (float, (-1.0, 1.0)),
        "PITCH": (float, (0.0, 0.0)),
        "SPEED_LIMIT": (float, (1.7882, 2.6822)),
        "HAS_JUNCTION": (bool, (0.0, 1.0)),
        "ROAD_WIDTH0": (float, (0.1275, 1.0)),
        "ROAD_WIDTH1": (float, (0.1708, 1.0)),
        "HAS_TL": (bool, (0.0, 1.0)),
        "TL_GO": (bool, (0.0, 1.0)),
        "TL_GOTOSTOP": (bool, (0.0, 1.0)),
        "TL_STOP": (bool, (0.0, 1.0)),
        "TL_STOPTOGO": (bool, (0.0, 1.0)),
        "IS_GIVEWAY": (bool, (0.0, 1.0)),
        "IS_ROUNDABOUT": (bool, (0.0, 1.0)),
    }

    # 경로 필드에 대해서 무작위로 초기화 하는 것
    # 초기화 방식이 혼자 조금 달라서 추가로 정의한 듯
    @staticmethod
    def randomize_route_field(field_types_ranges, vector: np.array, has_tl=None):
        if has_tl is None:
            has_tl = random.random() < 0.5
        for field_name in field_types_ranges:
            idx = getattr(RouteField, field_name)
            field_type_range = field_types_ranges[field_name]
            vector[idx] = random_value(field_type_range)
        # reset the traffic light state
        if has_tl is None:
            has_tl = random.random() < 0.75
        vector[RouteField.HAS_TL] = 1 if has_tl else 0
        vector[RouteField.TL_GO : RouteField.TL_STOPTOGO + 1] = (
            vector[RouteField.TL_GO : RouteField.TL_STOPTOGO + 1] * 0.0
        )
        if vector[RouteField.HAS_TL] == 1:
            tl_state = random.randint(0, 3)
            vector[RouteField.TL_GO + tl_state] = 1

    @classmethod
    def randomize(cls, vector: np.array, has_tl=None):
        cls.randomize_route_field(cls.FIELD_TYPES_RANGES, vector, has_tl)

class EgoFieldRandom(Randomizable):
    ENUM = EgoField
    FIELD_TYPES_RANGES: Dict[str, Tuple[Any, Any]] = {
        "ACCEL": (float, (-4.1078, 1.9331)),
        "SPEED": (float, (-3.2891e-10, 2.1762)),
        "BRAKE_PRESSURE": (float, (0.0, 0.89103)),
        "STEERING_ANGLE": (float, (-7.8079, 7.3409)),
        "PITCH": (float, (0.0, 0.0)),
        "HALF_LENGTH": (float, (0.0, 2.3250)),
        "HALF_WIDTH": (float, (0.0, 1.0050)),
        "HALF_HEIGHT": (float, (0.0, 0.7800)),
        "UNSPECIFIED_8": (bool, (0.0, 1.0)),
        "CLASS_START": (bool, (0.0, 0.0)),
        "CLASS_END": (bool, (0.0, 0.0)),
        "DYNAMICS_START": (bool, (0.0, 1.0)),
        "DYNAMICS_END": (bool, (0.0, 0.0)),
        "PREV_ACTION_START": (float, (0.0, 0.99949)),
        "UNSPECIFIED_17": (float, (-0.98492, 0.91599)),
        "PREV_ACTION_END": (float, (0.0, 0.99169)),
        "RAYS_LEFT_START": (bool, (0.0, 1.0)),
        "UNSPECIFIED_20": (bool, (0.0, 1.0)),
        "UNSPECIFIED_21": (bool, (0.0, 1.0)),
        "UNSPECIFIED_22": (bool, (0.0, 1.0)),
        "UNSPECIFIED_23": (bool, (0.0, 1.0)),
        "RAYS_LEFT_END": (bool, (0.0, 1.0)),
        "RAYS_RIGHT_START": (bool, (0.0, 1.0)),
        "UNSPECIFIED_26": (bool, (0.0, 1.0)),
        "UNSPECIFIED_27": (bool, (0.0, 1.0)),
        "UNSPECIFIED_28": (bool, (0.0, 1.0)),
        "UNSPECIFIED_29": (bool, (0.0, 1.0)),
        "RAYS_RIGHT_END": (bool, (0.0, 1.0)),
    }


class LiableVehiclesRandom(Randomizable):
    ENUM = LiableVechicleField
    FIELD_TYPES_RANGES: Dict[str, Tuple[Any, Any]] = {
        "VEHICLE_0": (bool, (0.0, 1.0)),
        "VEHICLE_1": (bool, (0.0, 1.0)),
        "VEHICLE_2": (bool, (0.0, 1.0)),
        "VEHICLE_3": (bool, (0.0, 1.0)),
        "VEHICLE_4": (bool, (0.0, 1.0)),
        "VEHICLE_5": (bool, (0.0, 1.0)),
        "VEHICLE_6": (bool, (0.0, 1.0)),
        "VEHICLE_7": (bool, (0.0, 1.0)),
        "VEHICLE_8": (bool, (0.0, 1.0)),
        "VEHICLE_9": (bool, (0.0, 1.0)),
        "VEHICLE_13": (bool, (0.0, 1.0)),
        "VEHICLE_14": (bool, (0.0, 1.0)),
        "VEHICLE_15": (bool, (0.0, 1.0)),
        "VEHICLE_16": (bool, (0.0, 1.0)),
        "VEHICLE_17": (bool, (0.0, 1.0)),
        "VEHICLE_18": (bool, (0.0, 1.0)),
        "VEHICLE_19": (bool, (0.0, 1.0)),
        "VEHICLE_20": (bool, (0.0, 1.0)),
        "VEHICLE_21": (bool, (0.0, 1.0)),
        "VEHICLE_22": (bool, (0.0, 1.0)),
        "VEHICLE_23": (bool, (0.0, 1.0)),
        "VEHICLE_24": (bool, (0.0, 1.0)),
        "VEHICLE_25": (bool, (0.0, 1.0)),
        "VEHICLE_26": (bool, (0.0, 1.0)),
        "VEHICLE_27": (bool, (0.0, 1.0)),
        "VEHICLE_28": (bool, (0.0, 1.0)),
        "VEHICLE_29": (bool, (0.0, 1.0)),
    }

@dataclass
class VectorObservation:
    # 차량 주변에 있는 요소를 벡터 형식으로 표현하는 구조
    # 다양한 정보를 torch.FloatTensor형식으로 저장하게 된다
    """
    Vectorized representation
    It stores information about the environment in the float torch tensors, coding flags and properties
    about the route, nearby vehicles, pedestrians etc.

    Arrays of dynamic number of objects, such as nearby pedestrians use the following scheme:
     - an array is preallocated for a max capacity and initialized with 0s
     - every found object sets first number to 1
    Example: we allocate 20 rows to describe pedestrians, but if there are 5 pedestrians around, then
        only first 5 rows of that array will have 1 in the first column. Others would be 0s
    All objects like that are ordered by distance, such that the closest object goes in the row 0 and so on.

    """

    ROUTE_DIM = 17
    VEHICLE_DIM = 33
    PEDESTRIAN_DIM = 9
    EGO_DIM = 31

    # A 2d array per ego vehicle describing the route to follow.
    # It finds route points for each vehicle. Each point goes into a new row, then for each point:
    # - (x, y, z) of a point
    # - its direction
    # - pitch
    # - speed limit
    # - is junction?
    # - width of the road
    # - is traffic light and its state
    # - is give way?
    # - is a roundabout?
    #  따라가야 하는 경로에 대한 정보를 저장함
    # 좌표, 진행 방향 경사, 속도 제한
    # 교차로인지, 신호등 상태 등
    route_descriptors: torch.FloatTensor

    # A 2d array per ego vehicle describing nearby vehicles.
    # First, if finds nearby vehicle in the neighbourhood of the car.
    # Then allocates an array of zeros a fixed max size (about 30).
    # There is a logic that tries to fit dynamic and static vehicles into rows of that array.
    # For every vehicle:
    # - "1" for marking that a vehicle is found in the row (empty rows will have this still "0")
    # - Is it dynamic or static (parked) vehicle
    # - its speed
    # - its relative position in the ego coordinates
    # - its relative orientation
    # - its pitch
    # - its size
    # - vehicle class
    # - positions of its 4 corners
    # 주변 차량에 대한 정보를 저장함
    # 속도, 상대적 위치, 방향 크기
    vehicle_descriptors: torch.FloatTensor

    # A 2d array per ego vehicle describing pedestrians.
    # First, if finds nearby pedestrians in the neighbourhood of the car.
    # Then allocates an array of zeros a fixed max size (about 20).
    # Then every found pedestrian is described in a row of that array:
    # - "1" for marking that a pedestrian is found in the row (empty rows will have this still "0")
    # - ped. speed
    # - its relative position in the ego coordinates (x, y, z)
    # - its relative orientation
    # - pedestrian type
    # - intent of crossing the road
    # 보행자에 대한 정보
    # 속도, 위치, 방향 등 , 건너려는 의도
    pedestrian_descriptors: torch.FloatTensor

    # A 1D array per ego vehicle describing its state. Specifically,
    # - VehicleDynamicsState (acc, steering, pitch ..)
    # - Vehicle size
    # - Vehicle class
    # - Vehicle dynamics type
    # - Previous action
    # - 2 lidar distance arrays, placed on the front corner of the vehicle
    # 자신에 대한 상태를 설명함
    ego_vehicle_descriptor: torch.FloatTensor

    # Deprecated
    liable_vehicles: Optional[torch.FloatTensor] = None

class VectorObservationConfig:
    num_route_points: int = 30
    num_vehicle_slots: int = 30
    num_pedestrian_slots: int = 20

    radius_m: float = 100.0
    pedestrian_radius_m: float = 50.0
    pedestrian_angle_threshold_rad: float = math.pi / 2
    route_spacing_m: float = 2.0
    num_max_static_vehicles: int = 10

    # Only include vehicles and pedestrians that have line of sight inside footpath and road navmesh
    line_of_sight: bool = False
