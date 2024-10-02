import pickle
import torch

from enum import IntEnum
import numpy as np
import pandas as pd

import os
import sys
module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.vector_utils import (
    VehicleField,
    PedestrianField,
    RouteField,
    EgoField,
    LiableVechicleField,
)

def load_and_print_pkl(file_path):
    """
    주어진 pkl 파일을 읽어와서 내용을 출력하는 함수.

    Args:
        file_path (str): 읽을 pkl 파일의 경로
    """
    # pkl 파일을 읽기 모드로 엽니다.
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # 읽어온 데이터를 출력합니다.
    print("Loaded data from:", file_path)

    # 데이터가 딕셔너리 형태일 경우, 키와 함께 살펴봅니다.
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"Key: {key}, Value: {value}")

    # 데이터가 리스트 형태일 경우, 일부 데이터만 출력합니다.
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            print(f"Item {idx}: {item}")
            if idx >= 5:  # 처음 5개 항목만 출력
                break

    # 그 외의 데이터 형태도 출력합니다.
    else:
        print(data)

def load_and_save_items(file_path, output_path, count=10):
    """
    주어진 pkl 파일에서 지정된 개수의 데이터를 읽어와 새로운 pkl 파일로 저장하는 함수.

    Args:
        file_path (str): 읽을 pkl 파일의 경로
        output_path (str): 저장할 새로운 pkl 파일의 경로
        count (int): 저장할 데이터의 개수 (기본값: 10)
    """
    # pkl 파일을 읽기 모드로 엽니다.
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # 데이터가 리스트나 딕셔너리 형태일 경우 지정된 개수만큼 데이터를 선택합니다.
    if isinstance(data, list):
        items_to_save = data[:count]
    elif isinstance(data, dict):
        keys = list(data.keys())
        selected_keys = keys[:count]
        items_to_save = {key: data[key] for key in selected_keys}
    else:
        raise TypeError("The input data must be a list or dictionary.")

    # 선택한 데이터를 새로운 pkl 파일로 저장합니다.
    with open(output_path, 'wb') as output_file:
        pickle.dump(items_to_save, output_file)

    print(f"{count} items saved to {output_path}.")

def init_ego_vehicle(
        accel=0.0, speed=0.0, brake_pressure=0.0, steering_angle=0.0, pitch=0.0,
        half_length=2.325, half_width=1.005, half_height=0.78, unspecified_8=1.0,
        class_start=0.0, class_end=0.0, dynamics_start=1.0, dynamics_end=0.0,
        prev_action_start=0.3679, unspecified_17=0.0, prev_action_end=0.0,
        rays_left_start=0.0798, unspecified_20=0.0798, unspecified_21=0.0798,
        unspecified_22=0.1423, unspecified_23=0.1751, rays_left_end=1.0,
        rays_right_start=1.0, unspecified_26=0.8573, unspecified_27=0.4652,
        unspecified_28=0.3694, unspecified_29=0.3674, rays_right_end=0.4039
):
    # Initialize ego_vehicle descriptor with provided values
    ego_vehicle = {
        EgoField.ACCEL: accel,
        EgoField.SPEED: speed,
        EgoField.BRAKE_PRESSURE: brake_pressure,
        EgoField.STEERING_ANGLE: steering_angle,
        EgoField.PITCH: pitch,
        EgoField.HALF_LENGTH: half_length,
        EgoField.HALF_WIDTH: half_width,
        EgoField.HALF_HEIGHT: half_height,
        EgoField.UNSPECIFIED_8: unspecified_8,
        EgoField.CLASS_START: class_start,
        EgoField.CLASS_END: class_end,
        EgoField.DYNAMICS_START: dynamics_start,
        EgoField.DYNAMICS_END: dynamics_end,
        EgoField.PREV_ACTION_START: prev_action_start,
        EgoField.UNSPECIFIED_17: unspecified_17,
        EgoField.PREV_ACTION_END: prev_action_end,
        EgoField.RAYS_LEFT_START: rays_left_start,
        EgoField.UNSPECIFIED_20: unspecified_20,
        EgoField.UNSPECIFIED_21: unspecified_21,
        EgoField.UNSPECIFIED_22: unspecified_22,
        EgoField.UNSPECIFIED_23: unspecified_23,
        EgoField.RAYS_LEFT_END: rays_left_end,
        EgoField.RAYS_RIGHT_START: rays_right_start,
        EgoField.UNSPECIFIED_26: unspecified_26,
        EgoField.UNSPECIFIED_27: unspecified_27,
        EgoField.UNSPECIFIED_28: unspecified_28,
        EgoField.UNSPECIFIED_29: unspecified_29,
        EgoField.RAYS_RIGHT_END: rays_right_end
    }
    return ego_vehicle


def init_pedestrian(
        active=1.0, speed=1.7488, x=3.0088, y=0.3324, z=0.1082,
        dx=0.9687, dy=0.2483, crossing=0.0
):
    # Initialize pedestrian descriptor with provided values
    pedestrian = {
        PedestrianField.ACTIVE: active,
        PedestrianField.SPEED: speed,
        PedestrianField.X: x,
        PedestrianField.Y: y,
        PedestrianField.Z: z,
        PedestrianField.DX: dx,
        PedestrianField.DY: dy,
        PedestrianField.CROSSING: crossing
    }
    return pedestrian

def init_route(x = 2.3251e-01, y = 2.2888e-06, z = 0.0000e+00,
               tangent_dx = 1.0000e+00, tangent_dy = 2.9802e-08,
               pitch =  0.0000e+00, speed_limit = 1.7882e+00,
               has_junction =  0.0000e+00, road_width0 =  1.7678e-01, road_width1 = 5.3921e-01,
               has_tl = 0.0000e+00, tl_go = 0.0000e+00, tl_gotostop = 0.0000e+00,
               tl_stop = 0.0000e+00, tl_stoptogo = 0.0000e+00, is_giveway = 0.0000e+00,
               is_roundabout = 0.0000e+00):
    route = {
        RouteField.X: x,
        RouteField.Y: y,
        RouteField.Z: z,
        RouteField.TANGENT_DX: tangent_dx,
        RouteField.TANGENT_DY: tangent_dy,
        RouteField.PITCH: pitch,
        RouteField.SPEED_LIMIT: speed_limit,
        RouteField.HAS_JUNCTION: has_junction,
        RouteField.ROAD_WIDTH0: road_width0,
        RouteField.ROAD_WIDTH1: road_width1,
        RouteField.HAS_TL: has_tl,
        RouteField.TL_GO: tl_go,
        RouteField.TL_GOTOSTOP: tl_gotostop,
        RouteField.TL_STOP: tl_stop,
        RouteField.TL_STOPTOGO: tl_stoptogo,
        RouteField.IS_GIVEWAY: is_giveway,
        RouteField.IS_ROUNDABOUT: is_roundabout
    }

    return route

def init_vehicle(active = 1.0000e+00, dynamic = 1.0000e+00,
                 speed = 0.0000e+00, x = -6.4630e-01, y = -3.2034e-01,
                 z = 0.0000e+00, dx = -1.0000e+00, dy = 2.4220e-04,
                 pitch = 0.0000e+00, half_length = 2.2145e+00,
                 half_width = 9.4450e-01, half_height = 6.3350e-01):
    vehicle = {
        VehicleField.ACTIVE: active,
        VehicleField.DYNAMIC: dynamic,
        VehicleField.SPEED: speed,
        VehicleField.X: x,
        VehicleField.Y: y,
        VehicleField.Z: z,
        VehicleField.DX: dx,
        VehicleField.DY: dy,
        VehicleField.PITCH: pitch,
        VehicleField.HALF_LENGTH: half_length,
        VehicleField.HALF_WIDTH: half_width,
        VehicleField.HALF_HEIGHT: half_height,
        VehicleField.UNSPECIFIED_12: 0.0,
        VehicleField.UNSPECIFIED_13: 0.0,
        VehicleField.UNSPECIFIED_14: 0.0,
        VehicleField.UNSPECIFIED_15: 0.0,
        VehicleField.UNSPECIFIED_16: 0.0,
        VehicleField.UNSPECIFIED_17: 0.0,
        VehicleField.UNSPECIFIED_18: 0.0,
        VehicleField.UNSPECIFIED_19: 0.0,
        VehicleField.UNSPECIFIED_20: 0.0,
        VehicleField.UNSPECIFIED_21: 0.0,
        VehicleField.UNSPECIFIED_22: 0.0,
        VehicleField.UNSPECIFIED_23: 0.0,
        VehicleField.UNSPECIFIED_24: 0.0,
        VehicleField.UNSPECIFIED_25: 0.0,
        VehicleField.UNSPECIFIED_26: 0.0,
        VehicleField.UNSPECIFIED_27: 0.0,
        VehicleField.UNSPECIFIED_28: 0.0,
        VehicleField.UNSPECIFIED_29: 0.0,
        VehicleField.UNSPECIFIED_30: 0.0,
        VehicleField.UNSPECIFIED_31: 0.0,
        VehicleField.UNSPECIFIED_32: 0.0
    }

    return vehicle

def create_driving_scenario_dataset(output_path):
    """
    주어진 형식에 맞는 자율 주행 시뮬레이션 데이터를 생성하여 .pkl 파일로 저장하는 함수.

    Args:
        output_path (str): 저장할 pkl 파일의 경로
    """
    # 데이터 생성
    ego_vehicle = init_ego_vehicle()

    ego_vehicle_tensor = torch.tensor([ego_vehicle[field] for field in EgoField])

    pedestrian_list = [init_pedestrian() for _ in range(20)]
    pedestrian_tensors = [torch.tensor([ped[field] for field in PedestrianField]) for ped in pedestrian_list]
    pedestrian_tensor = torch.stack(pedestrian_tensors)

    route_list = [init_route() for _ in range(30)]
    route_tensors = [torch.tensor([route[field] for field in RouteField]) for route in route_list]
    route_tensor = torch.stack(route_tensors)

    vehicle_list = [init_vehicle() for _ in range(30)]
    vehicle_tensors = [torch.tensor([vehicle[field] for field in RouteField]) for vehicle in vehicle_list]
    vehicle_tensor = torch.stack(vehicle_tensors)

    data = {
        "frame_num": 0,
        "observation": {
            "ego_vehicle_descriptor": ego_vehicle_tensor,
            "liable_vehicles": None,
            "pedestrian_descriptors": pedestrian_tensor,
            "route_descriptors": route_tensor,
            "vehicle_descriptors": vehicle_tensor,
        },
        "input_prompt": (
            "I'm observing 1 cars and 0 pedestrians.\n"
            "A moving car; Angle in degrees: -0.00; Distance: 7.80m; Direction of travel: same direction as me; My attention: 100%\n"
            "The distance to the closest intersection is 30.32m\n"
            "There is no traffic lights.\n"
            "My car 0.00m from the lane center and 0.00 degrees right off center.\n"
            "My current speed is 0.00 mph\n"
            "Steering wheel is 0.00% right.\n"
            "I need to go straight for at least 60.32m.\n\n"
            "Here are my actions:\n\n"
            "- Accelerator pedal 2%\n"
            "- Brake pedal 0%\n"
            "- Going to steer 7% to the left."
        ),
        "response_content": [
            {"question": "What are you observing?", "answer": "I'm observing 1 cars and 0 pedestrians."},
            {"question": "What are you paying attention to and why?",
             "answer": "I'm paying attention to the moving car at angle -0.00 because it could be a potential hazard and requires immediate attention."},
            {"question": "Are there any traffic lights/ what's the color of the traffic light?",
             "answer": "There is no traffic light."},
            {"question": "What's your current speed and steering angle/ current state?",
             "answer": "My current speed is 0.00 mph and my steering wheel is 0.00% right."},
            {"question": "What is your action and why / how are you going to drive in this situation and why?",
             "answer": "Here are my actions: -Accelerator pedal 2% -Brake pedal 0% -Going to steer 7% to the left. Reason being, I observed a car ahead and I'm steering to avoid collision while still maintaining a straight course."},
            {"question": "Summarize the current driving scenario in high-level / describe the current situation?",
             "answer": "There is one car in front of us, and we need to travel straight ahead for at least 60.32m. There are no traffic lights around, and we are not close to any intersection. My current speed is 0 miles per hour and the steering is 0% right."},
            {"question": "What is the distance to the closest intersection?",
             "answer": "The distance to the closest intersection is 30.32 m."},
            {"question": "What is the angle of the moving car?",
             "answer": "The angle of the moving car is -0.00 degrees."},
            {"question": "What direction is the car moving?",
             "answer": "The car is moving in the same direction as us."},
            {"question": "Is the car on the opposite lane?", "answer": "No, the car is not on the opposite lane."},
            {"question": "How many pedestrians are there?", "answer": "There are no pedestrians."},
            {"question": "What is the distance between the car and our car?",
             "answer": "The distance between the car and our car is 7.80 meters."},
            {"question": "What is the direction of travel of the car?",
             "answer": "The direction of travel of the car is the same direction as us."},
            {"question": "What actions are you going to take to avoid the car?",
             "answer": "Here are my actions: -Accelerator pedal 2% -Brake pedal 0% -Going to steer 7% to the left. Reason being, I observed a car ahead and I'm steering to avoid collision while still maintaining a straight course."},
            {"question": "What is the meaning of sharp right?",
             "answer": "Sharp right means an angle that is larger than 90 degrees and require the vehicle to make a sharp turn to the right."},
            {"question": "What is your favorite color?",
             "answer": "As an AI Driver, the question you asked is out of my scope, but I can try answer it. My favorite color is API Blue."},
        ]
    }

    # 데이터 저장
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    #################3. GENERATION###############################
    # output_path = 'data/test_dataset.pkl'
    #
    # # 데이터셋 생성 및 저장 함수 호출
    # create_driving_scenario_dataset(output_path)

    #####################2. EDITING ########################
    # 원래 pkl 파일 경로
    file_path = 'data/vqa_train_10k.pkl'
    # 저장할 새로운 pkl 파일 경로
    output_path = 'data/train_test.pkl'

    # 파일을 읽고 하나의 데이터를 새로운 파일로 저장하는 함수 호출
    load_and_save_items(file_path, output_path, count=10)

    # 원래 pkl 파일 경로
    file_path = 'data/vqa_test_1k.pkl'
    # 저장할 새로운 pkl 파일 경로
    output_path = 'data/val_test.pkl'

    # 파일을 읽고 하나의 데이터를 새로운 파일로 저장하는 함수 호출
    load_and_save_items(file_path, output_path, count=1)

    ####################1. READING###############################
    # pkl 파일의 경로를 설정합니다.
    # file_path = 'data/vqa_test_1k.pkl'
    file_path = ('data/train_test.pkl')

    # 파일을 읽고 내용을 출력하는 함수를 호출합니다.
    load_and_print_pkl(file_path)



