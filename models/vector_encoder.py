from math import sqrt

import torch
import torch.nn as nn

import os
import sys
module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)

from models.mlp import MLP
from models.transformer import Perceiver
from utils.vector_utils import VectorObservation, VectorObservationConfig


class VectorEncoderConfig:
    model_dim: int = 256
    num_latents: int = 32
    num_blocks: int = 7
    num_heads: int = 8


class VectorEncoder(nn.Module):
    def __init__(
        self,
        encoder_config: VectorEncoderConfig,
        observation_config: VectorObservationConfig,
        num_queries: int,
    ):
        super().__init__()

        model_dim = encoder_config.model_dim
        self.ego_vehicle_encoder = MLP(
            VectorObservation.EGO_DIM, [model_dim], model_dim
        )
        self.vehicle_encoder = MLP(
            VectorObservation.VEHICLE_DIM, [model_dim], model_dim
        )
        self.pedestrian_encoder = MLP(
            VectorObservation.PEDESTRIAN_DIM, [model_dim], model_dim
        )
        self.route_encoder = MLP(VectorObservation.ROUTE_DIM, [model_dim], model_dim)
        self.route_embedding = nn.Parameter(
            torch.randn((observation_config.num_route_points, model_dim))
            / sqrt(model_dim)
        )

        self.perceiver = Perceiver(
            model_dim=model_dim,
            context_dim=model_dim,
            num_latents=encoder_config.num_latents,
            num_blocks=encoder_config.num_blocks,
            num_heads=encoder_config.num_heads,
            num_queries=num_queries,
        )

        self.out_features = model_dim

    def forward(self, obs: VectorObservation):
        batch = obs.route_descriptors.shape[0]
        device = obs.route_descriptors.device

        route_token = self.route_embedding[None] + self.route_encoder(
            obs.route_descriptors
        )
        vehicle_token = self.vehicle_encoder(obs.vehicle_descriptors)
        pedestrian_token = self.pedestrian_encoder(obs.pedestrian_descriptors)
        context = torch.cat((route_token, pedestrian_token, vehicle_token), -2)
        context_mask = torch.cat(
            (
                torch.ones(
                    (batch, route_token.shape[1]), device=device, dtype=bool
                ),  # route
                obs.pedestrian_descriptors[:, :, 0] != 0,  # pedestrians
                obs.vehicle_descriptors[:, :, 0] != 0,  # vehicles
            ),
            dim=1,
        )

        ego_vehicle_state = obs.ego_vehicle_descriptor
        ego_vehicle_feat = self.ego_vehicle_encoder(ego_vehicle_state)

        feat, _ = self.perceiver(ego_vehicle_feat, context, context_mask=context_mask)
        feat = feat.view(
            batch,
            self.perceiver.num_queries,
            feat.shape[-1],
        )

        return feat


def test_vector_encoder():
    # Define mock configurations
    encoder_config = VectorEncoderConfig()
    # 관찰을 위한 객체 설정
    observation_config = VectorObservationConfig()

    # Create a mock VectorObservation with random tensor inputs
    batch_size = 4
    num_route_points = observation_config.num_route_points
    max_vehicles = 5
    max_pedestrians = 3

    # descriptor를 랜덤하게 작성함
    route_descriptors = torch.randn(batch_size, num_route_points, VectorObservation.ROUTE_DIM)
    vehicle_descriptors = torch.randn(batch_size, max_vehicles, VectorObservation.VEHICLE_DIM)
    pedestrian_descriptors = torch.randn(batch_size, max_pedestrians, VectorObservation.PEDESTRIAN_DIM)
    ego_vehicle_descriptor = torch.randn(batch_size, VectorObservation.EGO_DIM)

    # Create the VectorObservation object
    # 이제 관찰된 random 한 벡터들을 전달하게 됨 (실제 값이 얼마인지는 출력이 해봐야 알거 같다)
    # 벡터 형식으로 출력되는데 이게 실제 텍스트 데이터로 어떻게 변환하지? lanGen 값이 필요하다
    obs = VectorObservation(
        route_descriptors=route_descriptors,
        vehicle_descriptors=vehicle_descriptors,
        pedestrian_descriptors=pedestrian_descriptors,
        ego_vehicle_descriptor=ego_vehicle_descriptor
    )

    print(obs)
    # Initialize the VectorEncoder model
    num_queries = 3
    vector_encoder = VectorEncoder(encoder_config, observation_config, num_queries)

    print(vector_encoder)
    # Forward pass
    output = vector_encoder(obs)

    # Output the shape of the results
    print(f"Output shape: {output.shape}")
    print(output)
    # 이렇게 되면 실제 vector encoder를 거쳐서 값이 출력되게 된다
    assert output.shape == (batch_size, num_queries, encoder_config.model_dim), \
        f"Expected shape {(batch_size, num_queries, encoder_config.model_dim)}, got {output.shape}"



if __name__ == "__main__":
    test_vector_encoder()