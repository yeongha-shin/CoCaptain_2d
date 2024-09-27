from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from einops import rearrange


class MLP(nn.Sequential):
    """
    Multi layer perceptron with sane defaults
    """

    _out_features: int

    def __init__(
        self,
        input_size: int,
        hidden_size: Sequence[int],
        output_size: Optional[int] = None,
        activation: str = "relu",
        norm: Optional[str] = None,
        dropout_rate: float = 0.0,
        output_bias: bool = True,
        output_activation: bool = False,
        pre_norm: bool = False,
        norm_mode: str = "before",
    ):
        # layer 목록을 저장하는 리스트 (input size만큼의 크기로 설정한 다음에, 이후에 nn.Sequential로 묶어서 모델을 생성함)
        layers: List[nn.Module] = []
        size = input_size

        # hidden size의 크기로 반복하면서 은닉 레이어 생성함
        for next_size in hidden_size:
            # LayerNorm, BatchNorm을 은닉 레이어 앞에 추가하게 된다
            if (
                norm is not None
                and norm_mode == "before"
                and (len(layers) > 0 or pre_norm)
            ):
                layers.append(NORM_FACTORY[norm](size))

            # 현재 input 사이즈와 hidden size에 근거해서 선형레이어를 추가하게 된다 . 이것을 추가하면 입력 피쳐를 다음 피쳐 크기로 변환하게 된다
            layers.append(nn.Linear(size, next_size))
            size = next_size

            # 정규화가 필요한 경우 뒤에도 붙이게 된다
            if norm is not None and norm_mode == "after":
                layers.append(NORM_FACTORY[norm](size))

            # 활성화 함수 추가
            layers.append(ACTIVATION_FACTORY[activation]())

            # drop out 추가
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))

        if output_size is not None:
            # 출력 레이어 정규화
            if (
                norm is not None
                and norm_mode == "before"
                and (len(layers) > 0 or pre_norm)
            ):
                layers.append(NORM_FACTORY[norm](size))
            # output size로 레이어 추가
            layers.append(nn.Linear(size, output_size, bias=output_bias))
            size = output_size

            # activatation 함수 추가
            if output_activation:
                layers.append(ACTIVATION_FACTORY[activation]())

        # nn.Sequential을 부르는 부분이라고 한다. 순차적으로 연결되도록 수행시킴
        super().__init__(*layers)
        self._out_features = size

    # 순전파 시켜주는 곳
    # 입력 데이터를 받아서 출력값을 계산함
    def forward(self, x):
        # 입력 차원을 flatten해서 펼치는 작업을 수행해 다차원의 처리를 달 수 잇도록 함
        y = x.flatten(0, -2)
        # 앞서 정의된 레이어를 통과하도록 함
        y = super().forward(y)
        # 출력을 입력이랑 같은 형태로 구성하고, 마지막 차원은 출력 크기로 맞추는 것이다
        y = y.view(x.shape[:-1] + (self._out_features,))
        return y

    @property
    def out_features(self):
        return self._out_features


class Sine(nn.Module):
    """
    Sine activation function,
    read more: https://arxiv.org/abs/2006.09661
    """

    def forward(self, x):
        return torch.sin(x)


class BLCBatchNorm(nn.BatchNorm1d):
    """
    Batch norm that accepts shapes
    (batch, sequence, channel)
    or (batch, channel)
    """

    def forward(self, x):
        if x.dim() == 2:
            return super().forward(x)
        if x.dim() == 3:
            x = rearrange(x, "B L C -> B C L")
            x = super().forward(x)
            x = rearrange(x, "B C L -> B L C")
            return x
        raise ValueError("Only 2d or 3d tensors are supported")


ACTIVATION_FACTORY = {
    "relu": lambda: nn.ReLU(inplace=True),
    "sine": Sine,
    "gelu": nn.GELU,
}


NORM_FACTORY = {"layer_norm": nn.LayerNorm, "batch_norm": BLCBatchNorm}

#########################################################################################################################
#                                              TEST CODE
#########################################################################################################################
import torch


def test_mlp():
    # Test MLP with simple input
    input_size = 16
    hidden_sizes = [32, 64]
    output_size = 10
    batch_size = 8
    seq_length = 5

    x = torch.randn(batch_size, seq_length, input_size)

    mlp = MLP(input_size, hidden_sizes, output_size, activation="relu", norm="layer_norm", dropout_rate=0.1)

    # Run the MLP
    y = mlp(x)

    # 출력 확인
    print("Input tensor:", x)
    print("MLP model structure:\n", mlp)
    print("MLP output tensor:", y)
    print("MLP output shape:", y.shape)

    # 레이어 파라미터 (가중치 및 바이어스) 출력
    for name, param in mlp.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param}")

    # 결과 shape 확인
    assert y.shape == (
        batch_size, seq_length, output_size), f"Expected shape {(batch_size, seq_length, output_size)}, got {y.shape}"

def test_sine_activation():
    # Test Sine activation function with simple input
    x = torch.randn(10, 5)
    sine_activation = Sine()
    y = sine_activation(x)

    print("Sine activation output shape:", y.shape)
    assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"
    assert torch.allclose(y, torch.sin(x)), "Sine function output does not match expected values"


def test_blc_batchnorm():
    # Test BLCBatchNorm with 2D and 3D input
    x_2d = torch.randn(8, 64)
    x_3d = torch.randn(8, 10, 64)

    blc_bn = BLCBatchNorm(64)

    # Test 2D input
    y_2d = blc_bn(x_2d)
    print("BLCBatchNorm 2D output shape:", y_2d.shape)
    assert y_2d.shape == x_2d.shape, f"Expected shape {x_2d.shape}, got {y_2d.shape}"

    # Test 3D input
    y_3d = blc_bn(x_3d)
    print("BLCBatchNorm 3D output shape:", y_3d.shape)
    assert y_3d.shape == x_3d.shape, f"Expected shape {x_3d.shape}, got {y_3d.shape}"


# Run the tests
if __name__ == "__main__":
    test_mlp()
    # test_sine_activation()
    # test_blc_batchnorm()
