# pylint: skip-file
from typing import List, Optional, Tuple, Union

import torch
from peft import PeftModelForCausalLM
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

import os
import sys
module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)

from models.vector_encoder import VectorEncoder, VectorEncoderConfig
from utils.vector_utils import VectorObservation, VectorObservationConfig


class LlamaForCausalLMVectorInput(LlamaForCausalLM):
    # 기존의 라마 모델을 상속받음
    def __init__(self, config):
        super().__init__(config)
        # 숫자 토큰에 대해서 가중치를 부여하기 위함이다
        # 이 특정 토큰에 대해서 loss를 3배의 가중치를 부여하게 된다
        self.weighted_loss_on_numbers = True
        if self.weighted_loss_on_numbers:
            # 이 토큰을 변환해 보니까 0부터 9까지의 숫자였다.
            # 이것을 통해서 확인할 수 있는 것은 , 숫자에 대한 민감도를 높게 주고 싶은 것이다
            number_tokens = [
                448,
                29900,
                29889,
                29896,
                29906,
                29941,
                29946,
                29945,
                29953,
                29955,
                29947,
                29929,
            ]  # -0.123456789
            weighted_mask = torch.ones(self.config.vocab_size)
            weighted_mask[number_tokens] = 3.0
            # 이 버퍼에 모델의 상태를 등록해서, 손실 계산시 사용함
            self.register_buffer("weighted_mask", weighted_mask)
        else:
            self.register_buffer("weighted_mask", None)

    # 텍스트 생성 과정에서, 모델에 전달할 입력 데이터를 준비하는 역할
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        route_descriptors=None,
        vehicle_descriptors=None,
        pedestrian_descriptors=None,
        ego_vehicle_descriptor=None,
        query_embeds=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        model_inputs.update(
            {
                "query_embeds": query_embeds,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        query_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Ingest vectors if in generation mode (query_embeds is not None)
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        elif inputs_embeds is None and input_ids is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 여기서 query_embeds가 새롭게 추가된 주변 정보를 의미한다고 한다
        if query_embeds is not None and past_key_values is None:
            # 새롭게 벡터를 증가시킨 값
            # 새롭게 증가시킨 벡터는 input에 대한 값이 된다
            inputs_embeds, attention_mask, _ = ingest_vectors(
                input_ids,
                inputs_embeds,
                query_embeds,
                attention_mask,
            )
            position_ids = None

        # from modeling_llama.py
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # Llama 모델의 디코더를 통해서 hidden state를 출력함
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # 이것에 대해서 모델의 마지막 레이어인 lm_head를 통해서 logits 을 계산함
        # 레이블이 주어진 경우에는 cross entropy loss를 통해서 예측값과 레이블 간의 손실을 계산함
        # 이 중에서도 weighted mask( 숫자에 대해서 가중치가 주어진 것에 대해서) loss를 계산하게 된다
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(weight=self.weighted_mask)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # 출력 형식을 결정하게 된다
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output


        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 기존의 LLM모델에 대해서, 추가적인 입력 벡터를 포함 시켜서 학습을 수행함
# Peft를 상속받아서 , LoRA를 통해서 효율적으로 fine Tuning 수행할 수 있음
# Peft는 parameter efficient fine tuning의 약자로, 파인 튜닝하는데 필요한 파라미터 수를 크게 줄이는 방법이다
# 이것을 위해서 사용되는 방법이 두가지 이다
# 먼저 low rank adaptaion 방법을 통해서는, 기존 모델의 큰 가중치 행렬은 수정하지 않고,
# 저차원 행렬을 통해서 가중치를 조정하는 것이다
# 두번째 방법인 adapter tuning은 기존 모델에 대해서 adapter계층을 삽입해서, 이것만 학습 시키는 것이다
class VectorLMWithLoRA(PeftModelForCausalLM):
    # 여기서 num_vector_token은 쿼리의 의미인가봐
    # 차량 및 보행자 정보를 몇개의 토큰으로 변환할 것인지 설정하는 것이다
    def __init__(self, model, peft_config, num_vector_tokens=64):
        super().__init__(model, peft_config)
        self.num_vector_tokens = num_vector_tokens
        # 이 클래스를 통해서, 입력된 차량 및 보행자 정보를 인코딩 하는 것이다
        self.vector_encoder = VectorEncoder(
            VectorEncoderConfig(), VectorObservationConfig(), num_vector_tokens
        )
        # 그리고 llm_proj라는 선형 변환을 통해서, 이것을 llm모델의 임베딩 크기에 맞추게 되는 것이다
        self.llm_proj = torch.nn.Linear(
            self.vector_encoder.out_features, self.config.hidden_size
        )
        self.to(model.device)
        self.modules_to_save = ["vector_encoder", "llm_proj"]

        # 텍스트 생성을 위한 설정들이다. 최종 데이터 생성을 위한 값들
        self.generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=1,
            use_cache=False,
            do_sample=True,
            max_length=384,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            _from_model_config=False,
        )

    # vector와 prompt에 대해서 llm에게 임베딩 하는 과정이다
    # 임베딩이라는 것은 고차원 데이터를 저차원 공간으로 변환하는 것이다
    def embed_vector_and_prompt(
        self,
        input_ids,
        attention_mask,
        labels,
        route_descriptors,
        vehicle_descriptors,
        pedestrian_descriptors,
        ego_vehicle_descriptor,
    ):
        # Create the vector observation
        vector_obs = VectorObservation(
            route_descriptors=route_descriptors,
            vehicle_descriptors=vehicle_descriptors,
            pedestrian_descriptors=pedestrian_descriptors,
            ego_vehicle_descriptor=ego_vehicle_descriptor,
        )
        # 입력된 정보에 대해서, 인코딩을 수행하게 된다
        encoder_output = self.vector_encoder(vector_obs)
        # 이것을 llm_proj를 통해서 llm모델과 임베딩 크기를 일치시키게 된다
        inputs_vector = self.llm_proj(
            encoder_output
        )  # Adjust this line for multiple tokens

        # Generate token embeddings
        # inputs_embeds를 만들게 되고
        # 여기서 input_ids의 의미는 , 텍스트 데이터를 모델에 입력하기 위해서 숫자로 변환된 텍스트 시퀀스를 의미한다
        # 예상되는 바로는, prompt 정보가 변환된 정보가 input_ids가 될 것이고,
        # 관측된 정보가 ingest vectors를 통해서 추가로 확장되는 것 같다
        inputs_embeds = self.model.model.embed_tokens(input_ids)

        # Concatenate the vector embeddings with the token embeddings
        #input ids에 대해서 새롭게 생기게 된 input vector를 넣어서 새롭게 정의를 하게 되는 것이다

        new_inputs_embeds, new_attention_mask, new_labels = ingest_vectors(
            input_ids,
            inputs_embeds,
            inputs_vector,
            attention_mask,
            labels,
        )

        return new_inputs_embeds, new_attention_mask, new_labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        route_descriptors=None,
        vehicle_descriptors=None,
        pedestrian_descriptors=None,
        ego_vehicle_descriptor=None,
        **kwargs,  # those are 'user_input_ids', 'user_attention_mask'
    ):
        inputs_embeds, attention_mask, labels = self.embed_vector_and_prompt(
            input_ids,
            attention_mask,
            labels,
            route_descriptors,
            vehicle_descriptors,
            pedestrian_descriptors,
            ego_vehicle_descriptor,
        )

        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        loss = outputs.loss

        return {"loss": loss}

    def generate(self, **kwargs):
        route_descriptors = kwargs["route_descriptors"]
        vehicle_descriptors = kwargs["vehicle_descriptors"]
        pedestrian_descriptors = kwargs["pedestrian_descriptors"]
        ego_vehicle_descriptor = kwargs["ego_vehicle_descriptor"]

        vector_obs = VectorObservation(
            route_descriptors=route_descriptors,
            vehicle_descriptors=vehicle_descriptors,
            pedestrian_descriptors=pedestrian_descriptors,
            ego_vehicle_descriptor=ego_vehicle_descriptor,
        )
        encoder_output = self.vector_encoder(vector_obs)
        query_embeds = self.llm_proj(encoder_output)

        kwargs["query_embeds"] = query_embeds
        kwargs["input_ids"] = kwargs.pop("user_input_ids")
        kwargs["attention_mask"] = kwargs.pop("user_attention_mask")
        if "generation_config" not in kwargs:
            kwargs[
                "generation_config"
            ] = (
                self.generation_config
            )  # Override the generation config to make the padding tokens correct
        # 최종적으로 결과 값을 생성하는 단계이다 
        outputs = self.base_model.generate(**kwargs)
        return outputs

# 특정 토큰 시퀀스가 발견된 위치에, 추가적인 정보를 삽입해서 모델의 입력을 확장하는데 사용되는 방
def ingest_vectors(
    input_ids, inputs_embeds, input_vectors, attention_mask, labels=None
):
    batch_size = input_ids.shape[0]
    # 토큰의 수 (샘플 시퀀스의 길이 )
    seq_length = input_ids.shape[1]
    # 삽입할 벡터의 길이
    vector_length = input_vectors.shape[1]
    # Find the position of the specific token sequence (10567 and 29901) for each instance in the batch
    # 특정한 토큰의 위치를 찾는다
    token_sequence = torch.tensor([10567, 29901], device=input_ids.device)
    # 해당 토큰이 발생하는 위치를 불리언 배열로 표시한다
    positions = (input_ids[:, :-1] == token_sequence[0]) & (
        input_ids[:, 1:] == token_sequence[1]
    )

    # Add 3 to get the vector insertion positions, and handle cases where the sequence is not found
    # 토큰 시퀀스가 발견되는 위치에 대해서 3을 더해서 삽입 위치 계산
    # 왜 3을 더한다음에 설정하게 되나면, 특정 역할을 수행하게 하기 위함이라고 ㅎ나다
    vector_input_positions = torch.argmax(positions.float(), dim=1) + 3
    # 발견되지 않은 경우, 위치를 0으로 설정
    vector_input_positions[vector_input_positions == 3] = 0
    # 만약에 초과하게 되면 끝에 삽입하게 된다
    vector_input_positions[vector_input_positions > seq_length] = seq_length

    # Create tensors to hold the updated inputs_embeds, attention_mask, and labels
    new_inputs_embeds = torch.zeros(
        batch_size,
        seq_length + vector_length,
        inputs_embeds.shape[2],
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )
    new_attention_mask = torch.zeros(
        batch_size,
        seq_length + vector_length,
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    new_labels = (
        torch.zeros(
            batch_size,
            seq_length + vector_length,
            device=labels.device,
            dtype=labels.dtype,
        )
        if labels is not None
        else None
    )
    # 배치 사이즈 별로
    for b in range(batch_size):
        # 삽입해야 하는 위치에 대해서
        vector_input_position = vector_input_positions[b]
        if vector_input_position == 0:
            vector_input_position = 1  # Insert the vector embeddings at position 1 if the token_sequence is not found (avoid the bos_token)
        # input embeds 업데이트
        new_inputs_embeds[b, :vector_input_position] = inputs_embeds[
            b, :vector_input_position
        ]
        new_inputs_embeds[
            b, vector_input_position : vector_input_position + vector_length
        ] = input_vectors[b]
        new_inputs_embeds[b, vector_input_position + vector_length :] = inputs_embeds[
            b, vector_input_position:
        ]

        # 에텐션 마스크 업데이트
        new_attention_mask[b, :vector_input_position] = attention_mask[
            b, :vector_input_position
        ]
        new_attention_mask[
            b, vector_input_position : vector_input_position + vector_length
        ] = 1
        new_attention_mask[b, vector_input_position + vector_length :] = attention_mask[
            b, vector_input_position:
        ]

        # 라벨 업데이트
        if labels is not None:
            new_labels[b, :vector_input_position] = labels[b, :vector_input_position]
            new_labels[
                b, vector_input_position : vector_input_position + vector_length
            ] = -100
            new_labels[b, vector_input_position + vector_length :] = labels[
                b, vector_input_position:
            ]

    return new_inputs_embeds, new_attention_mask, new_labels

from transformers import LlamaConfig
from torch.cuda.amp import autocast
def test_llama_for_causal_lm_vector_input(
    base_model: str = "baffo32/decapoda-research-llama-7B-hf",  # the only required argument
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Tuple = ("q_proj", "k_proj", "v_proj", "o_proj"),
    resume_from_checkpoint: str = "pretrained_model/",
    load_in_8bit: bool = True,
):
    # set DDP flags
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if ddp:
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    # Configuration and model instantiation
    llama_model = LlamaForCausalLMVectorInput.from_pretrained(
        base_model,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        device_map=device_map,
    )



def test_ingest_vectors():
    # 모의 데이터 설정
    batch_size = 2
    seq_length = 10
    embed_dim = 8
    vector_length = 3

    # 모의 input_ids (토큰 ID)
    input_ids = torch.tensor(
        [
            [101, 10567, 29901, 200, 300, 400, 500, 600, 700, 102],
            [101, 200, 300, 400, 500, 10567, 29901, 600, 700, 102],
        ],
        dtype=torch.long,
    )

    # 모의 inputs_embeds (기존 임베딩 값)
    inputs_embeds = torch.randn(batch_size, seq_length, embed_dim)

    # 모의 input_vectors (새로 삽입할 벡터)
    input_vectors = torch.randn(batch_size, vector_length, embed_dim)

    # 모의 attention_mask (기존 attention mask)
    attention_mask = torch.ones(batch_size, seq_length)

    # 모의 labels (라벨)
    labels = torch.randint(0, 100, (batch_size, seq_length), dtype=torch.long)

    # 함수 호출
    new_inputs_embeds, new_attention_mask, new_labels = ingest_vectors(
        input_ids, inputs_embeds, input_vectors, attention_mask, labels
    )

    # 결과 출력
    print("New Inputs Embeds Shape:", new_inputs_embeds.shape)
    print("New Inputs Embeds:", new_inputs_embeds)
    print("New Attention Mask Shape:", new_attention_mask.shape)
    print("New Attention Mask:", new_attention_mask)
    print("New Labels Shape:", new_labels.shape)
    print("New Labels:", new_labels)

    # 기대하는 크기 확인
    assert new_inputs_embeds.shape == (batch_size, seq_length + vector_length, embed_dim)
    assert new_attention_mask.shape == (batch_size, seq_length + vector_length)
    assert new_labels.shape == (batch_size, seq_length + vector_length)

    print("All assertions passed!")

def test_token_to_word(tokenizer, token_ids):
    # 주어진 token_ids를 텍스트로 변환
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    words = tokenizer.decode(token_ids)

    # 토큰 ID와 대응되는 텍스트 출력
    for token_id, token in zip(token_ids, tokens):
        print(f"Token ID: {token_id} -> Token: {token}")

    # 디코드된 텍스트 출력
    print(f"\nDecoded words: {words}")

if __name__ == "__main__":
    from transformers import LlamaTokenizer

    # test_llama_for_causal_lm_vector_input()
    # test_ingest_vectors()

    # Llama 토크나이저 로드
    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")

    # 확인할 토큰 ID 목록
    # number_tokens = [
    #     448, 29900, 29889, 29896, 29906, 29941, 29946,
    #     29945, 29953, 29955, 29947, 29929
    # ]  # -0.123456789
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,

    number_tokens = [
        10567, 29901
    ]
    # Input, ?

    # 토큰 ID를 단어로 변환하는 테스트 실행
    test_token_to_word(tokenizer, number_tokens)