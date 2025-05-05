# FILE NAME : ExecuteModel.py
import torch
import torch.nn as nn
import re
from PIL import Image
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    SiglipImageProcessor,
    SiglipVisionModel,
)
import os

Model_LLM = ""
tokenizer = None
model = None
vision_model = None
processor = None
projection_module = None

def clean_caption(text):
    # Buang semua kalimat pembuka umum
    text = re.sub(r"(?i)^.*?:\s*", "", text)
    # Hilangkan kalimat penutup jika ada
    text = re.sub(r"(?i)let me know.*", "", text)
    text.replace('"', '')
    return text.strip()
    
def tokenizer_image_token(prompt, tokenizer, image_token_index=-200):
    prompt_chunks = prompt.split("<image>")
    tokenized_chunks = [tokenizer(chunk).input_ids for chunk in prompt_chunks]
    input_ids = tokenized_chunks[0]

    for chunk in tokenized_chunks[1:]:
        input_ids.append(image_token_index)
        input_ids.extend(chunk[1:])  # Exclude BOS token on nonzero index

    return torch.tensor(input_ids, dtype=torch.long)


def process_tensors(input_ids, image_features, embedding_layer):
    split_indices = (input_ids == -200).nonzero(as_tuple=True)
    if len(split_indices[1]) == 0:
        raise ValueError("Token <image> tidak ditemukan dalam input_ids")
    split_index = split_indices[1][0]

    input_ids_1 = input_ids[:, :split_index]
    input_ids_2 = input_ids[:, split_index + 1:]

    embeddings_1 = embedding_layer(input_ids_1)
    embeddings_2 = embedding_layer(input_ids_2)

    device = image_features.device
    token_embeddings_part1 = embeddings_1.to(device)
    token_embeddings_part2 = embeddings_2.to(device)

    concatenated_embeddings = torch.cat(
        [token_embeddings_part1, image_features, token_embeddings_part2], dim=1
    )

    attention_mask = torch.ones(
        concatenated_embeddings.shape[:2], dtype=torch.long, device=device
    )
    return concatenated_embeddings, attention_mask


def unload_model(model, vision_model, projection_module):
    # Hapus model dan objek lain yang menggunakan GPU
    del model
    del vision_model
    del projection_module
    
    # Memaksa pembersihan cache GPU
    torch.cuda.empty_cache()
    
    # Panggil garbage collection untuk objek Python yang tidak digunakan
    import gc
    gc.collect()

def initialize_models():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        Model_LLM, use_fast=True
    )
    model = LlamaForCausalLM.from_pretrained(
        Model_LLM,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    for param in model.base_model.parameters():
        param.requires_grad = False

    model_name = "google/siglip-so400m-patch14-384"
    vision_model = SiglipVisionModel.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    processor = SiglipImageProcessor.from_pretrained(model_name)

    vision_model = vision_model.to("cuda")

    return tokenizer, model, vision_model, processor


class ProjectionModule(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size):
        super(ProjectionModule, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.model(x)


def load_projection_module(mm_hidden_size=1152, hidden_size=4096, device="cuda"):
    projection_module = ProjectionModule(mm_hidden_size, hidden_size)
    checkpoint = torch.load("./model/mm_projector.bin")
    checkpoint = {k.replace("mm_projector.", ""): v for k, v in checkpoint.items()}
    projection_module.load_state_dict(checkpoint)
    projection_module = projection_module.to(device).half()
    return projection_module


def answer_question(image_path, question, tokenizer, model, vision_model, processor, projection_module):
    image = Image.open(image_path).convert("RGB")
    tokenizer.eos_token = "<|eot_id|>"

    question = "<image>" + question
    prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    input_ids = tokenizer_image_token(prompt, tokenizer).unsqueeze(0).to(model.device)

    image_inputs = processor(
        images=[image],
        return_tensors="pt",
        do_resize=True,
        size={"height": 384, "width": 384},
    )["pixel_values"].squeeze(0).to("cuda")

    with torch.inference_mode():
        image_forward_outs = vision_model(
            image_inputs.unsqueeze(0).to(dtype=torch.float16),
            output_hidden_states=True,
        )
        image_features = image_forward_outs.hidden_states[-2]
        projected_embeddings = projection_module(image_features).to("cuda")

        embedding_layer = model.get_input_embeddings()
        new_embeds, attn_mask = process_tensors(input_ids, projected_embeddings, embedding_layer)
        new_embeds = new_embeds.to(model.device)
        attn_mask = attn_mask.to(model.device)

        generated_ids = model.generate(
            inputs_embeds=new_embeds,
            attention_mask=attn_mask,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=2000,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )[0]

        return tokenizer.decode(generated_ids, skip_special_tokens=True)


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"])


def multi_image(Model_Name, Prompt_Sting, Image_Path):
    global Model_LLM, tokenizer, model, vision_model, processor, projection_module

    Model_LLM = Model_Name
    if tokenizer is None or model is None or vision_model is None or processor is None:
        tokenizer, model, vision_model, processor = initialize_models()

    if projection_module is None:
        projection_module = load_projection_module()

    result = answer_question(
        Image_Path,
        Prompt_Sting,
        tokenizer,
        model,
        vision_model,
        processor,
        projection_module,
    )

    unload_model(model, vision_model, projection_module)

    txt_path = os.path.splitext(Image_Path)[0] + ".txt"
    hasil = clean_caption(result.strip())
    #hasil = result.strip()
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(hasil)

    print("Hasil deskripsi:")
    print(hasil)
    return hasil



def single_image(Model_Name, Prompt_Sting, Image_Path):
    global Model_LLM, tokenizer, model, vision_model, processor, projection_module

    Model_LLM = Model_Name
    if tokenizer is None or model is None or vision_model is None or processor is None:
        tokenizer, model, vision_model, processor = initialize_models()

    if projection_module is None:
        projection_module = load_projection_module()

    result = answer_question(
        Image_Path,
        Prompt_Sting,
        tokenizer,
        model,
        vision_model,
        processor,
        projection_module,
    )

    unload_model(model, vision_model, projection_module)

    txt_path = os.path.splitext(Image_Path)[0] + ".txt"
    hasil = clean_caption(result.strip())
    #hasil = result.strip()
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(hasil)

    print("Hasil deskripsi:")
    print(hasil)
    print("-" * 60)
    return hasil
