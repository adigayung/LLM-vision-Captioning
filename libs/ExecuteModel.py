# FILE NAME : ExecuteModel.py

import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    SiglipImageProcessor,
    SiglipVisionModel,
)
import torch.nn as nn

class ProjectionModule(nn.Module):
    def __init__(self, mm_hidden_size=1152, hidden_size=4096):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.model(x)

def tokenizer_image_token(prompt, tokenizer, image_token_index=-200):
    chunks = prompt.split("<image>")
    ids = [tokenizer(chunk).input_ids for chunk in chunks]
    result = ids[0]
    for chunk in ids[1:]:
        result.append(image_token_index)
        result.extend(chunk[1:])
    return torch.tensor(result).unsqueeze(0)

def process_tensors(input_ids, image_features, embed_layer):
    idx = (input_ids == -200).nonzero(as_tuple=True)[1][0]
    ids_1, ids_2 = input_ids[:, :idx], input_ids[:, idx+1:]
    emb_1 = embed_layer(ids_1)
    emb_2 = embed_layer(ids_2)
    return torch.cat([emb_1, image_features, emb_2], dim=1), torch.ones((1, emb_1.size(1) + image_features.size(1) + emb_2.size(1)), dtype=torch.long).to(image_features.device)

def execute_model(model_name, prompt_text, image_path):


    # Load semua
    tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct", use_fast=True)
    model = LlamaForCausalLM.from_pretrained(
        "unsloth/llama-3-8b-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
    )
    model.eval()

    processor = SiglipImageProcessor.from_pretrained(model_name)
    vision_model = SiglipVisionModel.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda").eval()

    projection = ProjectionModule().to("cuda").half()
    projection.load_state_dict(torch.load("mm_projector.bin", map_location="cuda"))

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to("cuda")

    with torch.no_grad():
        vision_out = vision_model(pixel_values)
        image_features = projection(vision_out.hidden_states[-2])

    prompt = f"<|start_header_id|>user<|end_header_id|>\n\n<image>{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    input_ids = tokenizer_image_token(prompt, tokenizer).to(model.device)

    embed_layer = model.get_input_embeddings()
    embeds, attn_mask = process_tensors(input_ids, image_features, embed_layer)
    embeds, attn_mask = embeds.to(model.device), attn_mask.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=embeds,
            attention_mask=attn_mask,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(result)
    return result

if __name__ == "__main__":
    execute_model("unsloth/llama-3-8b-Instruct", "what this image ?", "image.jpg")