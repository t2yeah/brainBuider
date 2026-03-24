from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "tokyotech-llm/Swallow-MS-7b-instruct-v0.1"

print("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

prompt = "次の音イベントから、考えられるシーンを3つ短く提案してください。\nAnimal: 0.44\nMeow: 0.38\nSpeech: 0.21\nSilence: 0.33"

messages = [
    {"role": "user", "content": prompt}
]

try:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
except Exception:
    # chat template が無い場合のフォールバック
    text = prompt

inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("===== OUTPUT =====")
print(result)
