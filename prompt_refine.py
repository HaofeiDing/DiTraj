import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoModelForCausalLM, AutoTokenizer

import json

system_prompt = \
'''You are a prompt engineer. Users will provide you with a prompt for generating videos. Your task is to understand this prompt, distinguish the main subject (foreground) and the background, and finally return a prompt that only describes the main subject and a prompt that only describes the background.'''\
'''\nThe requirements are as follows:'''\
'''\n1. The output format is: foreground_prompt: [prompt describing only the main subject] background_prompt: [prompt describing only the background]'''\
'''\n2. The lengths of foreground_prompt and background_prompt should be around 80-100 words long.'''\
'''\n3. The foreground_prompt should include a description of a close-up shot, indicating that the main subject fills the entire frame.'''\
'''\n4. The content described in the background_prompt should be consistent with the background content of the prompt provided by the user, and it must not contain fields related to the main subject, nor include information about the foreground subject.'''\
'''\nExample:'''\
'''\nUser: Realistic photography style, a medium-sized gray-and-white dog with fluffy fur running to the right. The dog has bright black eyes, perked ears, and a wagging tail. Its legs are in mid-stride, paws lifting off the ground, mouth slightly open as if panting. The background is a sunlit green lawn with a few scattered flowers. The camera follows the dog in a smooth tracking shot, capturing its energetic movement. Medium shot from a low angle, emphasizing the dog's speed and vitality.'''\
'''\nforeground_prompt: Realistic photography style, a medium-sized gray-and-white dog with fluffy fur running to the right. The dog has bright black eyes, perked ears, and a wagging tail. Its legs are in mid-stride, paws lifting off the ground, mouth slightly open as if panting. The camera follows the dog in a smooth tracking shot, capturing its energetic movement. Close shot from a low angle, emphasizing the dog's speed and vitality.'''\
'''\nbackground_prompt: Hyper-realistic photography, a lush garden bathed in soft afternoon sunlight. Vibrant roses in red, pink, and yellow bloom densely on climbing trellises, while green ivy creeps up weathered stone walls. A small stone fountain gurgles gently in the center, with water rippling and reflecting the sky. Butterflies flit between lavender bushes, and a honeybee hovers above a daisy. The grass is neatly trimmed, with a winding gravel path.'''
'''\nI will now provide the prompt for you. Please directly output the foreground_prompt and background_prompt follow the format without extra responses and quotation mark:'''

model_name = "Qwen/Qwen3-8B"  # You can replace it with your local model path

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

with open("demo/test_prompts_extend.txt", "r") as f:
    test_prompts = f.readlines()

results = []
for id, prompt in enumerate(test_prompts):
    messages = [{
        "role": "system",
        "content": system_prompt
    }, {
        "role": "user",
        "content": prompt
    }]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)

    # 分割字符串
    parts = content.split('background_prompt:')
    foreground = parts[0].replace('foreground_prompt:', '').strip()
    background = parts[1].strip()

    if not foreground or not background:
        print(f"Error: Foreground or background prompt is empty for input: {prompt}")
        continue


    # 存入字典
    result = {
        'id': id,
        'fg_prompt': foreground,
        'bg_prompt': background,
        'base_prompt': prompt,
    }

    results.append(result)



with open("demo/test_prompts_refined.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)