# Online fetch
from gradio_client import Client

client = Client("stabilityai/stable-diffusion-3.5-large")
result = client.predict(
		prompt="Hello!!",
		negative_prompt="Hello!!",
		seed=0,
		randomize_seed=True,
		width=1024,
		height=1024,
		guidance_scale=4.5,
		num_inference_steps=40,
		api_name="/infer"
)
print(result)
file_path = result[0]


# local fetch
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("capybara.png")
