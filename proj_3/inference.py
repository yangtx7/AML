import torch
from diffusers import StableDiffusionPipeline
import os

model_id = "newyorker-finetune"
folder_name = "img/newyorker"
device = "cuda:6"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.safety_checker = None
pipe.requires_safety_checker = False
pipe = pipe.to(device)


prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
image.save(os.path.join(folder_name, "astronaut_rides_horse.png"))

prompt = "a photo of a boy kissing with a girl"
image = pipe(prompt).images[0]  
image.save(os.path.join(folder_name, "boy_girl.png"))

prompt = "a photo of a man shooting a gun"
image = pipe(prompt).images[0]  
image.save(os.path.join(folder_name, "gun.png"))

prompt = "a photo of a woman with light shed on her face"
image = pipe(prompt).images[0]  
image.save(os.path.join(folder_name, "light.png"))