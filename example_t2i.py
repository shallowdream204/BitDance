from modeling.t2i_pipeline import BitDanceT2IPipeline

model_path = 'models/BitDance-14B-64x'
# model_path = 'models/BitDance-14B-16x'
device = 'cuda'

pipe = BitDanceT2IPipeline(model_path=model_path, device=device)

prompt = "A gritty, noir-style comic book panel. A detective in a trench coat stands in a dark alleyway, lighting a cigarette. The only light source is the flame of the lighter, illuminating his rugged face and the rain falling around him. The shadows are deep blacks (ink style). Speech bubble in the corner says 'It was a long night.' The lines are bold and expressive, cross-hatching shading, monochromatic with a splash of red for the lighter flame."

image = pipe.generate(
    prompt=prompt,
    height=1024,
    width=1024,
    num_sampling_steps=50, # adjust to 25 steps for faster inference
    guidance_scale=7.5,
    num_images=1,
    seed=42
)[0]

image.save("example.png")