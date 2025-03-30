from predict import Predictor

predictor = Predictor()
predictor.setup()

predictor.predict(
    image="https://saas-blogs-bucket.s3.us-east-1.amazonaws.com/wan_input_images/cdb9c61d-a29a-4482-9ba1-6e0f1ae1dedf.webp",
    prompt="The video begins with a photo of a person. then t2k1s takes off clothes revealing a lean muscular body and shows off muscles, looking towards the camera",
    negative_prompt="low quality, bad quality, blurry, pixelated, watermark",
    lora_url="https://huggingface.co/berkelmas/wan-video-loras/resolve/main/muscle_show_off_wan_lora.safetensors",
    lora_strength=1,
    duration=3,
    fps=16,
    num_inference_steps=40,
    guidance_scale=5,
    seed=None,
    resize_mode="auto",
)
