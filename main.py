from predict import Predictor

predictor = Predictor()
predictor.setup()

predictor.predict(
    image="https://saas-blogs-bucket.s3.us-east-1.amazonaws.com/d822c90b-52e4-4202-998f-f8e433d2335c.webp",
    prompt="The video begins with a photo of a person. Then the 8r1d3 bride effect occurs. The person is now in a white wedding dress, holding a bouquet.",
    negative_prompt="low quality, bad quality, blurry, pixelated, watermark",
    lora_url="https://huggingface.co/berkelmas/wan-video-loras/resolve/main/bride_it_wan_lora.safetensors",
    lora_strength=1,
    duration=3,
    fps=16,
    num_inference_steps=40,
    guidance_scale=5,
    seed=None,
    resize_mode="auto",
)