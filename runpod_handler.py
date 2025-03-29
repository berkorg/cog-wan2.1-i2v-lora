import runpod
import os
from dotenv import load_dotenv

from predict import Predictor

# Load environment variables from .env file
load_dotenv()

def handler(job):
    """Handler function that will be used to process jobs."""
    job_input = job["input"]

    name = job_input.get("name", "World")
    job_type = job_input.get("job_type", None)

    if job_type == None:
        return {"error": "You need to specify job_type"}

    # generate pre-signed URL
    if job_type == "create_image_to_video":
        image_url = job_input.get("image_url", None)
        prompt = job_input.get("prompt", None)
        negative_prompt = job_input.get("negative_prompt", "low quality, bad quality, blurry, pixelated, watermark")
        lora_url = job_input.get("lora_url", None)
        lora_strength = job_input.get("lora_strength", 1)
        duration = job_input.get("duration", 3)
        fps = job_input.get("fps", 16)
        num_inference_steps = job_input.get("num_inference_steps", 40)
        guidance_scale = job_input.get("guidance_scale", 5)
        seed = job_input.get("seed", None)
        resize_mode = job_input.get("resize_mode", "auto")
        
        predictor = Predictor()
        predictor.setup()

        public_video_url = predictor.predict(
            image=image_url,
            prompt=prompt,
            negative_prompt=negative_prompt,
            lora_url=lora_url,
            lora_strength=lora_strength,
            duration=duration,
            fps=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            resize_mode=resize_mode,
        )
        return {"public_video_url": public_video_url}

    elif job_type == "test_job":
        return {"status": "handler is fine!"}
    else:
        return {
            "error": "job_type should be one of 'create_image_to_video' "
        }


runpod.serverless.start({"handler": handler})
