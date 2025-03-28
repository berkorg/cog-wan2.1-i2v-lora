import os
import time
import torch
import tempfile
import requests
import subprocess
import numpy as np
import imageio
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from huggingface_hub import hf_hub_download

MODEL_CACHE = "workspace/checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/wan2.1/model_cache/Wan2.1-I2V-14B-480P-Diffusers.tar"

# Frame rates for the Wan model
MODEL_FRAME_RATE = 16  # WanVideo models use 16 fps

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

def calculate_frames(duration, frame_rate):
    """Calculate frames ensuring they follow the 4K+1 pattern"""
    raw_frames = round(duration * frame_rate)
    # Adjust to nearest 4K+1 value
    nearest_multiple_of_4 = round(raw_frames / 4) * 4
    # Then add 1 to get 4K+1
    return min(nearest_multiple_of_4 + 1, 81)  # Cap at 81 frames max

class Predictor():
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # download weights if they don't exist
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        MODEL = MODEL_CACHE + "/Wan2.1-I2V-14B-480P-Diffusers"
        print(f"Loading model from {MODEL}")
        try:
            # Load components with appropriate dtype
            self.image_encoder = CLIPVisionModel.from_pretrained(
                MODEL, 
                subfolder="image_encoder", 
                torch_dtype=torch.float32
            )
            self.vae = AutoencoderKLWan.from_pretrained(
                MODEL, 
                subfolder="vae", 
                torch_dtype=torch.float32
            )
            # Initialize pipeline with optimal memory settings and model CPU offload sequence
            pipe = WanImageToVideoPipeline.from_pretrained(
                MODEL,
                vae=self.vae,
                image_encoder=self.image_encoder,
                torch_dtype=torch.bfloat16
            )
            
            # Move model to GPU and enable CPU offloading for optimal memory usage
            self.pipe = pipe.to("cuda")
            # self.pipe.enable_model_cpu_offload()
            
            # Store parameters for VAE scale factors
            self.vae_scale_factor_temporal = self.pipe.vae_scale_factor_temporal
            self.vae_scale_factor_spatial = self.pipe.vae_scale_factor_spatial
            print(f"Model loaded successfully. VAE scale factors: temporal={self.vae_scale_factor_temporal}, spatial={self.vae_scale_factor_spatial}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def download_lora(self, lora_url):
        """Download LoRA weights from URL and return local path"""
        print(f"Downloading LoRA from: {lora_url}")
        try:
            if lora_url.startswith("https://huggingface.co"):
                # Extract repo_id and filename from HF URL
                parts = lora_url.split("/")
                repo_id = "/".join(parts[3:5])
                filename = parts[-1]
                print(f"Downloading from HuggingFace: repo_id={repo_id}, filename={filename}")
                
                # Download from HuggingFace
                return hf_hub_download(repo_id=repo_id, filename=filename)
            else:
                # Handle direct URLs (e.g., CivitAI)
                print(f"Downloading from direct URL: {lora_url}")
                response = requests.get(lora_url)
                if response.status_code == 200:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp:
                        tmp.write(response.content)
                        print(f"LoRA downloaded successfully to: {tmp.name}")
                        return tmp.name
                else:
                    raise ValueError(f"Failed to download LoRA from {lora_url}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error downloading LoRA: {str(e)}")
            raise RuntimeError(f"Failed to download LoRA: {str(e)}")

    def predict(
        self,
        image: str,
        prompt: str,
        negative_prompt: str = "low quality, bad quality, blurry, pixelated, watermark",
        lora_url: str = None,
        lora_strength: float = 1.0,
        duration: float = 3.0,
        fps: int = 16,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 28,
        resize_mode: str = "auto",
        seed: int = None
    ) -> str:
        """Run image-to-video inference with custom LoRA"""
        
        print(f"Starting prediction with: prompt='{prompt}', lora_strength={lora_strength}, duration={duration}s")
        # Calculate number of frames based on duration
        num_frames = calculate_frames(duration, MODEL_FRAME_RATE)
        print(f"Calculated {num_frames} frames for {duration}s at {MODEL_FRAME_RATE} fps")
        
        # Set random seed if provided
        if seed is not None:
            print(f"Setting random seed: {seed}")
            torch.manual_seed(seed)
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = None
            
        # Load and preprocess input image
        try:
            print(f"Loading input image from: {image}")
            input_image = load_image(str(image))
            print(f"Image loaded successfully: {input_image.width}x{input_image.height}")
        except Exception as e:
            raise RuntimeError(f"Failed to load input image: {str(e)}")
        
        # Calculate dimensions based on resize mode
        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        
        if resize_mode == "fixed_square":
            # ComfyUI workflow style - resize to 512x512 (standard for 480p model)
            width = height = 512
            print(f"Using fixed square dimensions: {width}x{height} (ComfyUI style)")
        else:
            # Calculate dimensions based on maximum area while preserving aspect ratio
            if resize_mode == "auto":
                # Determine if we should use fixed square based on aspect ratio
                aspect_ratio = input_image.height / input_image.width
                if 0.9 <= aspect_ratio <= 1.1:
                    # For nearly square images, use fixed square dimensions
                    width = height = 512
                    print(f"Auto-selected fixed square dimensions: {width}x{height} for aspect ratio {aspect_ratio:.2f}")
                else:
                    # For non-square images, preserve aspect ratio
                    resize_mode = "keep_aspect_ratio"
            
            if resize_mode == "keep_aspect_ratio":
                # Original dynamic resizing that preserves aspect ratio
                max_area = 480 * 832  # Match the example code (480x832 pixels)
                aspect_ratio = input_image.height / input_image.width
                
                height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
                print(f"Using aspect-preserving dimensions: {width}x{height} (aspect ratio: {aspect_ratio:.2f})")
        
        # Validate dimensions
        if height % 16 != 0 or width % 16 != 0:
            print(f"Warning: Adjusting dimensions to be divisible by 16. Original: {height}x{width}")
            height = (height // 16) * 16
            width = (width // 16) * 16
            
        print(f"Final dimensions: {width}x{height}")
        # Resize image to calculated dimensions
        input_image = input_image.resize((width, height))
        # Download and load LoRA weights with specified strength
        lora_path = None
        try:
            lora_path = self.download_lora(lora_url)
            print(f"Loading LoRA weights from: {lora_path} with strength: {lora_strength}")
            self.pipe.load_lora_weights(lora_path, multiplier=lora_strength)
            print("LoRA weights loaded successfully")
        except Exception as e:
            if lora_path and os.path.exists(lora_path):
                os.remove(lora_path)
            raise RuntimeError(f"Failed to load LoRA weights: {str(e)}")
        
        # Generate video frames with improved error handling
        try:
            print(f"Starting video generation: {num_frames} frames, {num_inference_steps} steps, guidance={guidance_scale}")
            start_time = time.time()
            
            output = self.pipe(
                image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).frames[0]
            
            generation_time = time.time() - start_time
            print(f"Video generation completed in {generation_time:.2f} seconds")
            
            # Save video
            output_path = tempfile.mkdtemp() + "/output.mp4"
            print(f"Exporting video to: {output_path} at {fps} fps")
            export_to_video(output, str(output_path), fps=fps)
            print(f"Video saved successfully to: {output_path}")
            
            # Cleanup
            if os.path.exists(lora_path):
                os.remove(lora_path)
                print(f"Cleaned up temporary LoRA file: {lora_path}")
                
            return output_path
            
        except Exception as e:
            # Clean up LoRA file if error occurs
            if lora_path and os.path.exists(lora_path):
                os.remove(lora_path)
            print(f"Error during video generation: {str(e)}")
            raise RuntimeError(f"Video generation failed: {str(e)}")
