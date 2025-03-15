# Wan Image-to-Video with LoRA

This is a [Cog](https://github.com/replicate/cog) implementation of the [Wan Image-to-Video 2.1 model](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) that supports custom LoRA weights. This model enables you to animate static images into short videos with various motion effects defined by text prompts and enhanced through custom LoRA weights.

## Features

- Transform any image into a fluid, high-quality video
- Apply custom LoRA weights for specialized motion effects
- Control video duration, frame rate, and generation parameters
- Multiple image resizing strategies to handle different aspect ratios

## Examples

Here's what you can create with this model:

- Apply a "squish" effect to static objects
- Create flowing water or waves from still images
- Generate dancing or movement in character images
- Add ripple effects to surfaces
- Convert landscape images into slow panning scenes

## Usage

### API Interface

```bash
# Example API call
curl -X POST https://api.replicate.com/v1/predictions \
  -H "Authorization: Token YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "YOUR_MODEL_VERSION",
    "input": {
      "image": "https://example.com/your-image.jpg",
      "prompt": "In the video, the object gently moves up and down",
      "lora_url": "https://huggingface.co/Remade/Squish/resolve/main/squish_lora.safetensors",
      "lora_strength": 1.0,
      "duration": 3.0,
      "resize_mode": "keep_aspect_ratio"
    }
  }'
```

### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `image` | Input image to animate | (required) | - |
| `prompt` | Text description of the desired motion/effect | (required) | - |
| `negative_prompt` | Description of qualities to avoid | "low quality, bad quality, blurry, pixelated, watermark" | - |
| `lora_url` | URL to LoRA weights (HuggingFace or CivitAI) | (required) | - |
| `lora_strength` | LoRA effect intensity | 1.0 | 0.0 - 2.0 |
| `duration` | Video duration in seconds | 3.0 | 1.0 - 5.0 |
| `fps` | Frames per second in output video | 16 | 7 - 30 |
| `guidance_scale` | CFG scale for generation strength | 5.0 | 1.0 - 20.0 |
| `num_inference_steps` | Number of denoising steps | 28 | 1 - 100 |
| `resize_mode` | Image resizing strategy | "auto" | "auto", "fixed_square", "keep_aspect_ratio" |
| `seed` | Random seed for reproducibility | None | - |

### Resize Modes

- **auto**: Automatically selects between square (for nearly square images) and aspect ratio preserving 
- **fixed_square**: Resizes to 512×512, suitable for square-centric effects
- **keep_aspect_ratio**: Preserves the original aspect ratio while optimizing for model performance

## Technical Details

This implementation uses the Wan 2.1 Image-to-Video model with 14B parameters, optimized for 480p output. The model:

- Takes a source image as input
- Encodes it using a CLIP vision encoder
- Generates a sequence of video frames based on the image and prompt
- Uses LoRA fine-tuning weights to control the motion style
- Renders a video at specified frames per second

### Frame Count

The model works best with frame counts that follow the 4K+1 pattern. The implementation automatically calculates the appropriate frame count based on your requested duration, capped at 81 frames maximum.

### System Requirements

- NVIDIA GPU with at least 16GB VRAM recommended
- CUDA 12.4 compatible system

## Known LoRA Resources

Some publicly available LoRA weights for animation effects:

- [Squish LoRA](https://huggingface.co/Remade/Squish)
- [Water Flow LoRA](https://huggingface.co/Remade/FlowingWater)
- [Wind Blowing LoRA](https://huggingface.co/distralfi/WindBlowing_wan21)

## Troubleshooting

- If you encounter CUDA memory issues, try using a smaller image or the "fixed_square" resize mode
- For best results, use clear images with good lighting and minimal background clutter
- Prompts should be detailed and specifically describe the motion you want to see

## License

This implementation uses the Wan 2.1 model, which is subject to its own license terms. Please refer to the [Wan-AI license](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) for more details.

## Acknowledgments

- [Wan-AI](https://huggingface.co/Wan-AI) for creating the Wan image-to-video models
- The diffusers library for the pipeline implementation
- The creators of various LoRA weights that enhance the animation capabilities 