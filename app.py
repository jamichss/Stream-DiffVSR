import os
import shutil
import subprocess
import tempfile
import time
import uuid
import logging
from pathlib import Path

import gradio as gr

try:
    import gradio_client.utils as _gr_client_utils

    _orig_get_type = _gr_client_utils.get_type
    _orig_json_schema = _gr_client_utils._json_schema_to_python_type
    _orig_json_schema_to = _gr_client_utils.json_schema_to_python_type

    def _get_type_patched(schema):
        if isinstance(schema, bool):
            return {}
        return _orig_get_type(schema)

    def _json_schema_patched(schema, defs):
        if isinstance(schema, bool):
            return "Any"
        return _orig_json_schema(schema, defs)

    _gr_client_utils.get_type = _get_type_patched
    _gr_client_utils._json_schema_to_python_type = _json_schema_patched
    def _json_schema_to_python_type_patched(schema):
        if isinstance(schema, bool):
            return "Any"
        return _orig_json_schema_to(schema)

    _gr_client_utils.json_schema_to_python_type = _json_schema_to_python_type_patched
except Exception:
    pass

import torch
from accelerate.utils import set_seed
from PIL import Image
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from diffusers import DDIMScheduler
from pipeline.stream_diffvsr_pipeline import (
    StreamDiffVSRPipeline,
    ControlNetModel,
    UNet2DConditionModel,
)
from temporal_autoencoder.autoencoder_tiny import TemporalAutoencoderTiny

torch.backends.cuda.matmul.allow_tf32 = True

MODEL_ID = os.getenv("STREAM_DIFFVSR_MODEL_ID", "Jamichsu/Stream-DiffVSR")
TMP_ROOT = Path(os.getenv("STREAM_DIFFVSR_TMP_DIR", ".gradio_tmp"))
OUTPUT_ROOT = Path(os.getenv("STREAM_DIFFVSR_OUTPUT_DIR", "outputs"))
MAX_FRAMES_PER_SEQ = int(os.getenv("STREAM_DIFFVSR_MAX_FRAMES_PER_SEQ", "30"))

_PIPELINE = None
_OF_MODEL = None
_DEVICE = None

logger = logging.getLogger("stream_diffvsr_ui")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _run_cmd(cmd):
    logger.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{stderr}")


def _get_video_fps(video_path):
    logger.info("Reading FPS for %s", video_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    rate = result.stdout.strip()
    if not rate:
        raise RuntimeError("Could not determine input video FPS.")
    if "/" in rate:
        num, den = rate.split("/", 1)
        fps_value = float(num) / float(den)
    else:
        fps_value = float(rate)
    return rate, fps_value


def _has_audio(video_path):
    logger.info("Checking for audio stream in %s", video_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip() == "audio"


def _extract_frames(video_path, frames_dir):
    logger.info("Extracting frames from %s into %s", video_path, frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-vsync",
        "0",
        str(frames_dir / "frame_%06d.png"),
    ]
    _run_cmd(cmd)


def _split_frames_into_sequences(frames_dir, seq_root, max_frames):
    seq_root.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError("No frames were extracted from the input video.")
    logger.info("Splitting %d frames into sequences of %d", len(frame_paths), max_frames)
    seq_dirs = []
    seq_index = 1
    current_dir = seq_root / f"seq_{seq_index:04d}"
    current_dir.mkdir(parents=True, exist_ok=True)
    seq_dirs.append(current_dir)
    for idx, frame_path in enumerate(frame_paths, start=1):
        if idx > 1 and (idx - 1) % max_frames == 0:
            seq_index += 1
            current_dir = seq_root / f"seq_{seq_index:04d}"
            current_dir.mkdir(parents=True, exist_ok=True)
            seq_dirs.append(current_dir)
        shutil.move(str(frame_path), current_dir / frame_path.name)
    return seq_dirs


def _prepare_preview_video(video_path):
    logger.info("Preparing H.264 preview for %s", video_path)
    preview_dir = TMP_ROOT / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"preview_{uuid.uuid4().hex[:10]}.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(preview_path),
    ]
    _run_cmd(cmd)
    return str(preview_path)


def _is_preview_path(video_path):
    try:
        path = Path(video_path).resolve()
        preview_root = (TMP_ROOT / "previews").resolve()
        return path == preview_root or path.is_relative_to(preview_root)
    except Exception:
        return False


def _extract_audio(video_path, audio_path):
    logger.info("Extracting audio from %s into %s", video_path, audio_path)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(audio_path),
    ]
    _run_cmd(cmd)


def _assemble_video(frames_dir, output_path, fps_rate):
    logger.info("Encoding video from %s into %s", frames_dir, output_path)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        fps_rate,
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run_cmd(cmd)


def _mux_audio(video_path, audio_path, output_path):
    logger.info("Muxing audio %s into %s", audio_path, output_path)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run_cmd(cmd)


def _load_component(cls, weight_path, model_id, subfolder):
    path = weight_path if weight_path else model_id
    sub = None if weight_path else subfolder
    return cls.from_pretrained(path, subfolder=sub)


def _resolve_video_path(video_input):
    if not video_input:
        return None
    if isinstance(video_input, str):
        return video_input
    if isinstance(video_input, dict):
        for key in ("path", "name", "data"):
            value = video_input.get(key)
            if isinstance(value, str) and value:
                return value
    for attr in ("path", "name"):
        value = getattr(video_input, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def _ensure_models_loaded(progress=None):
    global _PIPELINE, _OF_MODEL, _DEVICE
    if _PIPELINE is not None:
        return

    if progress is not None:
        progress(0.02, desc="Loading models")

    logger.info("Loading models on device: %s", "cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    controlnet = _load_component(ControlNetModel, None, MODEL_ID, "controlnet")
    unet = _load_component(UNet2DConditionModel, None, MODEL_ID, "unet")
    vae = _load_component(TemporalAutoencoderTiny, None, MODEL_ID, "vae")
    scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    _PIPELINE = StreamDiffVSRPipeline.from_pretrained(
        MODEL_ID,
        controlnet=controlnet,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
    )

    _PIPELINE = _PIPELINE.to(_DEVICE)
    try:
        _PIPELINE.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    _OF_MODEL = raft_large(weights=Raft_Large_Weights.DEFAULT).to(_DEVICE).eval()
    _OF_MODEL.requires_grad_(False)


def _load_frames(frames_dir):
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}.")
    logger.info("Loading %d frames from %s", len(frame_paths), frames_dir)
    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as img:
            frames.append(img.convert("RGB"))
    return frame_paths, frames


def _upscale_frames(frames, num_inference_steps):
    logger.info("Upscaling %d frames with %d inference steps", len(frames), num_inference_steps)
    with torch.inference_mode():
        output = _PIPELINE(
            "",
            frames,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=0,
            of_model=_OF_MODEL,
        )
    return output.images


def upscale_video(video_path, num_inference_steps, original_video_path=None, progress=gr.Progress()):
    resolved_original = _resolve_video_path(original_video_path)
    resolved_preview = _resolve_video_path(video_path)
    video_path = resolved_original or resolved_preview
    if not video_path:
        raise gr.Error("Please upload a source video first.")

    logger.info("Starting upscale for %s", video_path)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    _ensure_models_loaded(progress=progress)

    fps_rate, fps_value = _get_video_fps(video_path)
    logger.info("Source FPS: %s (%.2f)", fps_rate, fps_value)
    output_name = f"upscaled_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
    output_path = OUTPUT_ROOT / output_name

    with tempfile.TemporaryDirectory(dir=TMP_ROOT) as work_dir:
        work_dir = Path(work_dir)
        input_frames_dir = work_dir / "input_frames"
        output_frames_dir = work_dir / "output_frames"

        progress(0.08, desc="Extracting frames")
        _extract_frames(video_path, input_frames_dir)

        progress(0.2, desc="Splitting into sequences")
        seq_root = work_dir / "input_seqs"
        seq_dirs = _split_frames_into_sequences(
            input_frames_dir,
            seq_root,
            MAX_FRAMES_PER_SEQ,
        )
        logger.info("Created %d sequence folders in %s", len(seq_dirs), seq_root)

        output_frames_dir.mkdir(parents=True, exist_ok=True)
        total_seqs = len(seq_dirs)
        for seq_index, seq_dir in enumerate(seq_dirs, start=1):
            logger.info("Processing sequence %d/%d (%s)", seq_index, total_seqs, seq_dir.name)
            progress(
                0.25 + 0.5 * (seq_index - 1) / max(total_seqs, 1),
                desc=f"Upscaling sequence {seq_index}/{total_seqs}",
            )
            frame_paths, frames = _load_frames(seq_dir)
            frames_hr = _upscale_frames(frames, num_inference_steps)
            del frames
            for frame, frame_path in zip(frames_hr, frame_paths):
                image = frame[0] if isinstance(frame, (list, tuple)) else frame
                image.save(output_frames_dir / frame_path.name)

            del frames_hr
            if _DEVICE and _DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        progress(0.82, desc="Encoding video")
        video_no_audio = work_dir / "upscaled_no_audio.mp4"
        _assemble_video(output_frames_dir, video_no_audio, fps_rate)

        progress(0.92, desc="Muxing audio")
        if _has_audio(video_path):
            audio_path = work_dir / "audio.m4a"
            _extract_audio(video_path, audio_path)
            _mux_audio(video_no_audio, audio_path, output_path)
        else:
            logger.info("No audio stream found; skipping mux.")
            shutil.move(video_no_audio, output_path)

    if _DEVICE and _DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    logger.info("Upscale complete: %s", output_path)
    status = f"Done. Output: {output_path.name} ({fps_value:.2f} fps)"
    return str(output_path), str(output_path), status


def prepare_source_video(video_input, original_video_state, preview_video_state, ignore_next_change):
    video_path = _resolve_video_path(video_input)
    original_path = _resolve_video_path(original_video_state)
    preview_path_state = _resolve_video_path(preview_video_state)
    if ignore_next_change:
        return gr.update(), original_path, preview_path_state, False
    if not video_path:
        return None, None, None, False
    try:
        preview_path = _prepare_preview_video(video_path)
    except Exception:
        preview_path = video_path
    return preview_path, video_path, preview_path, True


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&display=swap');

:root {
  --bg-1: #f4f1ea;
  --bg-2: #e6dfd3;
  --accent: #2f6b5f;
  --accent-2: #c87941;
  --ink: #1e2b2b;
}

.gradio-container {
  font-family: "Space Grotesk", "Helvetica Neue", Arial, sans-serif;
  background: linear-gradient(135deg, var(--bg-1), var(--bg-2));
}

#title {
  font-size: 2.1rem;
  font-weight: 600;
  color: var(--ink);
  margin-bottom: 0.25rem;
}

.panel {
  border: 1px solid rgba(30, 43, 43, 0.15);
  border-radius: 14px;
  padding: 16px;
  background: rgba(255, 255, 255, 0.7);
  backdrop-filter: blur(6px);
}

#run-button {
  background: var(--accent);
  color: #fff;
  border: none;
}

#run-button:hover {
  background: #26594f;
}
"""


def build_ui():
    with gr.Blocks(css=CSS, theme=gr.themes.Soft(primary_hue="orange", secondary_hue="green")) as demo:
        gr.Markdown("Stream-DiffVSR Video Upscaler", elem_id="title")
        gr.Markdown(
            "Upload a video, choose the number of inference steps, and run the upscaler. "
            "The output keeps the original FPS and audio. CUDA GPU recommended.",
        )

        with gr.Row():
            with gr.Column():
                with gr.Group(elem_classes="panel"):
                    input_video = gr.Video(
                        label="Source Video",
                        sources=["upload"],
                        format=None,
                    )
                    original_video_path = gr.State()
                    preview_video_path = gr.State()
                    ignore_next_change = gr.State(False)
                    steps = gr.Slider(
                        minimum=1,
                        maximum=8,
                        step=1,
                        value=4,
                        label="Inference Steps",
                    )
                    run_button = gr.Button("Start Upscale", elem_id="run-button", variant="primary")
                    status = gr.Markdown("")

            with gr.Column():
                with gr.Group(elem_classes="panel"):
                    output_video = gr.Video(label="Upscaled Video", format="mp4")
                    download = gr.DownloadButton(
                        label="Download Upscaled Video",
                    )

        run_button.click(
            fn=upscale_video,
            inputs=[input_video, steps, original_video_path],
            outputs=[output_video, download, status],
        )
        input_video.change(
            fn=prepare_source_video,
            inputs=[input_video, original_video_path, preview_video_path, ignore_next_change],
            outputs=[input_video, original_video_path, preview_video_path, ignore_next_change],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "8000")),
        show_api=False,
    )
