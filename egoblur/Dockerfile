# EgoBlur (Gen2) — CPU-only image for blurring faces / license plates
# in local images and videos.
#
# Models are NOT baked in: they sit behind a license click on the Project
# Aria site. Download them yourself and bind-mount them at runtime.
#   https://www.projectaria.com/tools/egoblur
#   (you want ego_blur_face_gen2.jit and ego_blur_lp_gen2.jit)

FROM python:3.12-slim

# --- System libraries -------------------------------------------------------
# libgl1 / libglib2.0-0 : required by OpenCV (cv2), which EgoBlur uses
# libheif-examples      : provides `heif-convert` for iPhone HEIC -> PNG/JPEG
#                         (OpenCV can't read HEIC, so convert first)
# git                   : EgoBlur ships its demo assets via the repo
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libheif-examples \
        git \
    && rm -rf /var/lib/apt/lists/*

# --- Python deps ------------------------------------------------------------
# Install the CPU-only build of torch FIRST, so that installing `egoblur`
# doesn't drag in the multi-GB CUDA wheels from PyPI. If pip later reports a
# torch version conflict, pin the exact versions egoblur asks for right here.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision && \
    pip install --no-cache-dir egoblur

# --- EgoBlur repo (for the demo assets + to confirm the CLI is on PATH) -----
RUN git clone --depth 1 https://github.com/facebookresearch/EgoBlur.git /opt/EgoBlur
ENV EGOBLUR_REPO=/opt/EgoBlur

# --- Runtime layout ---------------------------------------------------------
# /models -> bind-mount your downloaded *.jit model files here
# /data   -> bind-mount your photos here (serves as both input and output)
RUN mkdir -p /models /data
ENV EGOBLUR_MODELS=/models
WORKDIR /data

# Print the CLI help by default; override with your own command at `docker run`.
CMD ["egoblur-gen2", "--help"]
