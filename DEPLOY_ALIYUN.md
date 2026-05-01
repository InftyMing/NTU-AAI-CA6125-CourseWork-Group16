# Aliyun Deployment Guide

## Option A: Deploy on ECS with Docker

1. Install Docker and Docker Compose on the ECS instance.
2. Open security group inbound port `8000`, or map the service behind Nginx on port `80/443`.
3. Upload or pull the project code on the ECS instance.
4. Build and run:

```bash
docker compose up --build -d
```

5. Visit:

```text
http://<ECS_PUBLIC_IP>:8000
```

## Option B: Push Image to Aliyun Container Registry

```bash
docker build -t feedback-ell-demo:latest .
docker tag feedback-ell-demo:latest registry.cn-<region>.aliyuncs.com/<namespace>/feedback-ell-demo:latest
docker login registry.cn-<region>.aliyuncs.com
docker push registry.cn-<region>.aliyuncs.com/<namespace>/feedback-ell-demo:latest
```

On ECS:

```bash
docker pull registry.cn-<region>.aliyuncs.com/<namespace>/feedback-ell-demo:latest
docker run -d --name feedback-ell-demo -p 8000:8000 registry.cn-<region>.aliyuncs.com/<namespace>/feedback-ell-demo:latest
```

## Production Notes

- The web container is for presentation and inference display, not for full training.
- Train models locally or on a GPU machine first, then copy `experiments/artifacts` and lightweight model outputs into the image or mount them as read-only volumes.
- Keep `kaggle.json`, `.env`, and private data out of the image.
- If using Nginx, reverse proxy `/` to `http://127.0.0.1:8000`.

## Video Demo Flow

1. Open the web home page.
2. Show project overview and six target dimensions.
3. Open Data Audit and explain sample counts and score distributions.
4. Open Experiments and compare baseline with Transformer results.
5. Use Essay Scoring Demo with one example essay.
6. Show Submission section and generated file path.
7. Conclude with reproducibility and limitations.
