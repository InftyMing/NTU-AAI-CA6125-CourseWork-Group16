"""Group metadata used by reports and the demo page."""

GROUP_CODE = "Group16"

TEAM_MEMBERS = [
    {
        "name": "DHAYALA MURTHY SAVITHACCDS",
        "role": "Literature review, problem framing, and video/PDF production",
        "role_zh": "文献调研、问题定义与视频/PDF 制作",
        "contribution": (
            "Surveyed Feedback Prize public solutions and AES research, gathered the "
            "Qwen/LoRA/DeBERTa-v3 references used in the report, and co-produced the "
            "presentation video and the final report PDF rendering."
        ),
        "contribution_zh": (
            "调研 Feedback Prize 公开方案与 AES 相关研究，整理报告中引用的 Qwen / LoRA / "
            "DeBERTa-v3 等参考文献，并联合负责展示视频与报告 PDF 的制作。"
        ),
    },
    {
        "name": "HUANG ZIXUANCCDS",
        "role": "Data audit, feature engineering, and RAG retrieval (§7.8)",
        "role_zh": "数据审计、特征工程与 RAG 检索（§7.8）",
        "contribution": (
            "Built the data audit module, the ten text statistics, and the TF-IDF + SVD "
            "feature pipelines; implemented the training-free DeBERTa-v3-base + cosine-KNN "
            "RAG scorer reported in §7.8."
        ),
        "contribution_zh": (
            "实现数据审计模块、十个文本统计量与 TF-IDF + SVD 特征流水线；负责 §7.8 中免训练的 "
            "DeBERTa-v3-base + 余弦 KNN 检索式打分器。"
        ),
    },
    {
        "name": "SUN MINGCCDS",
        "role": "Modeling and evaluation",
        "role_zh": "建模与评估",
        "contribution": (
            "Implemented MCRMSE, the cross-validation harness, the per-target Ridge / "
            "fused Ridge / LightGBM-SVD components, and the per-target convex stack that "
            "reaches 0.51873 CV MCRMSE."
        ),
        "contribution_zh": (
            "实现 MCRMSE、交叉验证框架、per-target Ridge / 融合 Ridge / LightGBM-SVD 三个成分模型，"
            "以及最终 CV MCRMSE 为 0.51873 的按目标列凸组合堆叠器。"
        ),
    },
    {
        "name": "TAN XUAN ZHAOCCDS",
        "role": "Web demo, deployment, and video/PDF production",
        "role_zh": "前端演示、部署与视频/PDF 制作",
        "contribution": (
            "Built the FastAPI service, the SVG-based front end (including the LLM and RAG "
            "panel), the Docker image, and the Aliyun deployment; co-produced the final "
            "video walkthrough and PDF render of the report."
        ),
        "contribution_zh": (
            "构建 FastAPI 服务、SVG 前端（含“LLM and RAG”面板）、Docker 镜像与阿里云部署；"
            "联合负责最终视频录制与报告 PDF 排版。"
        ),
    },
    {
        "name": "WANG QIANCCDS",
        "role": "Report, coordination, and LLM fine-tune (§7.7)",
        "role_zh": "报告、项目协调与 LLM 微调（§7.7）",
        "contribution": (
            "Coordinated the group, integrated experiment results into the bilingual "
            "report, and ran the Qwen2.5-1.5B-Instruct + LoRA fine-tune reported in §7.7, "
            "including the JSON post-processor that snaps outputs to the official grid."
        ),
        "contribution_zh": (
            "统筹小组工作，将实验结果整合为中英文报告，并完成 §7.7 中 Qwen2.5-1.5B-Instruct + LoRA "
            "微调的训练、推理与结构化分数 JSON 的后处理。"
        ),
    },
]
