# 录制视频交互流程（中文速记）

视频目标时长 11 分 30 秒，硬上限 15 分钟，对应英文脚本 `reports/video_script_en.md`。下面给出中文录制人员的关键 cut 点与对照说明。

## 镜头清单

- **Tab A**：阿里云演示站 `http://47.237.107.46:18080`，已重新部署最新版（含“LLM and RAG”面板）。
- **Tab B**：GitHub 仓库 `InftyMing/NTU-AAI-CA6125-CourseWork-Group16`。
- **Tab C**：Kaggle 比赛页 `feedback-prize-english-language-learning`。
- **Tab D**：VS Code，打开 `LLM experiments/local training_essay_scoring_m_chip.ipynb`、`LLM experiments/kaggle inference_group-16.ipynb`、`RAG experiments/kaggle inference_rag_group-16.ipynb` 与 `src/feedback_ell/enhanced.py`。
- **Tab E**：`Group16_report.pdf`。

## 时间轴对照

| 段落 | 时长 | 屏幕 | 中文要点 |
| :--- | :--- | :--- | :--- |
| 0:00 – 0:30 标题卡 | 30s | 静态 | 自我介绍、六维评分、官方指标 MCRMSE 越低越好。 |
| 0:30 – 1:00 比赛页 | 30s | Tab C | Code competition、public test 仅 3 篇、连续分 1.0–5.0、按 0.5 步长。 |
| 1:00 – 1:40 仓库结构 | 40s | Tab D | `src/feedback_ell/`、`scripts/`、`web/`、`experiments/artifacts/` 一一指给观众。 |
| 1:40 – 2:30 Demo 总览 | 50s | Tab A | KPI：CV MCRMSE 0.51873；3911 篇训练数据；六个目标。 |
| 2:30 – 3:30 数据审计 | 60s | Tab A `#data` | 均值条形图、相关性热力图、长度箱线图。 |
| 3:30 – 5:00 模型与 leaderboard | 90s | Tab A `#models` | 八个模型从差到好的递进、点击 stacked_ensemble 行触发 per-target 图。 |
| **5:00 – 6:30 LLM 与 RAG 探针（新增）** | 90s | Tab A `#llm-rag` + Tab D | 左卡 Qwen2.5-1.5B + LoRA：r=8、α=16、3 epoch、训练 loss 0.184→0.165、val_loss=NaN 的 fp16/MPS 现象；右卡 DeBERTa-v3-base + KNN：冻结编码器、k=20、softmax 温度 10、500 篇 hold-out 上 0.5445 MCRMSE。补一句：传统堆叠仍在 5 折 CV 下领先，因此作为头条提交。 |
| 6:30 – 7:30 误差分析 | 60s | Tab A `#errors` | 长度桶最长 vs 最短差 6%；分数桶低/中/高对应 0.62/0.46/0.59；per-target 表显示残差均值≈0。 |
| 7:30 – 9:00 试用作文打分 | 90s | Tab A `#demo` | 默认样例点击 Predict；粘贴明显较弱的样例对比；强调表层特征预测器仅供演示。 |
| 9:00 – 9:30 团队与提交清单 | 30s | Tab A `#team` + `#submission` | 五位成员分工；最终模型卡 + CSV 列表。 |
| 9:30 – 10:00 报告 PDF | 30s | Tab E | 抽 §7、§7.7、§7.8、§8 几页带过；强调表与 demo 数据一致。 |
| 10:00 – 10:30 GitHub 与 Notebook | 30s | Tab B | `notebooks/Group16_inference.ipynb`（堆叠模型推理）、`LLM experiments/`、`RAG experiments/`；说明 `.gitignore` 已排除敏感文件。 |
| 10:30 – 11:00 收获总结 | 30s | Tab A 顶部 | 三点：传统特征仍有用；堆叠需要异质成分；正面回应 LLM/RAG 主题增强了项目说服力。 |
| 11:00 – 11:30 致谢与片尾 | 30s | 静态 | 列出五位成员；致谢公开方案；不公开源码声明。 |

## 录制前 checklist（中文）

1. 重新加载阿里云演示站，确认 KPI 显示 0.51873 且“LLM and RAG”面板加载成功。
2. 在 VS Code 中预先打开三个 Notebook，避免录制中现场翻找。
3. `Group16_report.pdf` 已停留在第 1 页；不要在录制中现场用搜索框。
4. 提前在 OBS 配好 5 个场景，使用 cut 转场。
5. 录完后导出 1080p；上传 YouTube 设为 **Unlisted**；把链接写入 `Group16_video.txt`。
