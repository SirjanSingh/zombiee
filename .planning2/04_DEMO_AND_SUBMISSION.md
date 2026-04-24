# Demo and Submission Requirements

## Minimum submission requirements (non-negotiable from organizers)

1. **OpenEnv (latest release)** used with `Environment` / `MCPEnvironment` base classes.
2. **Training script** using Unsloth or HF TRL, Colab-runnable, committed as `training/notebook.ipynb`.
3. **Real training evidence** — loss + reward plots committed as PNGs in `plots/`.
4. **Mini-blog OR <2-min YouTube video** — linked from README.
5. **Hugging Face Space** hosting the env, URL in README.
6. **README** that motivates the problem, explains the env, shows results, links everything.

## README structure

```markdown
---
sdk: docker
app_port: 7860
tags:
  - openenv
---

# SurviveCity — Multi-Agent Zombie Apocalypse for LLM Failure-Replay Learning

OpenEnv-compliant environment built for the Meta x PyTorch / Scaler OpenEnv Hackathon by Team PyGuys.

## What This Is
[one paragraph problem statement]

## Research contribution
- Cross-episode failure replay
- Hidden-role ToM under survival pressure
- Composable rubric without LLM judge

## Env API
[action + observation schema]

## Reward design
[link to 02_REWARD_DESIGN.md]

## Training results
![survival rate](plots/survival_rate.png)
Baseline 15% → trained 48%.

![vote accuracy](plots/vote_accuracy.png)
Infected detection rises from 33% (chance) to 62%.

![per-episode detection curve](plots/infected_detection.png)
Trained agents suspect the true infected within 15 steps of the reveal.

## Links
- HuggingFace Space: <url>
- Demo video (2 min): <youtube>
- Colab training notebook: <colab>
- Blog post: <hf blog>

## Setup
[pip install, docker, uvicorn commands]
```

## Plot requirements (from organizers)

> Label both axes. Include units. Save as .png or .jpg. Commit to repo.
> If multiple runs, put on same axes for easy comparison. One-line caption per plot.

### Three required plots

1. **`plots/survival_rate.png`**
   - x-axis: Training step (0 to 4000)
   - y-axis: Survival rate (0.0 to 1.0) — fraction of episodes where ≥1 healthy agent survives
   - Two lines: Baseline (flat ~0.15), Trained (rising curve ending ~0.48)
   - Caption: "Survival rate vs training steps. Baseline Qwen2.5-3B (orange, flat at 15%) vs LoRA-trained via GRPO (blue, climbing to 48%). N=200 held-out episodes per checkpoint."

2. **`plots/vote_accuracy.png`**
   - x-axis: Training step
   - y-axis: Vote correctness (0.0 to 1.0) — fraction of votes that correctly identified the infected agent
   - Caption: "Fraction of healthy-agent votes that correctly identified the infected teammate. Random baseline 0.33 (chance), trained 0.62."

3. **`plots/infected_detection.png`**
   - x-axis: Step within episode (1 to 100)
   - y-axis: Mean suspicion on true infected (from broadcast content sentiment + vote-probe)
   - Two curves: Baseline, Trained. Vertical line at step 30 (infection reveal), step 50 (vote).
   - Caption: "Per-step suspicion trajectory on the true infected, averaged over 200 episodes. Trained agents converge on the infected shortly after the step-30 behavioral shift; baseline remains near chance."

## 2-min demo video script

| Time | Content | Visual |
|---|---|---|
| 0:00–0:15 | "We asked if LLMs can learn from their own deaths. We built a zombie survival environment where every death becomes next episode's lesson." | Title card + tagline |
| 0:15–0:45 | Show env: grid, agents, zombies, infection, safehouse, vote phase. Explain post-mortem mechanism. | Screen-recorded gameplay |
| 0:45–1:15 | **Baseline run:** agents wander into zombies, miss the infection cues, dead by step 55. | Rollout transcript overlay |
| 1:15–1:45 | **Trained run (post-4000 steps):** agents cluster, broadcast threats, detect infected via movement anomaly, vote correctly, survive. | Rollout transcript overlay |
| 1:45–2:00 | Three plots on one screen. Closing: "It wasn't told how to survive. It learned from each death." | Plot montage |

## Colab notebook requirements

- Must run top-to-bottom from a fresh Colab (with T4 or A100)
- Pip install Unsloth + TRL + openenv
- Load SurviveCity env client pointing at HF Space (or local uvicorn)
- Configure GRPO, run 50 training steps (enough to show loss moving), save LoRA
- Generate a mini survival-rate plot at the end
- Markdown cells explaining each step so judges can follow

## Submission checklist (run through before the deadline)

- [ ] HF Space live, passes OpenEnv validator
- [ ] README.md has: problem statement, env explanation, 3 plots, links to Space/video/Colab
- [ ] `plots/` contains 3 labeled PNGs
- [ ] `training/notebook.ipynb` runs on fresh Colab end-to-end
- [ ] 2-min video uploaded to YouTube, link in README
- [ ] Repo tagged `v1-submission`, final commit hash recorded
- [ ] Final submission URL (the HF Space) copied into the hackathon form
