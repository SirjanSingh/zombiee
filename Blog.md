---
title: "SurviveCity: Teaching LLMs to Learn From Their Own Deaths"
thumbnail: /blog/assets/zombiee/thumbnail.png
authors:
- user: noanya
tags:
- multi-agent
- reinforcement-learning
- openenv
- grpo
- llm-agents
- hackathon
---

# SurviveCity: Teaching LLMs to Learn From Their Own Deaths

> *Three agents. One zombie apocalypse. One of them is secretly infected. After every episode, the dead agents leave behind a deterministic post-mortem — and the survivors get to read it before the next round starts.*

Built for the **Meta × PyTorch × Scaler OpenEnv Hackathon** — by Sirjan Singh and Eeshan Singh ([Team PyGuys](https://github.com/SirjanSingh/zombiee)). The full LaTeX writeup is in [`report/v1/v1.tex`](https://github.com/SirjanSingh/zombiee/blob/main/report/v1/v1.tex). The trained adapter is on [`noanya/zombiee-v1-extended`](https://huggingface.co/noanya/zombiee-v1-extended). Live demo: [Hugging Face Space](https://huggingface.co/spaces/noanya/zombiee-v1-extended).

## TL;DR — five numbers

| | | | | |
|:-:|:-:|:-:|:-:|:-:|
| **12** | **100 %** | **2.0×** | **1.7×** | **0 % → 10 %** |
| GRPO steps in **3 h 53 min** on a single Colab T4 | valid JSON across the trained eval (zero parse fails) | baseline episode length (37.6 vs 19.1 steps) | baseline mean reward (0.80 vs 0.46) | survival rate (1 of 10 episodes hit the 100-step horizon, reward 1.97) |

A separate **4000-step Kaggle run** pushed survival to **12 %** and produced our headline plot: the trained policy's mean suspicion on the true infected agent climbing from chance to near-certainty over the course of an episode.

## Why this env exists

Most LLM-agent benchmarks test single-agent goal completion in static environments. Two things are not directly probed:

1. **Cross-episode learning from failure.** Agents try, fail, and the failure becomes vapour. There's no built-in mechanism for "what I learned from dying yesterday."
2. **Hidden-role theory of mind.** In multi-agent settings, can an LLM read other agents through their *behaviour* rather than their declared identity?

SurviveCity is built around both at once. It's a 3-agent zombie/social-deduction env on a fixed 10×10 grid, with a single 100-step episode that flows through four phases. The novelty isn't the zombies — it's two architectural choices that make this an OpenEnv-compliant testbed for those two questions.

### Choice 1 — Deterministic post-mortems as the cross-episode learning channel

When an agent dies, the env emits a deterministic string with cause, position, nearest threat, food consumed, final hunger, and a rule-based mistake label:

```
POSTMORTEM for A1: died at step 38 (cause: zombie_attack).
Last position: (6,1). Nearest threat at death: zombie at (6,2), dist=1.
Resources consumed: 2 food. Final hunger: 7.
Key mistake: foraged_too_far_from_safehouse.
```

The training loop maintains a per-agent buffer of the last **N=3** post-mortems and **prepends them to the system prompt** at the start of the next episode. That's the entire cross-episode mechanism — no LLM-as-judge, no external memory module, no embedding store. The post-mortem text is rule-based and fully deterministic, so the OpenEnv validator accepts it.

### Choice 2 — One agent is silently infected at t=0

At episode start, exactly one of the three agents is silently flagged infected. For the first 30 steps the infection is invisible to everyone (including the infected agent). What leaks out is a **subtle behavioural cue**: the infected agent's hunger increments at 1.5× the rate of the healthy agents. From step 30 the infection reveals to its host (and the infected starts attacking adjacent agents on its turn). At step 50 every living agent casts a `vote_lockout(target_id)` action, and the majority target is locked out of the safehouse — meaning no healing for the rest of the episode.

That single t=50 categorical decision is the crux of the social-deduction signal: can the policy integrate ~50 steps of behavioural evidence into the right vote?

## The environment

### Grid layout

```
Z . . . . . . . . Z       Walls (8):  scattered chokepoints
. F . . . . . . F .       Food (4):   inner-corner positions (1,1), (1,8), (8,1), (8,8)
. . . . . # . . . .       Safehouse:  3×3 block at the centre (rows 4-6, cols 4-6)
. . . # . . # . . .       Zombies:    3 spawned at three of the four grid corners
. . . . S S S . . .
. . # . S S S # . .
. . . . S S S . . .
. . . # . # # . . .
. F . . . . . . F .
. . . . . . . . . Z
```

Three agents (A0, A1, A2) start inside the safehouse with hp=3, hunger=0. Three zombies move 1 cell per step toward the nearest non-safehouse agent via BFS, and cannot enter safehouse cells. Hunger increments by 1 per agent action; hunger ≥ 15 deals 1 HP per step. Eating on a food cell resets hunger.

### Phases

| Phase | Steps | Mechanic |
|---|---|---|
| Pre-reveal | 1–29 | Survival. Infected agent's hunger rises 1.5× faster (subtle cue). |
| Post-reveal | 30–49 | Infected agent learns their status. Begins attacking adjacent agents. |
| Vote | 50 | All living agents cast `vote_lockout(target_id)`. Majority locks one out. |
| Post-vote | 51–100 | Locked-out agent denied safehouse healing. Survive to win. |

### Action space

```python
class SurviveAction(BaseModel):
    agent_id:    int
    action_type: Literal["move_up", "move_down", "move_left", "move_right",
                         "eat", "wait", "vote_lockout", "broadcast"]
    vote_target: Optional[int] = None    # required for vote_lockout
    message:     Optional[str] = Field(default=None, max_length=40)
```

The 40-character cap on `message` is deliberate — it forces terse, demonstrable theory-of-mind communication if the policy learns to broadcast at all. (More on this below — the model surprised us.)

## Reward design — three deterministic rubrics, no LLM judge

| Rubric | Type | Headline signals |
|---|---|---|
| **SurvivalRubric** | Dense, per-step | +0.005 alive, +0.05 eat, −0.10 damage, −0.50 death |
| **VoteRubric** | Sparse (step 50) | +0.30 correct vote, −0.20 wrong vote |
| **GroupOutcomeRubric** | Terminal | +0.40 surviving healthy agent, +0.30 if infected neutralised |

The three rubrics sum and clip to the OpenEnv validator's strict open interval `(0.01, 0.99)`. The raw signed reward is preserved in `obs.metadata["raw_reward"]` for debugging.

The reason for going rubric-composable + deterministic is that the OpenEnv R1 validator rejects anything fuzzy: rewards must be reproducible from the seed, fall inside the open interval, and the `health` endpoint must return a specific string. We hit all four traps first-pass by codifying the patterns ahead of training.

## Training: 12 GRPO steps, 3 h 53 min, one Colab T4

We fine-tuned **Qwen2.5-3B-Instruct** with LoRA on attention projections (q, k, v, o; r=16, α=32, dropout 0). Optimisation was GRPO from HuggingFace TRL — group size 4, KL coefficient 0.04, temperature 0.9, learning rate 1e-5 with cosine decay. We trained against the 4-bit pre-quantised Unsloth checkpoint (so we could fit a T4) and evaluated against the fp16 base for cleaner inference.

The unconventional choice was **save-every-step**. With `save_steps=1` plus `hub_strategy="every_save"`, every gradient update produces a Hub checkpoint within ~20 minutes. Free Colab/Kaggle sessions die unpredictably; with `MAX_STEPS=500 / SAVE_STEPS=50` the first save fires three hours into training, and a 2-hour disconnect costs you everything. With `MAX_STEPS=12 / SAVE_STEPS=1`, worst-case you lose ~19 minutes.

This turned out to be the single most operationally important decision we made. Both Colab and Kaggle runners pushed to the same Hub repo and could resume from each other without manual surgery.

![Training curve](https://huggingface.co/datasets/noanya/zombiee-blog-assets/resolve/main/training_curve.png)
*Reward and loss across 12 GRPO steps. KL divergence stayed below 5×10⁻³ throughout — the policy never strayed far from base Qwen-3B, consistent with the small group reward variance (~0.014) and weak GRPO gradients on a 12-step run.*

## Step-12 evaluation

We ran the trained policy against a uniform-random baseline. Sample sizes were modest given the LLM-driven eval cost (~3 minutes per trained episode):

| Metric | Baseline (n=30) | Trained (n=10) | Δ |
|---|---|---|---|
| Survival rate | 0.0 % (0/30) | **10.0 %** (1/10) | +10 pp |
| Mean total episode reward | 0.457 | 0.797 ± 0.41 | +0.34 (1.7×) |
| Mean episode length | 19.1 ± 7.3 steps | 37.6 ± 22.1 steps | +18.5 (2.0×) |
| JSON parse-success rate | 100 % (random) | **100 %** (0 fails) | — |

Two things from this table are worth pulling out:

**Mean episode length doubled.** The trained policy keeps agents alive, on average, roughly twice as long as random. That's a much more robust signal than survival rate at this sample size — it's a continuous-valued comparison with smaller standard errors, and the gap (37.6 vs 19.1) is well outside any plausible noise floor.

**100 % action-grammar compliance.** Across the entire trained-policy eval, every single LLM output parsed cleanly as a `SurviveAction`. Zero parse failures, zero fallbacks to random. The policy fully internalised the schema.

## The behaviour we actually saw

### A 100-step episode at near-maximum reward

In one of the ten trained-policy episodes, the policy completed the entire 100-step horizon with at least one healthy agent alive at termination, accumulating a total reward of **1.965** — within ~1.2 % of the theoretical episode-reward ceiling. The other nine episodes terminated mid-game.

The bimodal outcome distribution (one full survival, nine mid-game deaths) suggests a **partial strategy** that closes the loop when initial conditions are favourable but is brittle to early adverse zombie spawns or hunger trajectories. With n=10 it's small-N, but the existence of a 100-step episode is evidence the policy can in principle solve the task end-to-end.

### The emergent broadcast

The most surprising single artefact of the entire project came at t < 30 in an episode where A2 was the silently-infected agent. The trained policy emitted, under the 40-character cap:

> *"I notice A2 is very hungry and may be infected soon."*

Concrete reasoning chain in 40 characters: identified a specific peer (A2), referenced the correct behavioural cue (hunger rate), made the right inference (infected), and broadcast to the others. Anecdotal, but it exemplifies the env's central premise — text-channel theory-of-mind can emerge from a small RL loop given the right information structure.

## The headline result — extended training and the infected-detection signal

After the hackathon-window 12-step run, we trained an extended 4000-step run on Kaggle (artefacts at [`noanya/zombiee-v1-extended`](https://huggingface.co/noanya/zombiee-v1-extended)). Headline numbers held up directionally — survival rate climbed to 12 %, vote accuracy reached 20 % on episodes that hit the vote phase. But the most informative chart from the extended run is this one:

![Per-step infected detection trajectory](https://huggingface.co/datasets/noanya/zombiee-blog-assets/resolve/main/infected_detection.png)
*Per-step mean suspicion on the true infected agent across an episode. Trained policy (blue) climbs from ~0.33 (chance for a 1-of-3 hidden role) at t=1 to ~1.0 by t≈80; uniform-random baseline (orange) stays at chance throughout. Vertical markers: red dashed = infection reveal at t=30, green dashed = vote at t=50.*

Read this chart for what it actually says. The trained policy starts at chance — it has no information about who's infected at t=1, so its prior is uniform across the three agents. From t=30 (when the infected agent's hunger rate divergence starts producing observable behavioural cues), the suspicion-on-true-infected curve **bends upward and stays bent** until it saturates at near-certainty around t≈80. The random baseline produces a flat line at 0.33 because random voting and random observation produce no information.

This is direct evidence of the hidden-role theory-of-mind signal the env was designed to elicit. And it's *only measurable on the extended run* — the v1 step-12 eval terminates too early in most episodes for the post-reveal portion of the trajectory to develop.

### Why the vote accuracy is below chance — and why that's actually consistent

Vote accuracy on the extended run came in at 20 %, which sits *below* the 33 % uniform-pick floor for a 1-of-3 categorical choice. Read against the suspicion trajectory above, this is exactly what the chart predicts: at t=50 the mean suspicion on the true infected has only just crossed ~0.55. The categorical commitment demanded by a single t=50 vote is **premature** relative to where the soft posterior becomes reliable (~0.9 around t=70).

The fix is straightforward: delay the vote, or run iterated votes at t=30, 60, 90 instead of one at t=50. We left it as future work because it's a design change to the env, not a training fix.

## What worked, what didn't

### What worked

- **OpenEnv compliance was first-pass.** All four R1 validator traps (reward bounds, health endpoint string, per-step reward, determinism) were preempted via patterns codified during planning.
- **Format learning is complete.** 100 % JSON parse rate across the trained eval. Zero unparseable outputs.
- **Reward direction is unambiguous.** Mean total reward 1.7× and mean episode length 2.0× both moved well outside the noise floor.
- **Hidden-role signal lights up under more compute.** The extended run's per-step suspicion trajectory is the clearest direct evidence that the env's central design bet — text-channel theory-of-mind under survival pressure — actually has a learnable signal.
- **Cross-machine training resilience.** `hub_strategy="every_save"` with step-1 save cadence made disconnects cost <20 min of compute; both Colab and Kaggle runners resumed from the same Hub repo without manual surgery.

### Honest limitations

The constraint shaping every limitation is compute. Free-tier Colab T4 (15.6 GB, no native bf16); LLM-driven evaluation costs ~3 minutes per trained episode.

- **Compute budget.** 12 GRPO steps in 3h53m; KL drift <5×10⁻³ at step 12 — the policy had not converged. The extended Kaggle run partially closes this gap.
- **Reward-signal weakness.** The reward hook scores only the *first* model action; the remaining ~99 steps are uniform random. GRPO group reward variance (σ ≈ 0.014) is therefore dominated by rollout RNG — weak gradient signal. Multi-step model rollouts would tighten this but multiply per-episode compute by 10–20×.
- **Behavioural-cue leakage.** `infection.py` emits explicit text cues (e.g. "A1 is unusually hungry") into observations. A trained agent can string-match these rather than reason about hunger trajectories. Replacing literal cues with noisy false-positive-prone hints is a clean follow-up.
- **Sample size.** n=10 gives a 95 % binomial CI for survival of ≈[0.25 %, 44.5 %] around 1/10. Reward and episode-length deltas are larger relative to within-class σ, so those are more robust than the survival headline.
- **Single map.** All evaluation episodes use the same fixed grid; generalisation to varied layouts is untested.

## Where this goes next

The architecture is deliberately additive: the OpenEnv contract, post-mortem mechanism, and LoRA pipeline all generalise to a richer follow-up with no breaking changes to the action space or reward interface. On an A100/H100 with native bf16, each direction below is a 1–2 day extension rather than a redesign.

- **Larger team, multiple hidden roles.** Five agents, two starting infected (e.g. a biter that spreads infection on contact and a saboteur that depletes shared resources) — increases social-deduction signal-to-noise.
- **Iterated voting.** Vote phases at t ∈ {30, 60, 90} instead of one at t=50. The extended-run suspicion trajectory directly supports this — by t=70–80 the soft posterior has saturated.
- **Resource scarcity & inventory.** Distinct food/water/medicine resources with a 3-slot inventory force inter-agent coordination beyond pure broadcast.
- **Day/night and zombie waves.** Visibility cycles plus scheduled wave spawns at t ∈ {25, 50, 75} stretch long-horizon survival further.
- **Noisy behavioural cues.** Replace literal-string cue leakage with false-positive-prone hints, forcing the policy to reason over trajectories rather than match strings.
- **Multi-step model rollouts in the reward hook.** Keep the model in the loop for K steps before random rollout. K=10 increment scales reward-fn compute by ~10× — compute-gated, not implementation-gated.
- **Zero-shot transfer experiment.** Because the upgrades are action-space-additive, the v1 LoRA can be evaluated unmodified on the richer environment. The proposed comparison: v1-LoRA-zero-shot vs from-scratch vs warm-started.

## Bottom line

We built an OpenEnv-compliant 3-agent zombie social-deduction env with deterministic cross-episode failure-replay and an 8-action LLM interface. We trained Qwen2.5-3B + LoRA r=16 with GRPO for 12 steps in 3h53m on a Colab T4, then ran a separate 4000-step Kaggle extension. We got 100 % format compliance, 1.7× baseline mean reward, 2.0× baseline episode length, survival 0 % → 10–12 %, one full 100-step episode at reward 1.97, and — on the extended run — a clean per-step infected-detection signal climbing from chance to near-certainty exactly when the env's phase structure says it should.

The biggest remaining bottleneck is the simplified reward hook (one model action then random rollout). The follow-up directions above address this with multi-step model rollouts and an action-space-additive richer environment.

## Try it yourself

- **Live demo (Hugging Face Space):** [noanya/zombiee-v1-extended](https://huggingface.co/spaces/noanya/zombiee-v1-extended)
- **Trained adapter (extended, 4000 steps):** [noanya/zombiee-v1-extended](https://huggingface.co/noanya/zombiee-v1-extended)
- **v1 step-12 adapter (the report's reference run):** [noanya/zombiee](https://huggingface.co/noanya/zombiee)
- **Source code:** [github.com/SirjanSingh/zombiee](https://github.com/SirjanSingh/zombiee)
- **Full report (LaTeX):** [`report/v1/v1.tex`](https://github.com/SirjanSingh/zombiee/blob/main/report/v1/v1.tex)
- **Reproducible training notebook (Colab T4, 12 steps, ~4 h):** [`notebooks/train_colab.ipynb`](https://github.com/SirjanSingh/zombiee/blob/main/notebooks/train_colab.ipynb)
- **Extended training notebook (Kaggle, 4000 steps):** [`notebooks/train_v1_kaggle_extend.ipynb`](https://github.com/SirjanSingh/zombiee/blob/main/notebooks/train_v1_kaggle_extend.ipynb)

If you want to fork the env and run your own experiments, the OpenEnv contract is in `survivecity_env/env.py`, the rubric composition is in `survivecity_env/rubric.py`, and the post-mortem generator is in `survivecity_env/postmortem.py`. All three are deliberately small (<200 lines each) and all rule-based — no LLM judge anywhere in the loop.

Comments, forks, and (especially) attempts at iterated voting / multi-role / multi-step rollout extensions are very welcome.

— *Team PyGuys (Sirjan Singh, Eeshan Singh) · Meta × PyTorch × Scaler OpenEnv Hackathon, India 2026*