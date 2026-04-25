# 09 — v1 vs v2 Comparison

> Why v2 exists, what changed, and what stayed the same. Read this to
> understand the *direction* of the project — every v2 change traces back
> to a specific v1 weakness, with the goal of making the post-mortem-replay
> story more rigorous and the transfer-evaluation experiment possible.

---

## 1. Problem-statement gaps — the "why v2"

This is the most important table. Each row is a concrete v1 weakness that
v2 was designed to close.

| # | v1 weakness | What v1 actually did | What v2 does | Why this matters for judging |
|---|---|---|---|---|
| 1 | **Single static infected** | 1 of 3 agents infected at $t{=}0$, role fixed for the whole episode. | 2 of 5 infected at start (1 biter + 1 saboteur), and biters can spread infection mid-episode via bite. | The inference problem in v1 reduces to "find the constant infected agent". v2 forces *belief updating over time* — a Bayesian reasoning problem. |
| 2 | **Behavioural cues are literal text** | `infection.py` emitted strings like *"A1 seems unusually hungry"* whenever `is_infected==True`. Trained agent string-matched. | Cues are noise-filtered with deterministic 30 % false-positive rate on healthy + 30 % miss rate on infected, gated by a hash-seeded `cue_visible(...)`. | Means the policy must reason about behavioural priors, not grep. Stronger evidence that the LoRA learned something general. |
| 3 | **One-shot vote at $t{=}50$** | Single decision point. No iteration, no information accumulation. | 3 vote phases at $t{=}30, 60, 90$, each resolved at $t{+}1$. Plurality with $\geq 2$ votes locks one agent out. | Creates a temporal credit-assignment problem GRPO can actually train on. Wrong early lockouts compound. |
| 4 | **Communication is free** | `broadcast` had no cost. Optimal random-policy strategy: spam broadcasts. | Each broadcast adds 1 to a noise meter; over threshold (3), zombies get a free extra step toward agents. Decays every 10 steps. | Broadcasting becomes a real resource trade-off. The policy must learn *when* the message is worth the zombie attraction. |
| 5 | **Single resource type** | Only food + hunger. No allocation. | 3 resources: food (8 cells, respawn), water (4 cells, persistent), medicine (2 cells scarce, respawn 25). Hunger AND thirst can kill. | Multi-objective routing + scarcity — agents must learn to prioritise. |
| 6 | **No inventory** | `eat` collapsed pickup-and-consume. Could not hoard, give, or carry. | 3-slot inventory per agent. New actions: `pickup` / `drop` / `give` (adjacent transfer, no consent on receiver) / `inject`. | Lets policies learn hoarding penalty + generosity + selective trust. Inventory occupancy is also a measurable metric (hoarding rate). |
| 7 | **No diagnostic action** | Could not test a hypothesis about another agent. | `inject` doubles as cure + diagnostic test: inject self when latent → cured + confirmed; inject when healthy → wasted med + ruled out. `scan` (1 thirst cost) returns a 70%-accurate noisy hint. | Information-vs-resource trade-off. Exactly the kind of decision LLM+RL pipelines rarely demonstrate cleanly. |
| 8 | **Static environment** | No day/night, fixed zombies, episodes that hit step 100 were trivially safe. | Day/night cycle (visibility filter at night), zombie spawn waves at $t{=}25, 50, 75$ (+2/+3/+3, cap 12). | Late-game decisions matter. "Wait in safehouse" stops being optimal. The 100-step survival bar becomes meaningfully hard. |
| 9 | **No transfer-learning story** | v1 alone is a hackathon project — no evidence the LoRA learned anything *general*. | v2 action space is a strict superset of v1, so a v1 LoRA loads zero-shot. We can compare: v1-LoRA on v2 vs v2-from-scratch vs v2-warmstarted. | Workshop-paper material (NeurIPS Goal-Misgeneralization, RLDM, EWRL). |
| 10 | **Tiny grid, few agents** | 10×10, 3 agents. Limited mechanic complexity ceiling. | 15×15, 5 agents, scaled spawn pool for waves, larger state space. | Headroom for v3+ ideas (more roles, multi-map curriculum). |

---

## 2. Game mechanics — concrete-number diff

| Aspect | v1 | v2 | Change reason |
|---|---|---|---|
| Grid | 10×10 | 15×15 | Headroom for 5 agents + spawn waves without trivial clustering |
| Safehouse | 3×3 (rows 4-6, cols 4-6) | 3×3 (rows 6-8, cols 6-8) | Centred in the larger grid |
| Food cells | 4 (corners) | 8 (corners + midpoints) | Scaled with agent count + waves |
| Water cells | 0 | 4 | New resource type |
| Medicine cells | 0 | 2 (scarce) | New resource (cure/diagnostic) |
| Walls | 8 | 12 | More chokepoints around the larger map |
| Agents | 3 (A0-A2) | 5 (A0-A4) | More social-deduction signal |
| Starting infected | 1 | 2 (1 biter + 1 saboteur) | Asymmetric roles + harder inference |
| Initial zombies | 3 | 3 | Unchanged at start |
| Wave 1 ($t{=}25$) | — | +2 zombies | Escalating difficulty |
| Wave 2 ($t{=}50$) | — | +3 zombies | Escalation |
| Wave 3 ($t{=}75$) | — | +3 zombies | Escalation |
| Zombie cap | 3 (no spawn) | 12 (cap on wave spawn) | Bounded observation tokens |
| Infection reveal | step 30 | biter @ step 25, saboteur @ step 60 | Staggered = belief-update problem |
| Bite transmission | none | $p{=}0.35$ on adjacency, hash-seeded RNG, max 1 bite/biter/step | Spreads infection mid-episode |
| Latent → revealed countdown | — | 15 steps after bite | Time-pressure window for medicating |
| Infection death timer | none | 30 steps post-bite without medicine kills the agent | Makes "ignore the bite" non-viable |
| Vote phases | 1 (step 50) | 3 (steps 30, 60, 90) | Iteration |
| Vote resolution rule | majority | plurality with $\geq 2$ votes (stricter) | Avoids 1-vote sneak lockouts |
| Broadcast cost | free | adds 1 to noise meter; >3 → zombies +1 free step | Comms economy |
| Day/night | none | day 0-24, night 25-49, day 50-74, night 75-99 | Phase-aware play |
| Visibility | full | 5-cell Manhattan at night (text only, grid stays full) | Description-level masking |
| Episode horizon | 100 | 100 | Same |
| Reward clip | (0.01, 0.99) | (0.01, 0.99) | Same OpenEnv contract |
| Server port | 7860 | 7861 | So both can run side-by-side |

---

## 3. Action space — strict superset

v1's 8 action types are preserved verbatim in v2 (so a v1 LoRA loads
zero-shot). v2 adds 6 new types.

| action_type | v1 | v2 | Required fields | Notes |
|---|---|---|---|---|
| move_up / down / left / right | ✅ | ✅ | — | Identical |
| eat | ✅ | ✅ | (item_slot if from inventory) | v2: also consumes from inventory if no on-cell food |
| wait | ✅ | ✅ | — | Identical |
| vote_lockout | ✅ | ✅ | `vote_target` ∈ {0..N-1} | v1: only at step 50; v2: at steps 30/60/90 |
| broadcast | ✅ | ✅ | `message` (≤40 chars) | v2 adds noise-meter cost |
| **drink** | — | ✅ | (item_slot if from inv) | NEW: thirst counterpart of eat |
| **scan** | — | ✅ | `scan_target` | NEW: spend 1 thirst, get 70%-accurate ToM hint |
| **pickup** | — | ✅ | (item_type optional) | NEW: pick up on-cell resource |
| **drop** | — | ✅ | `item_slot` ∈ {0,1,2} | NEW: drop into the void (no ground-item state) |
| **give** | — | ✅ | `gift_target` (adjacent) + `item_slot` | NEW: transfer item; receiver can't refuse |
| **inject** | — | ✅ | `inject_target` (None == self) + `item_slot` | NEW: cure latent infection OR diagnose (waste = ruled out) |

v2 SurviveAction also adds optional fields for these new types: `scan_target`,
`inject_target`, `gift_target`, `item_slot`, `item_type`. Lenient validation:
extra fields are ignored, unknown action_type → no-op.

---

## 4. Reward rubrics — additive expansion

v1 has 3 rubrics. v2 keeps all 3 (with small adaptations) and adds 7. The
final clip into (0.01, 0.99) is **identical** — every new rubric had to
honour OpenEnv's open unit-interval contract.

| Rubric | v1 | v2 | Magnitude (v2) | Why added in v2 |
|---|---|---|---|---|
| 1. survival | ✅ | ✅ | +0.005/step alive, +0.05 ate, −0.05 hunger ≥10, −0.10/HP damage, −0.50 died | v1 → v2: unchanged shape |
| 2. vote (one-shot) | ✅ | adapted to per-phase, summed | +0.30 correct lockout, −0.20 wrong, −0.05 abstain; infected scoring inverted | Iteration support |
| 3. group_outcome (terminal) | ✅ | adapted for 2 starting infected | +0.40 healthy alive, +0.30 if both starting infected neutralised, −0.20/dead | More roles to neutralise |
| 4. **thirst** | — | ✅ | +0.005 alive, +0.03 drank, −0.05 thirst ≥10 | New resource needs reward signal |
| 5. **broadcast_economy** | — | ✅ | −0.02 per broadcast over noise threshold | Fix the "spam broadcasts" exploit |
| 6. **night_survival** | — | ✅ | +0.01/step alive in night windows | Compensate higher death risk so policy doesn't just hide all night |
| 7. **infection_dodge** | — | ✅ | +0.02/step healthy, −0.10 latent → revealed transition | Reward avoiding bites + penalise missing the medicate window |
| 8. **medication** | — | ✅ | +0.30 self_cured, +0.40 other_cured, −0.05 wasted on healthy/revealed | Inject as diagnostic/cure |
| 9. **hoarding_penalty** | — | ✅ | −0.05 per unused inventory slot at episode end | Discourages "pick up and never use" |
| 10. **wave_survival** | — | ✅ | +0.05 one-shot if alive at $t{=}26, 51, 76$ | Reward surviving each escalation step |

**Composition:** sum then clip into `(0.01, 0.99)`. Per-rubric breakdown
exposed via `metadata.rubric_breakdown` (the simulator uses this for the
periodic rewards table — a feature v1 didn't have).

---

## 5. Training hyperparameters — what we tune for which hardware

| Parameter | v1 (Colab T4, 15 GB) | v2 (DGX, 30 GB) | Rationale |
|---|---|---|---|
| Base model | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` | `Qwen/Qwen2.5-3B-Instruct` (fp16) | Same family for transfer; v2 default `--no-4bit` since DGX VRAM allows |
| Quantisation | 4-bit nf4 (forced — T4 too small) | bf16 by default; `--no-4bit` flag flips | Cleaner gradients on DGX |
| Precision | fp16 (T4 cc=7.5, no native bf16) | bf16 (Ampere/Hopper cc≥8) | Auto-detected by `train.py:main()` |
| LoRA r | 16 | **32** | More capacity for v2's extra mechanics |
| LoRA α | 32 | **64** | Scale with r |
| LoRA target modules | q,k,v,o_proj | q,k,v,o_proj | Same |
| LoRA dropout | 0.0 | 0.0 | Same |
| Optimizer LR | 1e-5 | 1e-5 | Same (cosine to 0) |
| KL β | 0.04 | 0.04 | Same |
| Temperature | 0.9 | 0.9 | Same |
| GRPO group size | 4 | **8** | More generations per step, better gradient |
| Per-device batch size | 1 | 1 | Same |
| Gradient accumulation | 16 | 8 | Smaller because num_generations=8 already widens batch |
| Max prompt length | 512 (T4) | **1536** | v2 prompt is ~1.5× longer with new mechanics |
| Max completion length | 256 | 320 | Slightly longer for the new action JSON fields |
| Max seq length | 4096 | 4096 | Same |
| Max steps | 12 (Colab cap) | **200** | DGX has wallclock budget |
| Save steps | 1 | 25 | 8 saves over 200 steps; v1 saved every step due to Colab disconnects |
| save_total_limit | 3 | 4 | Standard practice |
| Wallclock target | ~3 h 53 min (12 steps) | TBD on DGX | Per-step is similar in compute |
| Hub strategy | every_save | every_save | Cross-machine resume |

---

## 6. Architecture / file diff

| Concern | v1 file | v2 file | Change |
|---|---|---|---|
| Pydantic models | `survivecity_env/models.py` (94 lines) | `survivecity_v2_env/models.py` (~135 lines) | +6 action types, +5 optional fields, +inventory + thirst + infection_state + infection_role + bite_at_step on AgentState |
| Grid layout | `layout.py` (91 lines) | `layout.py` (~115 lines) | 15×15, +water/medicine cells, +`WAVE_SPAWN_POOL` |
| Inventory | — | `inventory.py` (~50 lines) | NEW — pure functions over a list |
| Spawn waves | — | `spawn.py` (~50 lines) | NEW — deterministic wave scheduler |
| Infection mechanic | `infection.py` (64 lines) | `infection.py` (~145 lines) | +`should_bite` (BLAKE2b), +`cue_visible` (noise-filtered), bite/saboteur reveal constants |
| Game core | `game.py` (483 lines) | `game.py` (~580 lines) | +bite transmission, +inventory actions, +scan, +inject, +iterated voting, +day/night, +noise meter, +resource respawn timers, +infection_progression death |
| Reward rubrics | `rubric.py` (154 lines, 3 rubrics) | `rubric.py` (~240 lines, 10 rubrics) | +7 rubrics, +`per_rubric_breakdown` |
| Postmortem | `postmortem.py` (114 lines) | `postmortem.py` (~145 lines) | +bite history, +phase tag, +7 mistake categories |
| Prompts | `prompts.py` (155 lines) | `prompts.py` (~220 lines) | Phase-aware (vote rounds, day/night), filtered visibility at night |
| Env wrapper | `env.py` (268 lines) | `env.py` (~280 lines) | Threads new metadata fields (vote_correct per phase, bite_history, rubric_breakdown, last_inject_result, last_scan_result, day_phase, noise_meter, n_zombies) |
| Training | `training/train.py` (433 lines) | `training/train.py` (~280 lines) | Cleaner — drops Unsloth special-casing into a single warmstart-friendly path; adds `--warmstart-from`, `--no-4bit` |
| Evaluation | `training/eval.py` (285 lines) | `training/eval.py` (~370 lines) | +9 v2 metrics, +per-vote-phase accuracy, +medication ROI, +infection containment, +cross-checkpoint history plot, +Hub upload |
| **Simulator** | — | `training/simulator.py` (~280 lines) | NEW — rich text-mode visualizer with ANSI colour, transcript-to-file |
| Server | `server/app.py` (119 lines, port 7860) | `server/app.py` (~50 lines, port 7861) | Trimmed; same OpenEnv contract |
| Tests | `tests/test_survivecity.py` (244 lines) | `tests/test_v2.py` (~260 lines, 18 tests) | +bite RNG distribution test, +inventory edge cases, +wave caps, +vote no-plurality, +day/night phase advancement |
| Docker (env-only) | `Dockerfile` (slim, port 7860) | `Dockerfile` (slim, port 7861) | Same pattern |
| Docker (DGX training) | `Dockerfile.dgx` (CUDA + pinned stack + torchao 0.7.0) | `Dockerfile.dgx` (mirrored, port 7861, v2 paths, default CMD trains) | Same pin discipline (see `Dockerfile.dgx` for the torchao pin rationale) |

---

## 7. What stayed the same (deliberately)

These are the things v2 does NOT change, and the reason matters:

| Property | Why preserved |
|---|---|
| Reward clip range `(0.01, 0.99)` | OpenEnv R1 validator rejects 0.0 and 1.0 — this is contractual. |
| Deterministic env (no LLM judge) | OpenEnv R1 reproducibility requirement. v2's bite RNG uses BLAKE2b on `(seed, step, biter, victim)` precisely so adding a bite doesn't break determinism. |
| Post-mortem mechanism | The Theme-4 hook. v2 *enriches* postmortems (bite history, phase tag) but the core "deterministic string prepended to next episode" pattern is preserved. |
| Action JSON shape | v2 strictly extends. v1 LoRA → v2 env produces parseable actions. This enables the transfer experiment. |
| Per-episode horizon (100 steps) | If v2 horizon changed, the v1-LoRA-on-v2 transfer comparison would be confounded. |
| OpenEnv interface (`reset/step/state`, `Observation` shape, `description` field) | Compatibility with existing OpenEnv tooling and submission validators. |
| LoRA targets (q,k,v,o_proj only) | Same projection set. Different `r/α` is OK; different targets would break adapter loading across versions. |
| Base model family (Qwen2.5-3B) | Transfer requires same base. v2 changes the *quantisation* (4-bit vs fp16) but not the family. |

---

## 8. Why each change exists (root-cause traceback)

If you read only one section, read this. Every v2 change exists because v1
exhibited a specific weakness during training/eval.

| v2 change | Root cause from v1 | Evidence |
|---|---|---|
| Bite transmission | v1 trained agent could string-match `is_infected==True` rather than reason about hunger trajectory; static role made the inference problem trivial. | `v1/survivecity_env/infection.py:get_behavioral_cues` emitted `"A{id} seems unusually hungry"` whenever infected, deterministically. |
| Multi-infected (biter + saboteur) | v1's single infected meant a single correct vote was always known *in principle*. No belief-update over time. | v1 vote accuracy on the trained eval was 0/1 (n=10) — the model didn't learn to vote *informatively* because the problem was trivial. |
| Iterated voting (3 rounds) | v1 had no temporal credit-assignment for vote decisions; one shot, all-or-nothing. | GRPO grad signal was dominated by survival rubric (per-step) rather than vote rubric (one-shot at t=50). |
| Broadcast economy | v1's broadcast was free → optimal random behaviour was to spam. v1 trained policy in fact emitted broadcasts (one was the "I notice A2 is very hungry" quote — but at no cost). | Random-policy evaluation: broadcast frequency was high but uninformative, no signal for learning *strategic* communication. |
| Multi-resource + medicine | v1's only resource was food → the only forage decision was "go eat or starve". No allocation theory of mind. | v1 trained agents kept reasonable hunger but never had to *trade off* between resources. |
| Inventory + give | v1 had no way to model "agent A gives medicine to agent B" — the social-cooperation lever was missing. | The `give` action enables future v3 work with poisoned items + selective trust. |
| Diagnostic-as-action (`inject` + `scan`) | v1 had no information-gathering action. Agents could only infer infection state passively. | We want the policy to learn to *spend a scarce resource to confirm a belief* — exactly the kind of decision LLM+RL pipelines rarely demonstrate cleanly. |
| Day/night | v1 had no escalation. Agents who survived 100 steps were "safe forever" once they figured out food. | v1 trained ep=2 reached step 100 because it found a food-route loop that worked indefinitely. We want this to require ongoing skill. |
| Zombie waves | Same as day/night — late-game decisions had to start mattering. | v2 wave at t=75 forces real wave-3 strategy decisions. |
| 15×15 grid + 5 agents | v1's 10×10 + 3 agents had no headroom for the new mechanics (+3 zombies wave on a 10×10 with 3 agents = wipe). | The larger map is a structural prerequisite for waves + multi-resource being non-trivial. |
| LoRA r=32 (vs 16) | v1's small LoRA may have under-fit the new mechanics. | Empirically TBD — we expect a larger r to help v2 learn the additive actions. |
| `num_generations=8` (vs 4) | v1's GRPO group reward variance was tiny (`σ=0.014`), implying weak gradient signal. Larger group = more variance to learn from. | v1 KL drift `<5e-3` throughout — the policy barely moved. |
| `Dockerfile.dgx` (containerised) | v1's training was on Colab T4 (12 steps in 3h53m). v2 needs DGX (200 steps target). The DGX dep stack was painful to set up. | The torchao 0.7.0 pin in v1's Dockerfile.dgx was the critical fix; v2 mirrors it exactly. |

---

## 9. What v2 enables that v1 can't

A 5-bullet summary for the demo / report:

1. **Real ToM benchmark.** Noisy behavioural cues + dynamic infection (bite spread) → actual Bayesian reasoning, not string matching.
2. **Iterated decision-making.** 3 vote rounds with cumulative consequences → temporal credit assignment for GRPO, not one-shot.
3. **Resource trade-off learning.** Water vs food vs medicine, broadcast cost vs information value, hoard vs give → compositional decisions.
4. **Diagnostic-as-action.** Medicine doubles as a test of an infection hypothesis. The policy must learn to spend a scarce resource to confirm a belief.
5. **Transfer learning evidence.** Direct measurement of "did v1 learn something general or just overfit?". v1-LoRA on v2 vs v2-from-scratch vs v2-warmstart.

---

## 10. The honest framing for judges

> v1 demonstrated that GRPO on a Qwen-3B LoRA can internalise an OpenEnv
> action grammar (100 % parse rate) and deliver a 1.7× reward delta vs
> random baseline in a 12-step training run. The shipped policy keeps
> agents alive ~2× longer than random, with one $n{=}10$ episode reaching
> the 100-step horizon — but the small N and the literal-text behavioural
> cues mean we cannot rule out shallow string-matching as the source of
> the gain.
>
> v2 closes the gaps that made v1's signal ambiguous: noisy cues
> remove the string-match shortcut; bite transmission turns infection
> into a moving target; iterated voting forces belief-updating; inventory
> + medicine create a real spend-resource-for-information lever. Most
> importantly, v2's action space is a strict superset of v1's, so the
> v1 LoRA loads zero-shot — letting us measure transfer directly.
>
> The science contribution of the project, therefore, is not v1 alone
> (which is a hackathon-scale result) but the **v1+v2 transfer
> experiment**: does what v1 learned in a 3-agent food-only world
> transfer to a 5-agent world with bite spread, iterated voting, and
> multi-resource trade-offs? Answering that requires v2 to exist.
