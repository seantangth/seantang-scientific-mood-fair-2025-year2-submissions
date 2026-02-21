# Model Gap Execution Tracker v2

## Scope
Track real training-strategy validation runs (not submission-only probes), and eliminate directions sequentially.

## Baseline
- F1 floor to beat: `0.939961` (all-ones-equivalent).
- Any run with online `F1 < 0.939961` is marked `eliminated`.

## Status Legend
- `pending`: not started
- `in_progress`: currently executing
- `eliminated`: tested and did not beat baseline
- `keep_candidate`: beat baseline or worth one confirm run
- `promoted`: moved to final-candidate set

## Ordered Queue

| Order | ID | Track | Core Idea | Status | Run ID | Online F1 | Decision | Notes |
|---|---|---|---|---|---|---:|---|---|
| 1 | `A1_logit_adjust_xgb_official` | A | Prior-shift/logit adjustment over official labels | `keep_candidate` | `f1push_ranker_v1_20260214_075444` | 0.939968 | `beat_baseline` | Best so far `a1r754_q00001_v1.zip` (`f1=0.9399677`, `mcc=0.0100`); `a1r754_q00005_v1.zip` dropped to `f1=0.9399532`, so q should stay ultra-small. |
| 2 | `A2_balanced_softmax_style_binary` | A | Prior-aware reweight/loss (not naive inverse-frequency) | `eliminated` | `a2_balanced_xgb_v1_20260214_113741` | 0.939954 | `below_baseline` | `a2r137_q00001_v1.zip`: `f1=0.9399540`, `mcc=-0.00128`; below baseline `0.939961`. Drop per gate rule. |
| 3 | `A3_focal_cb_loss` | A | Focal + class-balanced loss | `eliminated` | `a3_focal_xgb_v1_20260215_122316` | 0.939954 | `below_baseline` | `a3r_q00001_v1.zip`: `auc=0.387470`, `acc=0.886711`, `f1=0.9399540`, `mcc=-0.001282`; below baseline `0.939961`. |
| 4 | `B1_group_dro_by_station` | B | Optimize worst-station behavior | `eliminated` | `b1_groupdro_xgb_v1_20260214_122519` | 0.939954 | `below_baseline` | `b1r519_q00001_v1.zip`: `f1=0.9399540`, `mcc=-0.00128`; below baseline `0.939961`. |
| 5 | `B2_union_label_with_official_head` | B | Dual-head label strategy | `eliminated` | `b2_union_xgb_v1_20260214_130826` | 0.939954 | `below_baseline` | `b2r826_q00001_v1.zip`: `f1=0.9399540`, `mcc=-0.00128` (< `0.939961` baseline). |
| 6 | `B3_nnpu_route` | B | Positive-unlabeled style risk control | `eliminated` | `negmine_v1_20260215_061254` | 0.939954 | `below_baseline` | `b3n254_b037_q00001_v1.zip`: `f1=0.9399540`, `mcc=-0.00128` (< `0.939961` baseline). |
| 7 | `C1_catboost_official` | C | Backbone swap to CatBoost | `eliminated` | `c1_distill_xgb_v1_20260215_152603` | 0.939954 | `below_baseline` | `c1d603_q00001_v1.zip`: `f1=0.9399540`, `mcc=-0.00128` (< `0.939961` baseline). |
| 8 | `C2_tcn_or_inceptiontime_official` | C | Temporal CNN backbone | `eliminated` | `c2_tcn_v1_20260215_092620` | 0.939954 | `below_baseline` | `c2r_q00001_v1.zip`: `auc=0.635056`, `acc=0.886711`, `f1=0.9399540`, `mcc=-0.001282`; below baseline `0.939961`. |
| 9 | `D1_consensus_rank_combo` | D | Multi-run consensus low-risk ranking (A1/A2/B1/B2/A3) | `eliminated` | `d1c_submission_pack_v1` | 0.939954 | `below_baseline` | `d1c_q00001_v1.zip`: `auc=0.643426`, `acc=0.886711`, `f1=0.9399540`, `mcc=-0.001282`; below baseline `0.939961`. |
| 10 | `D2_station_quota_consensus` | D | Station-quota consensus low-risk ranking | `promoted` | `d2s2_submission_pack_v1` | 0.939968 | `tie_break_win` | `q00001` is best: `d2s2_q00001_v1.zip` (`auc=0.643427`, `acc=0.886736`, `f1=0.9399677`, `mcc=0.010035`); `d2s2_q00005_v1.zip` degraded (`f1=0.9399532`, `mcc=0.004313`). |
| 11 | `E1_negexpert_xgb_prior_shift` | E | New retrain route: dedicated non-flood expert under prior-shift + station-quota policy | `eliminated` | `e1_negexpert_xgb_v1_20260215_141215` | 0.939968 | `no_gain_vs_d2` | `e1r2_q00001_v1.zip` and `e1r3_q00001_v1.zip` both gave `auc=0.5000568`, `acc=0.886736`, `f1=0.9399677`, `mcc=0.010035`; tie on F1/MCC but lose AUC tie-break to `d2s2_q00001_v1.zip` (`auc=0.643427`). |
| 12 | `E2_official_hourly_richer_features` | E | Official-only hourly sequence features + wider q regime (`0.005~0.04`) | `eliminated` | `e2_official_hourly_v1_20260215_185023` | 0.939636 | `below_baseline` | Online gates all below baseline: `q00100=0.939636`, `q00200=0.939113`, `q00500=0.937970`, `q01500=0.934353`; no configuration beats `0.939961`. |
| 13 | `E3_rankpair_nonflood_objective` | E | XGBRanker(pairwise) for direct non-flood ranking on official-only hourly features | `eliminated` | `e3_rankpair_official_v1_20260216_074256` | 0.939954 | `below_baseline` | Online gates stayed below baseline: `q00001=0.939954`, `q00002=0.939954`, `q00005=0.939953` (`auc~0.547234`); no gain over D2. |
| 14 | `E3b_rankpair_with_stage2_veto` | E | Post-E3 two-stage decision: rankpair top-q + 24h/72h safety veto before flip | `eliminated` | `e3b256_submission_pack_v1` | 0.939954 | `below_baseline` | Two online gates are identical and below baseline: `e3b256_q00001_m1015_v1.zip` = `e3b256_q00002_m1015_v1.zip` (`auc=0.5472339629`, `acc=0.8867106600`, `f1=0.9399540468`, `mcc=-0.0012819145`); stop E3b to save submission quota. |
| 15 | `E4_stationblend_official_timecv` | E | Global+station-specialist blended ranker with time-based CV selection | `eliminated` | `e4_stationblend_official_v1_20260216_191013` | 0.939837 | `below_baseline` | All three online gates are below baseline (`0.939961`): `q00500=0.9381887288`, `q00050=0.9398366000`, `q00100=0.9396227702`; best is `q00050`, but still not enough. |
| 16 | `E5_robustneg_official_uncertain_band` | E | Robust-negative retrain with uncertain-band down-weighting and NaN-safe future-window labels | `eliminated` | `e5_robustneg_official_v1_20260217_051752` | 0.939968 | `below_incumbent` | Final gates: `q00001=q00002` (`f1=0.9399677`, `mcc=0.010035`, `auc=0.624829`), `q00005` gave higher MCC (`0.010847`) but lower F1 (`0.9399669`); route cannot beat D2 under F1-first ranking. |
| 17 | `E6_marginreg_official_uncertainty_ranker` | E | New retrain route: future margin regression + uncertainty penalty for non-flood ranking | `eliminated` | `e6_marginreg_official_v1_20260217_124853` | 0.939954 | `below_incumbent` | Final gates: `q00001=q00002` (`auc=0.665859`, `f1=0.939954`, `mcc=-0.001282`), `q00005` improved MCC (`0.004313`) but reduced F1 (`0.939953`); route cannot beat D2 under F1-first ranking. |
| 18 | `E7_neustg_twostage_official` | E | Official 12-station NEUSTG training + two-stage gate/rank decision | `eliminated` | `e7_neustg_twostage_v1_20260217_195931` | 0.939954 | `below_baseline_and_unstable_gate` | `q00001` and `q00002` are identical and below baseline (`f1=0.939954 < 0.939961`); `q00005` failed twice (original + repack), so route is closed to save submission quota. |
| 19 | `E8_stacked_meta_official` | E | Stacked meta-learner over complementary official-only base models | `eliminated` | `e8_stacked_meta_official_v1_20260218_141946` | 0.939954 | `below_baseline` | First gate `e8r946_q00001_v1.zip` is below baseline (`auc=0.6086405335`, `acc=0.8867106600`, `f1=0.9399540468`, `mcc=-0.0012819145` < `0.939961`); close E8 and stop `q00002/q00005` uploads to save quota. |
| 20 | `E9_multihorizon_meta_official` | E | Multi-horizon meta route: 14d/7d/3d non-flood heads + future-max-relative regressor | `eliminated` | `e9_multihorizon_meta_official_v1_20260218_222247` | 0.939841 | `below_incumbent` | Three online gates: `q00010` (`f1=0.9399652`, `mcc=0.013718`, `auc=0.6238`), `q00100` (`f1=0.939841`, `mcc=0.030035`, `auc=0.6238`), `q00200` (`f1=0.9396452`, `mcc=0.037707`, `auc=0.6238`). MCC improves (up to 3.8× D2) but all F1 below incumbent D2 (`0.9399677`). OOD neg_precision on test ~41% (vs CV 97.8%), confirming severe OOD generalization failure. E9 closed under F1-first rule. |
| 21 | `F1_14day_history_window` | F | Extend history window 7d→14d to capture spring-neap tidal cycle (~14d period) | `eliminated` | `f1_14day_window_v1_20260220_112808` | — | `no_submit_save_quota` | Colab full run complete: `cv_auc=0.6460` (+0.002 vs E9's 0.6441 — negligible gain). `neg_precision=0.9561` at q=0.002 (`feature_dim=291`, selected=`f1_f3`). 14d window alone does NOT fix OOD covariate shift; E9 was eliminated at similar AUC (0.644→Codabench 0.624). Closed without submission to preserve quota. Feeds F4 (14d + station-adaptive norm combined). |
| 22 | `F2_lambdamart_direct_auc` | F | Replace binary classification objective with LambdaMART (rank:ndcg) to directly maximize AUC | `eliminated` | `f2_lambdamart_v1_20260219_165110` | — | `no_submit_save_quota` | CV AUC=0.553 (below E3's 0.594 and D2's 0.643). Pattern consistent with E3 (rank:pairwise, CV AUC=0.594 → Codabench AUC=0.547). Ranking objectives underperform binary classifiers on cross-station AUC. neg_precision=0.914 at q=0.002 (below E9's 0.978). Closed without submission to preserve quota. |
| 23 | `F3_station_adaptive_norm` | F | Station-adaptive feature normalization: per-station z-score BEFORE global scaler at both train and inference time | `in_progress` | `f3_station_adaptive_v1_20260219_215436` | — | `awaiting_online_gate` | Colab full run complete: `cv_auc=0.7227` (+0.079 vs E9, +0.079 vs D2 — largest AUC jump in all experiments), `neg_precision=0.9649` at q=0.002, `feature_dim=165`, selected=`e9_q1`. NaN packer bug found and fixed (bare `NaN` → `float('nan')`). All ZIPs repacked locally. `f3r_q00010_v1.zip` re-submitted to Codabench, awaiting result. |
| 24 | `F4_14day_adaptive_norm` | F | Combine F1 (14-day window, 291-dim) + F3 (station-adaptive norm): jointly attack tidal cycle coverage AND OOD covariate shift | `in_progress` | `f4_14day_adaptive_v1_20260220_183849` | — | `awaiting_online_gate` | Colab full run complete: `cv_auc=0.7484` (+0.026 vs F3, +0.104 vs E9 — best CV AUC in all experiments), `neg_precision=0.9583` at q=0.002, `feature_dim=291`, selected=`f1_q2`. NaN fix pre-applied in packer. ZIPs packed locally: `f4r_base/q00001/.../q00200_v1.zip`. Submit `f4r_q00010_v1.zip` first gate after F3 result clears. |

## Gate Rules
1. Run strictly in queue order.
2. Max 8 training runs in this cycle.
3. No new probe variants unless an entry is `keep_candidate` or `promoted`.
4. If A1-A3 and B1-B3 all `eliminated`, stop expansion and keep best stable package.

## Current Outcome
- Best online result now is `d2s2_q00001_v1.zip` by tie-break (`f1=0.9399677`, `mcc=0.010035`, `auc=0.643427`).
- C2 did not improve over baseline.
- A3 (`focal_cb`) online gate also failed to beat baseline; keep A1 as final incumbent.
- Plateau diagnosis: inference policy currently compresses nearly all outputs above 0.5, so many model/backbone changes collapse to near-identical binary decisions.
- D1 consensus-combo route increased AUC (`0.6434`) but still failed F1-first gate (`0.939954 < 0.939961`).
- D2 confirms a genuine policy win over A1 through station-quota allocation and AUC tie-break advantage.
- E1 route closed: no online gain over D2; keep D2 as final stable candidate.
- Root-cause audit completed (`Breakthrough_0p95_RootCause_and_Roadmap_v1.md`): recent `q` ranges were too small for 0.95 target; next cycle must expand q regime and move to richer training features/data.
- E2 route closed as eliminated: all tested q values remained below baseline (`best=0.939636 < 0.939961`).
- E3 route closed as eliminated: `q00001/q00002/q00005` all below baseline.
- E3b route closed as eliminated: `q00001_m1015` and `q00002_m1015` are identical and below baseline (`0.939954 < 0.939961`).
- E4 route closed as eliminated: `q00500/q00050/q00100` all below baseline (`best=0.9398366 < 0.939961`).
- E5 route closed as eliminated: best F1 ties D2 but loses AUC tie-break, and final `q00005` lowered F1.
- E6 route closed as eliminated: best AUC was high but all tested q values stayed below D2 on F1-first gate.
- E7 route closed: `q00001/q00002` are below baseline and `q00005` failed twice on platform, so E7 is eliminated to preserve submission quota.
- E8 route closed as eliminated: first gate `e8r946_q00001_v1.zip` is below baseline (`0.939954 < 0.939961`), so `q00002/q00005` are skipped to preserve submission quota.
- E9 route closed as eliminated: MCC improved significantly (up to 3.8× D2 at `q00200`), but all F1 values stayed below D2 incumbent (`0.9399677`) due to OOD neg_precision collapsing from CV 97.8% to ~41% on test. Root cause confirmed: covariate shift between 9 training stations and 3 OOD test stations.
- F-track opened (2026-02-19): three new routes targeting root causes — F1 (14-day window for tidal cycle), F2 (LambdaMART direct AUC), F3 (station-adaptive normalization at inference).
- F2 route closed without submission (2026-02-20): CV AUC=0.553 is below E3 (rank:pairwise, CV=0.594) and D2 (0.643). Ranking objectives (rank:pairwise/rank:ndcg) consistently underperform binary classifiers on this problem. Quota preserved.
- F3 Colab full run complete (2026-02-20): CV AUC=0.7227 — largest single-experiment AUC jump in all experiments (+0.079 over D2's 0.643). neg_precision=0.9649 at q=0.002. ZIPs packed; `f3r_q00010_v1.zip` submitted to Codabench, awaiting result.
- F1 Colab full run complete (2026-02-20) but eliminated without submission: CV AUC=0.6460 (+0.002 vs E9's 0.6441 — negligible gain). 14d window alone does NOT fix OOD covariate shift. Feeds F4.
- F4 route bootstrapped (2026-02-20): scripts + notebook ready. F4 = F1 (14d, 291-dim) + F3 (station-adaptive norm). Expected to further improve over F3 by combining both root-cause fixes.
- F3 NaN packer bug found (2026-02-21): `json.dumps(float('nan'))` → bare `NaN` token in model.py → `NameError` on Codabench. Fixed in both F3 and F4 packers via `re.sub`. F3 ZIPs repacked and `f3r_q00010_v1.zip` re-submitted.
- F4 Colab full run complete (2026-02-21): `cv_auc=0.7484` — best CV AUC in all 24 experiments (+0.026 vs F3, +0.104 vs E9). Synergy confirmed: 14d window + station-adaptive norm outperforms each individually. ZIPs packed and ready. Submit `f4r_q00010_v1.zip` after F3 gate result.

## Logging Rule
After each run:
1. Append one row update here.
2. Append one line in `0_README/Training_Tracking.md`.
3. Append one line in `0_README/Daily_Progress.md`.
