# Agent runbook: Powerball prediction pipeline on DGX Spark

You are an agent running this repo on an **NVIDIA DGX Spark**: GB10 Grace-Blackwell, 20 ARM64 cores, one Blackwell GPU, 128 GB unified memory. Follow the steps in order, keep a log, and report back at the end using the format at the bottom.

Game: **Australian Powerball**, 7 main numbers from 1–35 plus 1 Powerball from 1–20, drawn every Thursday. The current format started at draw #1144 (19 April 2018).

## Ground rules (do not break these)

1. **Never change the evaluation protocol to get a better-looking result.** That protocol is the walk-forward predictions, the validation/holdout split, the null runs and the acceptance test in `deep_ensemble.py`. If something crashes, fix the bug with the smallest possible change and say exactly what you changed.
2. **Do not pass `--force-model`** unless the user explicitly asks for it.
3. **Report results honestly.** No ticket set can raise the chance of the jackpot above `tickets / 134,490,400`. If the holdout or null test says "not better than chance", say so plainly.
4. **Do not run `pip install` for `torch` or `numpy` inside the NGC container.** It ships CUDA builds of both; reinstalling breaks GPU support.
5. **Work on branch `claude/determined-johnson-2azbqw`** unless told otherwise. Commit only source fixes and the `results/` folder (step 6).

## Repo map

| File | Role |
|---|---|
| `extract_powerball.py` | Scrapes results into `powerball_15_years_results.csv` |
| `deep_ensemble.py` | Heavy GPU and CPU model search with holdout and null calibration. Writes `model_probs_draw_<N>.json` and `ensemble_report_draw_<N>.json` |
| `top_predictions.py` | Scores all 6,724,520 combinations, picks a low-overlap and low-popularity ticket set, can backtest. Writes `top_predictions_draw_<N>.csv` |
| `predict_this_week.py`, `module*.py`, `*_sim.py` | Older and educational scripts. Not part of this run |

## Step 0: Environment

```bash
nvidia-smi && uname -m && nproc && free -g          # expect aarch64, 20 cores, GB10
cd <repo> && git pull origin claude/determined-johnson-2azbqw
```

Preferred: use the NGC PyTorch container. Pick the newest tag available locally or on nvcr.io.

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -it --rm -v "$PWD":/work -w /work nvcr.io/nvidia/pytorch:25.09-py3 bash
pip install pandas scikit-learn requests beautifulsoup4
python -c "import torch, sklearn, numpy; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), numpy.__version__)"
```

Alternative: a venv with a CUDA-enabled aarch64 PyTorch wheel, then `pip install -r requirements.txt`.

**Stop and report** if `torch.cuda.is_available()` is `False`. The neural families would silently run on CPU and be very slow.

## Step 1: Refresh the data

```bash
cp powerball_15_years_results.csv /tmp/pb_backup.csv
python extract_powerball.py
head -3 powerball_15_years_results.csv
```

- **Check the result:** the first row should be the most recent Thursday draw.
- **If scraping fails or returns fewer rows than before,** restore the backup (`cp /tmp/pb_backup.csv powerball_15_years_results.csv`), carry on, and report that the data is stale.
- **Year range:** `extract_powerball.py` fetches years `range(2012, 2027)`. From 2027 onwards, raise the upper bound to the current year + 1.

## Step 2: Smoke test (a few minutes)

```bash
python deep_ensemble.py --quick 2>&1 | tee logs_quick.txt
```

It should end with `Saved model_probs_draw_<N>.json ...`. Fix any crash before going further (rule 1), commit the fix, and run again.

## Step 3: Full model search (long run, use the hardware)

```bash
mkdir -p logs
nohup python deep_ensemble.py --configs 64 --seeds 8 --null-runs 20 \
      --gpu-workers 6 --cpu-workers 14 > logs/deep_ensemble.log 2>&1 &
tail -f logs/deep_ensemble.log
```

- **Scaling:** the real-data search is printed first, then one line per null run. Each null run costs about as much as the real search. More `--configs`, `--seeds` and `--null-runs` means more compute and a more reliable answer.
- **GPU nearly idle in `nvidia-smi`:** raise `--gpu-workers` (8–10).
- **Out of memory:** lower `--gpu-workers`.
- **Time budget:** if the run would take too long, run once with `--null-runs 0` to time it, then choose the null-run count that fits the budget. Use at least 10 for a meaningful p-value. For CPU-only debugging, use `--families freq markov logreg gbm`.

## Step 4: Generate tickets

```bash
N=$(ls -t model_probs_draw_*.json | head -1)
python top_predictions.py --probs "$N" --tickets 20 --mc-reps 100000 2>&1 | tee logs/top_predictions.log
```

If the user asked for a different number of tickets, use `--tickets K`. `--max-overlap 2` spreads the tickets more.

## Step 5: Backtest the ticket strategy

```bash
python top_predictions.py --tickets 20 --backtest 300 --workers 20 2>&1 | tee logs/backtest.log
```

`|z| < 2` on every division means the hit rate matches pure chance, which is expected.

## Step 6: Save and push the results

```bash
D=results/draw_$(python -c "import json,glob,os;f=max(glob.glob('model_probs_draw_*.json'),key=os.path.getmtime);print(json.load(open(f))['draw'])")
mkdir -p "$D"
cp model_probs_draw_*.json ensemble_report_draw_*.json top_predictions_draw_*.csv logs/*.log "$D"/ 2>/dev/null
git add "$D" powerball_*_results.csv
git commit -m "Results for $(basename $D)"
git push -u origin claude/determined-johnson-2azbqw
```

## Report back in this format

1. **Data:** latest draw number and date used, and whether the refresh worked.
2. **Hardware and run time:** GPU name, CUDA status, settings used, wall time.
3. **Model search,** for main balls and Powerball separately:
   - best validation and holdout gain per family
   - stacked holdout gain ± SE, z, t-test p, null p
   - ACCEPTED or REJECTED
4. **Tickets:** the table from `top_predictions.py`, plus coverage and P(at least one prize).
5. **Backtest:** the division table, with z-scores.
6. **Changes:** any code you modified and why (diff summary), and the commit hashes pushed.
