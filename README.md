# Constraint-Aligned Continual PEFT: Internship Proposal Artifact

This repository is a submission-ready project proposal package for quant/AI internship applications.

It is built from a single existing mechanism and provides:
- an interactive research website (live demo + math + evidence + deployment pathway),
- an authored proposal memo,
- an operational deployment playbook,
- and the original runnable validation script.

## Quick start

```bash
cd "/Users/kumar0002/Documents/New project"
python3 -m http.server 8000
```

Open `http://localhost:8000`.

## Run proof-of-concept script

```bash
cd "/Users/kumar0002/Documents/New project"
python3 -m pip install -r requirements.txt
./scripts/run_poc.sh
```

## What reviewers can do immediately
1. Run the live demo and adjust update/risk knobs.
2. Compare naive, anchor, and projected PEFT outcomes.
3. Export run metrics as JSON for internal discussion.
4. Read deployment assumptions and internship execution plan.

## Submission package
- `index.html` - primary interactive proposal artifact
- `docs/INTERNSHIP_PROPOSAL_MEMO.md` - concise authored narrative
- `docs/DEPLOYMENT_PLAYBOOK.md` - concrete integration and monitoring pathway
- `validation_experiment.py` - original executable proof-of-concept
- `scripts/run_poc.sh` - one-command PoC runner for figure + metrics generation
- `figs/validation_wide.png` - baseline proposal figure

## Project structure
- `styles.css` - visual system and responsive layout
- `src/main.js` - app bootstrap
- `src/ui/render.js` - controls, rendering, report export, decision card
- `src/ui/charts.js` - canvas chart rendering
- `src/workers/experiment-worker.js` - background compute worker
- `src/experiment/data.js` - synthetic regime generation
- `src/experiment/model.js` - frozen backbone + LoRA model
- `src/experiment/train.js` - naive / anchor / projection training
- `src/experiment/simulate.js` - portfolio simulation diagnostics
- `src/experiment/run-experiment.js` - experiment orchestration
- `src/content/references.js` - curated references list

## Continuous publishing to GitHub Pages
The workflow is already configured in:
- `.github/workflows/deploy-pages.yml`

It deploys on every push to `main` or `master`.

One-time setup in GitHub:
1. Push this repo to GitHub.
2. Go to `Settings -> Pages`.
3. Set source to `GitHub Actions`.
4. Push again (or manually run the deploy workflow).

Your site URL will be available in the workflow output.
