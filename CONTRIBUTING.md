# Contributing to Project Aya

Welcome to Project Aya! This guide covers our branching strategy, commit conventions, and how to get your work merged.

---

## Branching Strategy

We use a **three-tier branching model**:

```
main          (stable, release-ready)
  └── dev     (integration branch, all features merge here first)
       └── <name>/<description>   (personal working branches)
```

### Branch Rules

| Branch | Purpose | Who merges | Protection |
|--------|---------|-----------|------------|
| `main` | Stable, reviewed code only | Project leads via PR from `dev` | Protected — requires PR approval |
| `dev` | Integration and testing | Any contributor via PR from personal branch | Protected — requires PR approval |
| `<name>/<description>` | Individual work | You push freely | No restrictions |

### Your Personal Branch

Everyone works on their own branch. Branch names follow this pattern:

```
<your-name>/<short-description>

# Examples:
rob/baseline-benchmarks
sarah/kl-distillation
james/mgsm-eval-harness
priya/tokenizer-compat
```

---

## Workflow

### 1. Set Up (One-Time)

```bash
git clone https://github.com/Wayy-Research/project-aya.git
cd project-aya
git checkout dev
git checkout -b <your-name>/<feature>
```

### 2. Do Your Work

Work on your branch. Commit early, commit often, push regularly.

```bash
# Stage specific files
git add src/my_module.py tests/test_my_module.py

# Commit with a conventional message
git commit -m "feat: add KL divergence loss for distillation stage 2"

# Push your branch
git push -u origin <your-name>/<feature>
```

### 3. Submit a Pull Request to `dev`

When your work is ready for integration:

1. Make sure your branch is up to date with `dev`:
   ```bash
   git fetch origin
   git rebase origin/dev
   ```
2. Push your updated branch:
   ```bash
   git push --force-with-lease
   ```
3. Open a Pull Request on GitHub:
   - **Base branch:** `dev`
   - **Compare branch:** `<your-name>/<feature>`
   - Fill in the PR template (summary, what changed, how to test)
4. Request a review from at least one teammate
5. Address any feedback, then the reviewer merges

### 4. Merging `dev` into `main`

Merges from `dev` to `main` happen at milestone checkpoints (end of a phase, paper submission, model release). These require:

- All tests passing
- At least one project lead approval
- No unresolved review comments

---

## Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short description>

[optional body]
[optional footer]
```

### Types

| Type | When to use |
|------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `data` | Data pipeline or dataset changes |
| `eval` | Evaluation, benchmarks, metrics |
| `train` | Training scripts, configs, hyperparameters |
| `docs` | Documentation only |
| `refactor` | Code restructuring without behavior change |
| `test` | Adding or updating tests |
| `ci` | CI/CD and GitHub Actions |
| `chore` | Maintenance, dependencies, tooling |

### Examples

```
feat: implement sparse MoE routing with top-1 expert selection
fix: correct tokenizer offset for Devanagari script
eval: add mGSM benchmark runner for 10 languages
train: add gradient checkpointing to reduce VRAM usage
data: add Swahili subset to multilingual SFT dataset
docs: update roadmap with Phase 02 milestones
```

---

## Code Standards

### Python

- Python 3.10+
- Type hints on all functions
- Format with `black --line-length 88`
- Lint with `ruff check`
- Test with `pytest`

### ML-Specific

- **No lookahead bias** — never use future data in training pipelines
- **Reproducibility** — set and log random seeds, pin dependency versions
- **Config-driven** — training hyperparameters go in YAML configs, not hardcoded
- **Log everything** — use Weights & Biases or equivalent for experiment tracking

### File Organization

- Training scripts go in `scripts/`
- Model architecture code goes in `aetheris/`
- Evaluation harnesses go in `eval/`
- Notebooks are for exploration only — production code belongs in modules

---

## Pull Request Checklist

Before requesting review, confirm:

- [ ] Branch is rebased on latest `dev`
- [ ] Code runs without errors
- [ ] New code has type hints
- [ ] Tests added for new functionality
- [ ] No secrets, credentials, or large data files committed
- [ ] Commit messages follow conventional format
- [ ] PR description explains *what* and *why*

---

## Getting Help

- Open an issue for bugs or blockers
- Tag teammates in PR comments for specific questions
- Use branch naming to signal what you're working on so others can avoid conflicts

---

*Wayy Research — Buffalo, NY*
