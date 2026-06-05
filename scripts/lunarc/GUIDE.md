# BENDR Pre-training on LUNARC COSMOS — Step-by-Step Guide

This guide walks you through everything from first login to having a
pre-trained BENDR model ready for fine-tuning. Follow the steps in order.

---

## Before you start — what you need

- [ ] SUPR account approved (you applied already)
- [ ] COSMOS project allocation active (check at https://supr.naiss.se)
- [ ] **Pocket Pass** app installed on your phone (for two-factor authentication)
- [ ] Password set at https://phenix3.lunarc.lu.se/pss
- [ ] Your EDF files accessible (on a drive you can transfer from)

---

## Step 1: Log in to COSMOS

You have two options. Use **Option A** (HPC Desktop) — it's easier.

### Option A: HPC Desktop (recommended)

1. Open the **ThinLinc Client** on your computer
   - Download from: https://www.cendio.com/thinlinc/download
2. Enter the server address: `cosmos-dt.lunarc.lu.se`
3. Enter your LUNARC username and password
4. It will ask for a one-time password — open **Pocket Pass** on your phone and type the code
5. You'll see a Linux desktop. Right-click on the desktop → **Open Terminal**

### Option B: SSH (terminal only)

```bash
ssh cosmos.lunarc.lu.se -l YOUR_USERNAME
```

It will ask for your password, then a one-time code from Pocket Pass.

> **Tip:** If you get disconnected, ThinLinc will reconnect to your existing
> session. SSH won't — you'd need to use `screen` or `tmux`.

---

## Step 2: Upload your code to COSMOS

This copies the project to your home directory on COSMOS. Pick the
section that matches the machine you're sitting at. Either way you'll
be prompted for your LUNARC password and a Pocket Pass code.

### Step 2a — Set up SSH so rsync doesn't keep prompting (highly recommended)

LUNARC requires a password + a Pocket Pass one-time code on every
connection. With plain rsync this means typing both at the start of
*every* transfer, and the OTP prompt is awkward to enter mid-rsync
(common symptom: `Permission denied (gssapi-keyex,gssapi-with-mic,
keyboard-interactive,hostbased)`). The fix is an SSH **ControlMaster**:
you authenticate once interactively, and subsequent SSH/rsync calls
piggyback on that already-open session for several hours.

In your shell (macOS Terminal / Linux / WSL Ubuntu), create or edit
`~/.ssh/config` and add this block (use `nano ~/.ssh/config` if you're
not sure how — paste, then `Ctrl+O`, `Enter`, `Ctrl+X` to save and exit):

```
Host cosmos cosmos.lunarc.lu.se
    HostName cosmos.lunarc.lu.se
    User YOUR_LUNARC_USERNAME
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 4h
```

Fix the permissions (ssh refuses to use a world-readable config):

```bash
chmod 600 ~/.ssh/config
```

Then open the master connection once and exit:

```bash
ssh cosmos        # prompts for password + Pocket Pass
exit
```

For the next 4 hours, every `ssh cosmos …` and `rsync … cosmos:…` reuses
that authenticated channel without re-prompting. Re-run `ssh cosmos` to
refresh the master if it expires.

> **Sanity check before rsync:** if plain `ssh cosmos` works but rsync
> says "Permission denied", it's almost always the OTP-during-rsync
> problem. Open the master first; rsync will then just work.

### From macOS / Linux

```bash
rsync -avz --filter=':- .gitignore' --exclude '.git' \
  ~/Software/NED-Net/ \
  YOUR_USERNAME@cosmos.lunarc.lu.se:~/NED-Net/
```

The `:- .gitignore` filter makes rsync skip anything `.gitignore` skips —
so `.venv/`, `*.egg-info/`, `build/`, `dist/`, `.claude/`, `Extras/`, and
the EDF/data files stay on your laptop. `.git/` isn't in `.gitignore`, so
it's excluded explicitly.

### From Windows

You have three options — `rsync` via WSL is the closest to the macOS
flow and resumes cleanly if the link drops, so prefer that.

**Option A — WSL (recommended)**

```powershell
wsl --install                # one-time, if you don't have WSL yet
```

Then open an Ubuntu shell and run the same `rsync` line as above,
but using the WSL path to your repo:

```bash
rsync -avz --filter=':- .gitignore' --exclude '.git' \
  /mnt/c/Users/Marco/Software/NED-Net/ \
  YOUR_USERNAME@cosmos.lunarc.lu.se:~/NED-Net/
```

> If you already started an rsync that copied `.venv/` (or other ignored
> dirs) to COSMOS, clean them up before re-running:
> ```bash
> ssh YOUR_USERNAME@cosmos.lunarc.lu.se \
>   'rm -rf ~/NED-Net/.venv ~/NED-Net/*.egg-info ~/NED-Net/build ~/NED-Net/dist ~/NED-Net/.claude'
> ```

**Option B — PowerShell with built-in OpenSSH**

`scp` is included with modern Windows. Recursive, no excludes — fine
for the small code tree, less ideal for resumes:

```powershell
scp -r C:\Users\Marco\Software\NED-Net YOUR_USERNAME@cosmos.lunarc.lu.se:~/NED-Net
```

**Option C — WinSCP (GUI)**

1. Install from https://winscp.net
2. New session → Protocol `SFTP`, Host `cosmos.lunarc.lu.se`, your
   username, leave password blank (it'll prompt at connect)
3. Drag `C:\Users\Marco\Software\NED-Net\` into the right-hand pane
   pointing at `/home/YOUR_USERNAME/NED-Net/`

**How long:** A few minutes (just code, not data).

---

## Step 3: Upload your EDF files to COSMOS

> **Recommendation — start with 2-3 files.** Don't transfer the whole
> dataset on the first attempt. Upload a handful of converted EDFs,
> run the smoke tests in **Step 6**, then come back here and rsync the
> rest. The full upload takes hours-to-days; you do not want to discover
> a misconfigured path or quota issue after that completes.

Your EDF data is large (~25,000 hours), so it goes into the **project
storage** (not your home directory — home has a quota).

The SUPR project **LU 2026/2-60** covers both compute and storage on
this allocation. On COSMOS the storage maps to:

```
/lunarc/nobackup/projects/lu2026-2-60/
```

The same name is the Slurm account, so `#SBATCH -A lu2026-2-60` and
the EDF path above are both baked into the scripts in this folder —
you don't need to edit them.

Create the folder for the data (on COSMOS, via ThinLinc terminal or SSH):

```bash
mkdir -p /lunarc/nobackup/projects/lu2026-2-60/edf_data
```

Then pick the source you're transferring from.

### Recommended rsync flags for EDF transfers

Use these flags (and not `-avz`) for the big data copy:

```bash
rsync -av --partial --info=progress2 /source/ cosmos:/dest/
```

- **No `-z`** — EDFs are dense binary signals that compress poorly, so
  the per-file CPU cost of `-z` exceeds the bandwidth saving. Use `-z`
  only for the code transfer (Step 2), not the data transfer.
- **`--partial`** keeps half-transferred files on the destination so a
  re-run resumes mid-file instead of starting that file over.
- **`--info=progress2`** shows one rolling progress bar (% complete,
  ETA, throughput) for the whole transfer — far more useful than the
  per-file `--progress` output, which looks "stuck" while rsync is
  enumerating large folders.

> **Quote paths that contain spaces.** Wrap the path in double quotes
> (`"…"`) — don't rename source folders to avoid the issue, since other
> people / tools on the LU research share may depend on the original
> names. Example: `"/mnt/z/My Folder/EDFs/"`.

> **Trailing slash matters.** `…/EDF/` copies the *contents* into the
> destination; `…/EDF` (no trailing slash) creates a subfolder named
> `EDF` on the destination. Pick deliberately.

### Option 1 — From your macOS / Linux machine

```bash
rsync -av --partial --info=progress2 \
  /path/to/your/edf/files/ \
  cosmos:/lunarc/nobackup/projects/lu2026-2-60/edf_data/
```

### Option 2 — From your Windows machine

Same logic as Step 2, just pointed at the EDF folder and the LUNARC
project storage path. For terabytes of data, **use rsync via WSL** —
the WinSCP / `scp` paths don't resume cleanly if the link drops mid-transfer.

```bash
# Inside WSL (Ubuntu shell):
rsync -av --partial --info=progress2 \
  /mnt/d/path/to/edfs/ \
  cosmos:/lunarc/nobackup/projects/lu2026-2-60/edf_data/
```

WSL sees Windows drives as `/mnt/c`, `/mnt/d`, etc. If your EDFs live on
an external drive at `D:\rodent_eeg\`, that's `/mnt/d/rodent_eeg/`.

If WSL isn't an option, WinSCP works for slow first transfers — but for
the full 25,000-hour dataset, rsync's resume support is worth the WSL
install.

### Option 3 — Directly from the LU research server (`\\uw.lu.se\research`)

If the originals live on the LU shared research drive, the fastest path
is to **mount that share on a Windows machine that already has access
to it, then rsync from WSL straight to COSMOS** — no local copy needed.

#### Step 3.1 — Map the share in Windows

1. On a Windows machine on the LU network (or via LU VPN), open File Explorer
2. **This PC** → **Map network drive…**
3. Drive letter: `Z:` (anything free)
4. Folder: `\\uw.lu.se\research\<your_subfolder>`
5. Tick **Reconnect at sign-in** if you'll come back to it
6. Sign in with your LU credentials when prompted (`UW\luxxxxxx`)

Confirm Windows can see the files:

```powershell
dir Z:\path\to\edfs
```

#### Step 3.2 — Make the share visible to WSL

WSL doesn't always pick up mapped network drives automatically — and
even when it does, the connection can go idle. **Open the Z: drive in
File Explorer first** (double-click it) so Windows establishes the SMB
session. Then in WSL Ubuntu:

```bash
ls /mnt/z/
```

If that's empty or returns "cannot access", force WSL to remount Z::

```bash
sudo umount /mnt/z 2>/dev/null
sudo mkdir -p /mnt/z
sudo mount -t drvfs Z: /mnt/z
ls /mnt/z/
```

You should now see the top-level folders of the research share. **Use
tab-completion** to drill down (start typing a path, hit `Tab`) — that
guarantees you get the exact name including any spaces or special
characters.

> **Alternative — mount CIFS directly in WSL** (more robust for very
> long transfers; survives Windows logging you out / locking the screen):
>
> ```bash
> sudo apt update && sudo apt install -y cifs-utils
> sudo mkdir -p /mnt/lu-research
> sudo mount -t cifs //uw.lu.se/research /mnt/lu-research \
>   -o username=YOUR_LU_USERNAME,domain=uw,uid=$(id -u),gid=$(id -g),iocharset=utf8,vers=3.0
> ```
> Prompts for your LU password. Then use `/mnt/lu-research/…` instead
> of `/mnt/z/…` in the rsync command.

#### Step 3.3 — rsync from the mounted share to COSMOS

```bash
rsync -av --partial --info=progress2 \
  "/mnt/z/path/to/edfs/" \
  cosmos:/lunarc/nobackup/projects/lu2026-2-60/edf_data/
```

(Quotes around the path are belt-and-braces — required only if the path
contains spaces, but always safe.)

For long transfers, run rsync inside **tmux** so a closed WSL window
doesn't kill it:

```bash
tmux new -s xfer       # start a named tmux session
# … run rsync inside …
# detach: Ctrl-B then D
# reattach later from anywhere: tmux attach -t xfer
```

**Important caveats for the LU share:**

- The transfer goes **research server → your Windows box → COSMOS**.
  Your machine is the bottleneck; leaving it on overnight on a wired
  LU connection is far faster than a home Wi-Fi link.
- If your LU machine goes to sleep, the transfer pauses. Set the power
  plan to "Never sleep" while the copy is running (`powercfg /change
  standby-timeout-ac 0`).
- If WSL is unavailable on the LU machine, fall back to WinSCP pointed at
  `Z:\…` on the left and the LUNARC path on the right.
- The share path uses backslashes in Windows (`\\uw.lu.se\research`) but
  the **mounted drive letter is what you reference from WSL** — don't
  try to give rsync the UNC path directly, it doesn't translate.
- **Don't paste multi-line `\`-continued commands as a single line.**
  Each `\<space>` becomes a literal escaped space, which corrupts the
  next argument (you'll see "hostname contains invalid characters" if
  it lands on the destination). Either keep the real newlines, or put
  the whole command on one line with no `\` at all.

### How long?

Depends entirely on your link speed and total data size. Could be hours
or days for terabytes. `rsync` is resumable — if it stops, re-run the
same command and it picks up where it left off. For the largest runs,
launch it inside `tmux` or `screen` on the source machine (or use WSL +
Windows Task Scheduler to nudge it back on after sleep).

> **Important:** The `/lunarc/nobackup/` path means there is NO backup.
> Keep your original files safe on the research server.

---

## Step 4: Set up the Python environment (one-time)

On COSMOS (in the terminal), run these commands one by one:

```bash
# Go to your project folder
cd ~/NED-Net

# Load Anaconda (Python package manager)
module purge
module load Anaconda3/2024.06-1
source config_conda.sh

# Create a new Python environment called "bendr"
conda create -y -n bendr python=3.11

# Activate it
conda activate bendr

# Install PyTorch with GPU support (via pip, not conda)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies we need
pip install pyedflib scipy numpy tqdm

# Install our project
pip install -e "."
```

**How long:** 5–10 minutes.

> **Why pip and not `conda install pytorch pytorch-cuda=…`?**
> The `pytorch-cuda` conda metapackage was retired in 2024; the
> `-c pytorch` channel is no longer the recommended install path.
> The pip wheels bundle their own CUDA runtime (`cu124` = CUDA 12.4),
> so they work on any host with a driver new enough to support
> CUDA 12.4 — which the COSMOS A100 nodes do. If `nvidia-smi` on a
> GPU node shows `CUDA Version: 13.x`, cu124 is still correct
> (that field is the *max* supported runtime, not what you must install).

### Verify it worked:

The env lives in `$HOME`, which is shared with the GPU nodes — but a
proper GPU check needs to run *on* a GPU node. Two ways:

**On the login node** (quick syntax/import check; CUDA will show False):

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

Expected: `PyTorch 2.x.x+cu124, CUDA: False` — that's normal, login
nodes have no GPU.

**On a GPU node** (real verification):

```bash
interactive -A lu2026-2-60 -p gpua100 --gres=gpu:1 -t 00:10:00
# (wait for the prompt to land you on a GPU node)
module purge
module load Anaconda3/2024.06-1
source config_conda.sh
conda activate bendr
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
exit
```

You want to see something like:
```
2.6.0+cu124 | CUDA: True | Device: NVIDIA A100 80GB PCIe
```

If `CUDA: True` and a device name appears, the env is good.

---

## Step 5: Create the logs and output directories

```bash
mkdir -p ~/NED-Net/logs
mkdir -p ~/bendr_output
```

---

## Step 6: Smoke test with 2-3 EDFs (local + LUNARC)

Before committing to the full dataset upload, prove the workflow with
a handful of files. Two scripts make this cheap: one for your laptop
CPU, one for LUNARC.

### 6a — Local CPU dry run (Windows)

Catches import / path / EDF-reader bugs in minutes, with the full
traceback in front of you instead of buried in a Slurm log.

In a PowerShell from the repo root:

```powershell
.\scripts\local\test_tiny.ps1 -EdfDir C:\path\to\a\few\converted\edfs
```

What it does:
- Activates the project `.venv`, pins to CPU
- Refuses to start if the folder contains no `.edf` files
- Runs 2 epochs × 3 segments/file × batch 8 — finishes in 5–15 min
  depending on your file sizes
- Saves a checkpoint to `.\bendr_test_output\`

Pass if: loss decreases between epoch 1 and epoch 2 and a checkpoint
appears in the output folder. If it errors, fix locally before
shipping anything to the cluster.

### 6b — LUNARC GPU smoke test on the same files

Upload those same 2-3 EDFs to COSMOS (the Step 3 commands work for
any subset — just point them at a folder containing only the test
files) and submit:

```bash
cd ~/NED-Net
sbatch scripts/lunarc/test_gpu_tiny.sh
```

`test_gpu_tiny.sh` is the same as `test_gpu.sh` but tuned for a
small dataset (workers=2, val_fraction=0, segments_per_file=3) — so
it doesn't crash trying to validate on zero files or leave 6 workers
idle.

You'll see: `Submitted batch job 1234567`.

### Check it's queued / running

```bash
jobinfo -u $USER
```

Status column:
- `PD` = pending (waiting for a free GPU)
- `R` = running
- `CG` = completing

### Watch the output live

```bash
tail -f logs/bendr_tiny_1234567.out
```

(Replace `1234567` with your job ID. `Ctrl+C` to stop watching.)

### What to look for

1. `EDF files found:` line shows your 2-3 files — confirms the
   path/upload landed correctly
2. `PyTorch X.X.X, CUDA: True, Device: NVIDIA A100` — GPU detected
3. `Epoch 1/2` and `Epoch 2/2` — training actually runs
4. Checkpoint file in `$HOME/bendr_output/tiny_run/`

If both 6a (local) and 6b (LUNARC) pass, you've validated:
ssh + sbatch + queue, conda env, GPU module, EDF reader, data path,
checkpointing. Now go back to **Step 3** and rsync the full dataset.

### If something went wrong

Check the error log:

```bash
cat logs/bendr_tiny_1234567.err
```

Common issues:
- `ModuleNotFoundError: No module named 'pyedflib'` → conda env not
  activated. Verify Step 4 completed and `conda activate bendr`
  appears in the script.
- `No EDF files found` → `EDF_DIR` doesn't match what's on disk.
  Confirm with `ls /lunarc/nobackup/projects/lu2026-2-60/edf_data/`
- `CUDA not available` → module load / conda env didn't pick up the
  GPU runtime. Check `module list` in the log shows `Anaconda3`.

---

## Step 7: Run the full-scale test (30 minutes)

Once the full dataset is uploaded, repeat the same kind of test but
on the real data. This catches issues that only show up at scale
(slow first-epoch start due to file enumeration, memory pressure,
etc.) before you commit to the multi-day training run.

```bash
cd ~/NED-Net
sbatch scripts/lunarc/test_gpu.sh
```

You'll see something like: `Submitted batch job 1234567`

Monitor and debug the same way as Step 6 (`jobinfo -u $USER`, `tail -f
logs/bendr_test_<JOBID>.out`, `cat logs/bendr_test_<JOBID>.err`).

What's different at scale:
- `Found XX EDF files` should now report the full ~thousands count
  instead of 2-3
- First-epoch start can be slow while the dataset enumerates and the
  workers warm up — give it a few minutes before assuming it's hung
- A 5% validation split is now meaningful (Step 6 had it disabled)

If this passes too, you're cleared for the multi-day runs.

---

## Step 7.5: Generate the bad-channels map

The real training runs (Steps 8–9) exclude known-noisy channels on a
**per recording batch** basis — e.g. batch 1 drops electrode 1, batch 2
drops electrodes 6 and 8, batch 3 drops none. The trainer reads this from
a `bad_channels.json` file, which `scripts/make_bad_channels.py` generates
from the `batch N` folders in your uploaded data.

Run it once, on the **login node** (no GPU or conda needed — it's pure
standard-library Python and only reads directory paths, not EDF contents):

```bash
cd ~/NED-Net
python scripts/make_bad_channels.py \
  --data-dir /lunarc/nobackup/projects/lu2026-2-60/edf_data
```

Expected output (counts will match your data):

```
Scanned 1888 EDF(s) under /lunarc/nobackup/projects/lu2026-2-60/edf_data
  batch 1:   655 file(s)  exclude indices [0]
  batch 2:   670 file(s)  exclude indices [5, 7]
  batch 3:   563 file(s)  exclude indices none

Wrote 1325 file entr(ies) ... to:
  /lunarc/nobackup/projects/lu2026-2-60/bad_channels.json
```

By default it writes `bad_channels.json` to `<edf_data>/../`, which is
exactly the path the `pretrain_short.sh` / `pretrain.sh` / `resume.sh`
scripts pass to `--bad-channels`. Nothing else to wire up.

Sanity-check it landed:

```bash
head -20 /lunarc/nobackup/projects/lu2026-2-60/bad_channels.json
```

> **Why this isn't optional.** The real-run scripts pass `--bad-channels`
> unconditionally, so they **hard-fail at startup** if this file is
> missing — that's deliberate: better a fast, loud failure than silently
> training on the noisy channels you meant to drop. The smoke tests
> (Steps 6–7) do *not* use the map, so they run fine before you generate
> it.

The generator itself also hard-fails (non-zero exit, nothing written) if:

- the **same EDF filename appears under two different batches** (the map
  keys on filename, so this would be ambiguous), or
- an **EDF sits outside any `batch N` folder** (no rule to apply).

If you hit either, the data layout needs a look before training — the
`batch N` folders must be preserved from the transfer (don't flatten).

### Adjusting for future batches or different bad channels

All the exclusion logic lives in **one dict** at the top of
`scripts/make_bad_channels.py`:

```python
BATCH_EXCLUDE = {
    1: [0],      # electrode 1
    2: [5, 7],   # electrodes 6 and 8
    3: [],       # none
}
```

Keys are batch numbers; values are **0-based channel indices** to drop.

> **Electrode label vs. array index.** The lab numbers electrodes
> **1-based** (1–8); the model indexes channels **0-based** (0–7). So
> electrode *N* → index *N−1*. Electrode 1 → `0`, electrode 6 → `5`,
> electrode 8 → `7`. Edit the dict in **index** space.

To adjust:

1. **New batch arrives** (e.g. the next 24-animal cohort = batch 4): add a
   line, e.g. `4: [2],` for "drop electrode 3 in batch 4", or `4: [],` if
   it's clean. **You must add a line for every batch** — a batch folder
   with no entry in `BATCH_EXCLUDE` is left **untouched (all channels)**
   with a warning to stderr, *not* treated as "exclude none". This is on
   purpose so a forgotten batch is visible in the log rather than silently
   mis-handled.
2. **Bad channels differ from what's wired** (your channel check found
   something else): change the index list for that batch.
3. **Re-run the same command** above. It overwrites `bad_channels.json`
   in place; the training scripts already point at it, so no other edits
   are needed. Re-run it any time the data set or the rules change.

---

## Step 8: Short pre-training run (5 epochs)

Instead of jumping straight into a 7-day training job, we do a short run
first. This uses ALL the data but only trains for 5 epochs (~1-2 days).

Why? Two reasons:
1. You can check that the loss is going down (the model is learning)
2. You get an early model to fine-tune in NED-Net and see if it's useful

No work is wasted — the full run later **resumes from this checkpoint**.

```bash
cd ~/NED-Net
sbatch scripts/lunarc/pretrain_short.sh
```

### Monitor it:

```bash
# Check status
jobinfo -u $USER

# Watch output (replace job ID with yours)
tail -f logs/bendr_5ep_XXXXXXX.out

# Check how much of your allocation you've used
projinfo
```

### What to look for:

- The **loss should decrease** each epoch (e.g., 8.5 → 6.2 → 5.1 → ...)
- The **accuracy should increase** (e.g., 0.01 → 0.05 → 0.12 → ...)

If the loss is NOT decreasing after 3 epochs, something is wrong — let me
know and we'll debug.

### When it finishes:

You'll get an email. Download the early model (see Step 10) and try
fine-tuning it in NED-Net. If the results look promising, continue to
Step 9.

---

## Step 9: Continue to full training (30 epochs)

Once you're happy with the 5-epoch model, continue training to 30 epochs.
The resume script picks up exactly where the short run stopped:

```bash
cd ~/NED-Net
sbatch scripts/lunarc/resume.sh
```

This has a **7-day time limit**. If it doesn't finish all 30 epochs in
7 days, just submit it again — it resumes from the latest checkpoint
every time:

```bash
# If it timed out, just run it again:
sbatch scripts/lunarc/resume.sh
```

**Pro tip:** You can chain jobs so the resume starts automatically after
the first one finishes:

```bash
JOBID=$(sbatch --parsable scripts/lunarc/resume.sh)
sbatch -d afterok:$JOBID scripts/lunarc/resume.sh
```

### Alternative: run all 30 epochs from scratch

If you prefer to skip the short run and go straight to 30 epochs:

```bash
sbatch scripts/lunarc/pretrain.sh
```

This is fine too — it saves checkpoints every 5 epochs, so if the job
is stopped you can always resume.

---

## Step 10: Download the trained model

When training finishes (you'll get an email), the best model is at:

```
~/bendr_output/run1/best_model.pt
```

Download it to the machine where you run NED-Net.

### From macOS / Linux

```bash
mkdir -p ~/.eeg_seizure_analyzer/pretrained

scp YOUR_USERNAME@cosmos.lunarc.lu.se:~/bendr_output/run1/best_model.pt \
  ~/.eeg_seizure_analyzer/pretrained/bendr_rodent_25k.pt

# Optional: training log for reference
scp YOUR_USERNAME@cosmos.lunarc.lu.se:~/bendr_output/run1/pretrain_log.json \
  ~/.eeg_seizure_analyzer/pretrained/
```

### From Windows

```powershell
# Create the destination folder
mkdir $env:USERPROFILE\.eeg_seizure_analyzer\pretrained -Force

# Download the model
scp YOUR_USERNAME@cosmos.lunarc.lu.se:~/bendr_output/run1/best_model.pt `
  $env:USERPROFILE\.eeg_seizure_analyzer\pretrained\bendr_rodent_25k.pt

# Optional: training log
scp YOUR_USERNAME@cosmos.lunarc.lu.se:~/bendr_output/run1/pretrain_log.json `
  $env:USERPROFILE\.eeg_seizure_analyzer\pretrained\
```

(WinSCP works too — drag `best_model.pt` from the COSMOS pane to
`%USERPROFILE%\.eeg_seizure_analyzer\pretrained\`.)

---

## Step 11: Use the pre-trained model in NED-Net

What the cluster produced is a **self-supervised BENDR backbone**
(`bendr_rodent_25k.pt`, now in `~/.eeg_seizure_analyzer/pretrained/`).
That is **not a seizure detector on its own** — it's the pre-trained
starting point you **fine-tune** on labelled seizures inside NED-Net.

`Detection`, `Training`, `Dataset / Model`, and `Analysis` are all
**separate top-level tabs** (not nested). The workflow is an **iterative
loop** — each pass makes the detector better:

1. **Load** — open your EDF(s) and fill in the **Animal ID** for each
   channel (needed for a clean train/validation split).

2. **Detection → Seizure** — generate candidate events to review.
   - *First pass (no model yet):* pick a **rule-based** method
     (Spike-Train, Spectral Band, …) and click **Detect Seizures**.
   - *Later passes:* pick your **trained model** — method **BENDR** *or*
     **U-Net** — to propose far better candidates than rule-based. This is
     the engine of the loop: a better model means fewer corrections next
     round.

   These are candidates for review, not final results.

3. **Training → Seizure** — the **refinement / annotation** step. Step
   through each candidate and **Confirm** (real) or **Reject** (false
   positive), adjusting onset/offset where the detector was off.
   Annotations auto-save next to each EDF as `*_ned_annotations.json`
   (confirmed → positives, rejected → hard negatives).

4. **Dataset / Model → Dataset** — **Scan** the folder of annotated EDFs,
   choose an **Architecture**, and click **Start Training**. The trained
   detector is saved under `~/.eeg_seizure_analyzer/models/<name>/`.
   - **BENDR:** in **Pre-trained weights** choose `bendr_rodent_25k` (it
     appears because Step 10 placed it in `…/pretrained/`); set the
     fine-tune params (epochs, freeze-encoder epochs, encoder LR, …). This
     fine-tunes the self-supervised backbone on your labels.
   - **U-Net:** no pre-trained weights — it trains from scratch on your
     labels.

5. **Repeat 2 → 4.** Run the step-4 model in Detection (step 2) to surface
   better candidates, correct them in Training (step 3), and retrain
   (step 4). Each loop adds labels and improves the detector.

6. **Analysis** — once the detector is good enough, run the **final**
   detection: choose **Seizures**, select your model, set the threshold,
   and run in **Single**, **Batch** (a folder of recordings), or **Live**
   mode. Results are written to the database.

7. **Results** — review aggregated detections (daily burden, circadian
   distribution) and export CSV.

> **To keep improving the model, do you load your trained model as the
> Pre-trained weights?** **No.** The **Pre-trained weights** dropdown lists
> only the self-supervised backbones in `~/.eeg_seizure_analyzer/pretrained/`;
> your *trained* models live in `…/models/` and are **not** offered there
> (and U-Net has no pre-trained-weights field at all). You improve a model
> by **retraining on the expanded annotation set** (select old + new
> annotation files): BENDR re-fine-tunes from the `bendr_rodent_25k`
> backbone, U-Net retrains from scratch. Retraining on more data each round
> — rather than continuing the previous supervised weights — is deliberate:
> it avoids forgetting / drift and takes only minutes at these dataset
> sizes.

---

## Quick reference: useful commands on COSMOS

| What you want to do | Command |
|---------------------|---------|
| Check your jobs | `jobinfo -u $USER` |
| Cancel a job | `scancel JOBID` |
| Check disk space | `snicquota` |
| Check project allocation | `projinfo` |
| Watch job output live | `tail -f logs/bendr_pretrain_JOBID.out` |
| See what's in a directory | `ls -la /path/to/dir` |
| Check job estimated start time | `squeue --start -j JOBID` |
| See all available GPUs | `sinfo -p gpua100` |

---

## Troubleshooting

### "My job is stuck in PD (pending) for hours"

The GPU nodes are shared with other users. Check estimated start:

```bash
squeue --start -j YOUR_JOBID
```

If the wait is very long, it might be because all A100 nodes are busy.
This is normal — the job will start when a node frees up.

### "My job failed immediately"

Check the error log:

```bash
cat logs/bendr_pretrain_JOBID.err
```

### "I ran out of disk space"

```bash
snicquota
```

If your home is full, move large files to project storage:

```bash
mv ~/large_file /lunarc/nobackup/projects/lu2026-2-60/
```

### "I forgot my project account name"

```bash
projinfo
```

### "I want to check if my data transferred correctly"

```bash
ls /lunarc/nobackup/projects/lu2026-2-60/edf_data/ | head -20
ls /lunarc/nobackup/projects/lu2026-2-60/edf_data/ | wc -l
```

The first command shows the first 20 files. The second counts total files.
