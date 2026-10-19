# Windows PowerShell launcher, same as run_train.sh. Run from the segmentation\ folder:
#   .\train\run_train.ps1 profile      # 2-epoch throughput check; must pass before training
#   .\train\run_train.ps1 2            # train seed 2 (the shipped checkpoint); also run 0 and 1
param([string]$Mode = "2", [Parameter(ValueFromRemainingArguments = $true)] $Rest)
$Root = Split-Path -Parent $PSScriptRoot
if (-not $env:CORAL_DATA_V2) { $env:CORAL_DATA_V2 = Join-Path $Root "data\coralscapes_v2" }
if (-not $env:CORAL_V2_CACHE) { $env:CORAL_V2_CACHE = Join-Path $Root "cache_v2" }
$env:PYTHONIOENCODING = "utf-8"
$Labels = if ($env:LABELS) { $env:LABELS } else { Join-Path $Root "data\labels" }
$Py = if ($env:PY) { $env:PY } else { "python" }
$Workers = if ($env:WORKERS) { $env:WORKERS } else { "8" }

$TrainArgs = @("--classes", "7", "--total-steps", "17880", "--eval-every", "447", "--persist-every-eval",
               "--rare-crop-frac", "0.5", "--val-split", "test", "--workers", $Workers)
foreach ($l in "kenya", "red_sea", "chagos", "la_paz", "norfolk", "raja") { $TrainArgs += @("--extra-train", (Join-Path $Labels "sonnet7_$l")) }
foreach ($l in "kenya", "red_sea", "chagos", "la_paz", "norfolk") { $TrainArgs += @("--val-dir", (Join-Path $Labels "sonnet7_${l}_heldout")) }

$Out = Join-Path $Root "runs"
New-Item -ItemType Directory -Force $Out | Out-Null
$Script = Join-Path $Root "train\standalone_train.py"
if ($Mode -eq "profile") {
    & $Py $Script --profile 2 --out $Out @TrainArgs @Rest
} else {
    & $Py $Script --seed $Mode --tag "P3_rare_E2_all_cs_s$Mode" --out $Out @TrainArgs @Rest
}
exit $LASTEXITCODE
