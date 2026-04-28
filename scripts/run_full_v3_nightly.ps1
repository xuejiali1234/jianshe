param(
  [string]$Python = "D:/Anaconda/envs/traffic_pred/python.exe",
  [string]$Device = "cuda",
  [int]$DqnTimesteps = 10000,
  [int]$DqnRounds = 5
)

$ErrorActionPreference = "Stop"

function Run-Step {
  param(
    [string]$Name,
    [scriptblock]$Command
  )
  Write-Host ""
  Write-Host "===== $Name ====="
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed: $Name"
  }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$rawCsv = "data/raw/batch_movement_aggregates_full_v3.csv"
$scenarioDir = "data/raw/scenarios_full_v3"
$manifest = "$scenarioDir/manifest.csv"
$datasetDir = "data/datasets/full_v3_movement_control"
$artifactDir = "models/artifacts_full_v3"
$reportDir = "reports/full_v3_training"
$rlReportDir = "reports/rl_signal_control/full_v3_pred_control"
$rlArtifactDir = "models/artifacts_rl/full_v3_pred_control"

Run-Step "Full V3 batch simulation" {
  & $Python -m sim.scripts.run_batch_sumo `
    --overwrite `
    --scenario-preset full `
    --sim-end 3600 `
    --collector movement `
    --output-csv $rawCsv `
    --scenario-dir $scenarioDir
}

Run-Step "Full V3 prediction training and config update" {
  & $Python -m prediction.training `
    --csv $rawCsv `
    --dataset-dir $datasetDir `
    --artifact-dir $artifactDir `
    --report-dir $reportDir `
    --scenario-manifest $manifest `
    --smoke-test `
    --update-config
}

Run-Step "Webster evaluation" {
  & $Python -m rl.evaluate_policy `
    --config configs/rl_signal_config.json `
    --policy webster `
    --sim-end 1800 `
    --out "$rlReportDir/webster_1800_eval.csv"
}

Run-Step "MaxPressure evaluation" {
  & $Python -m rl.evaluate_policy `
    --config configs/rl_signal_config.json `
    --policy max_pressure `
    --sim-end 1800 `
    --out "$rlReportDir/max_pressure_1800_eval.csv"
}

Run-Step "DQN no-pred training" {
  & $Python -m rl.train_dqn `
    --config configs/rl_signal_config.json `
    --timesteps $DqnTimesteps `
    --sim-end 1800 `
    --use-prediction false `
    --device $Device `
    --out-dir $rlArtifactDir `
    --report-dir $rlReportDir `
    --run-name no_pred
}

Run-Step "DQN no-pred evaluation" {
  & $Python -m rl.evaluate_policy `
    --config configs/rl_signal_config.json `
    --policy dqn `
    --model-path "$rlArtifactDir/dqn_signal_single_tls_no_pred.zip" `
    --sim-end 1800 `
    --use-prediction false `
    --out "$rlReportDir/dqn_no_pred_eval.csv"
}

Run-Step "DQN pred-v2 sweep" {
  & $Python -m rl.optimize_pred_v2 `
    --config configs/rl_signal_config.json `
    --rounds $DqnRounds `
    --timesteps $DqnTimesteps `
    --sim-end 1800 `
    --device $Device `
    --report-dir $rlReportDir `
    --artifact-dir $rlArtifactDir
}

Run-Step "Final RL summary" {
  & $Python -m rl.summarize_results `
    --report-dir $rlReportDir `
    --window 100
}

Write-Host ""
Write-Host "Full V3 nightly pipeline finished."
Write-Host "Prediction metrics: $reportDir/metrics.csv"
Write-Host "RL summary: $rlReportDir/sweep_summary.csv"
