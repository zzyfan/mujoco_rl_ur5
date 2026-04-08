$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = "F:\minconda\envs\rl-mujoco-env\python.exe"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$queueRoot = Join-Path $repoRoot "runs\local\main\_queue\$timestamp"
New-Item -ItemType Directory -Force -Path $queueRoot | Out-Null

if (-not (Test-Path $pythonExe)) {
    throw "python executable not found: $pythonExe"
}

# 单窗口顺序训练：
# - 直接调用 rl-mujoco-env 里的 python.exe，避免 conda 包裹层吞掉或延迟进度条。
# - 三个算法按 td3 -> sac -> ppo 顺序跑，窗口里会依次显示各自的 tqdm 进度条。
# - 同时把标准输出写到每个算法自己的日志文件，方便事后复盘。
$commonArgs = @(
    "--total-timesteps", "5000000",
    "--n-envs", "32",
    "--device", "cuda",
    "--eval-freq", "1000000",
    "--eval-episodes", "1"
)

$jobs = @(
    @{ Algo = "td3"; RunName = "local_queue_td3_32env_cuda" },
    @{ Algo = "sac"; RunName = "local_queue_sac_32env_cuda" },
    @{ Algo = "ppo"; RunName = "local_queue_ppo_32env_cuda" }
)

$summaryPath = Join-Path $queueRoot "summary.log"
"queue_root=$queueRoot" | Set-Content -Path $summaryPath -Encoding UTF8
"python_exe=$pythonExe" | Add-Content -Path $summaryPath

Push-Location $repoRoot
try {
    foreach ($job in $jobs) {
        $algo = $job.Algo
        $runName = $job.RunName
        $logPath = Join-Path $queueRoot "$algo.log"
        New-Item -ItemType File -Force -Path $logPath | Out-Null

        $banner = ("=" * 18) + " start algo={0} run_name={1} time={2} " -f $algo, $runName, (Get-Date -Format "s") + ("=" * 18)
        Write-Host $banner -ForegroundColor Cyan
        Add-Content -Path $summaryPath -Value ("start algo={0} run_name={1} time={2}" -f $algo, $runName, (Get-Date -Format "s"))

        $arguments = @(
            "train_ur5_reach.py",
            "--algo", $algo,
            "--run-name", $runName
        ) + $commonArgs

        & $pythonExe @arguments 2>&1 | Tee-Object -FilePath $logPath -Append
        if ($LASTEXITCODE -ne 0) {
            Add-Content -Path $summaryPath -Value ("fail algo={0} exit_code={1} time={2}" -f $algo, $LASTEXITCODE, (Get-Date -Format "s"))
            throw "training failed for algo=$algo exit_code=$LASTEXITCODE"
        }

        Add-Content -Path $summaryPath -Value ("done algo={0} time={1}" -f $algo, (Get-Date -Format "s"))
        Write-Host ("completed algo={0} time={1}" -f $algo, (Get-Date -Format "s")) -ForegroundColor Green
    }

    Add-Content -Path $summaryPath -Value ("queue_complete time={0}" -f (Get-Date -Format "s"))
    Write-Host "queue_complete" -ForegroundColor Green
}
finally {
    Pop-Location
}
