$ErrorActionPreference = "Stop"

$remoteHost = "root@connect.bjb1.seetacloud.com"
$remotePort = 50402
$remoteBase = "/root/autodl-tmp/zero-arm"

$timeTag = Get-Date -Format "yyyyMMdd_HHmmss"
$localBase = "F:/zero-robotic-arm-master/5. Deep_LR/remote_artifacts/queue128_$timeTag"

New-Item -ItemType Directory -Force -Path "$localBase/models" | Out-Null
New-Item -ItemType Directory -Force -Path "$localBase/logs" | Out-Null

Write-Host "Watcher started. Local target: $localBase"

while ($true) {
    $status = & ssh -p $remotePort $remoteHost @"
test -f $remoteBase/models/td3/.done_128 && echo td3_done || true
test -f $remoteBase/models/ppo/.done_128 && echo ppo_done || true
test -f $remoteBase/models/sac/.done_128 && echo sac_done || true
"@

    $hasTd3 = $status -match "td3_done"
    $hasPpo = $status -match "ppo_done"
    $hasSac = $status -match "sac_done"

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] td3=$hasTd3 ppo=$hasPpo sac=$hasSac"

    if ($hasTd3 -and $hasPpo -and $hasSac) {
        Write-Host "All trainings finished. Pulling artifacts..."

        & scp -P $remotePort -r "${remoteHost}:$remoteBase/models/td3" "$localBase/models/"
        & scp -P $remotePort -r "${remoteHost}:$remoteBase/models/ppo" "$localBase/models/"
        & scp -P $remotePort -r "${remoteHost}:$remoteBase/models/sac" "$localBase/models/"
        & scp -P $remotePort -r "${remoteHost}:$remoteBase/logs/td3" "$localBase/logs/"
        & scp -P $remotePort -r "${remoteHost}:$remoteBase/logs/ppo" "$localBase/logs/"
        & scp -P $remotePort -r "${remoteHost}:$remoteBase/logs/sac" "$localBase/logs/"
        & scp -P $remotePort "${remoteHost}:$remoteBase/train_queue.log" "$localBase/"

        Write-Host "Artifacts pulled to: $localBase"
        break
    }

    Start-Sleep -Seconds 120
}
