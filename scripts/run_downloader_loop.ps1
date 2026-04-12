$log = "E:\Large Animal DatasetF\download_log.txt"
$script = "E:\Animal classifier Aastha\A-Hybrid-AI-Approach-for-Predicting-and-Preventing-Wild-Animal-Movements-\scripts\download_animal_images.py"
$python = "C:\Users\Astaa\AppData\Local\Programs\Python\Python311\python.exe"
while ($true) {
    try {
        "`n--- $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') starting run ---" | Add-Content -Path $log
        & $python -u $script 2>&1 | Tee-Object -FilePath $log
    } catch {
        "`n--- $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') run crashed: $($_.Exception.Message) ---" | Add-Content -Path $log
    }
    Start-Sleep -Seconds 10
}
