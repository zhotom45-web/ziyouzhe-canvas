$ErrorActionPreference = 'Stop'

$projectDirectory = Split-Path -Parent $PSScriptRoot
$packageFile = Join-Path $projectDirectory 'package.json'
$package = Get-Content -LiteralPath $packageFile -Raw -Encoding UTF8 | ConvertFrom-Json
$releaseDirectory = Join-Path $projectDirectory $package.build.directories.output
$executableName = $package.build.artifactName.Replace('${version}', $package.version).Replace('${ext}', 'exe')
$executablePath = Join-Path $releaseDirectory $executableName

if (-not (Test-Path -LiteralPath $executablePath -PathType Leaf)) {
  throw "Built application not found: $executablePath"
}

$desktopDirectory = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktopDirectory "$($package.build.productName).lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $executablePath
$shortcut.WorkingDirectory = $releaseDirectory
$shortcut.IconLocation = "$executablePath,0"
$shortcut.Description = "$($package.build.productName) v$($package.version)"
$shortcut.Save()

Write-Host "Desktop shortcut updated: $shortcutPath -> $executablePath"

$productName = [string]$package.build.productName
$oldProcesses = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
  $_.ProcessName -eq $productName -or $_.ProcessName -like "$productName-*"
})

if ($oldProcesses.Count -gt 0) {
  $oldProcessIds = @($oldProcesses | Select-Object -ExpandProperty Id)
  $oldProcesses | Stop-Process -Force -ErrorAction SilentlyContinue
  $deadline = [DateTime]::UtcNow.AddSeconds(10)
  do {
    Start-Sleep -Milliseconds 200
    $stillRunning = @($oldProcessIds | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
  } while ($stillRunning.Count -gt 0 -and [DateTime]::UtcNow -lt $deadline)
  Write-Host "Closed previous $productName processes: $($oldProcessIds -join ', ')"
}

Start-Process -FilePath $executablePath -WorkingDirectory $releaseDirectory
Write-Host "Started latest version: $executablePath"
