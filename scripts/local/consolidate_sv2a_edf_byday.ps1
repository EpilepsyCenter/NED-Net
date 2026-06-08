<#
.SYNOPSIS
    Consolidate all EDF files from the three SV2A chronic-KA KAHA recordings into
    SV2A_2024\batch {1,2,3}\Week{W}-Day{DD} per-day folders.

.DESCRIPTION
    Layout (per user): keep three batch folders; inside each, one folder per
    recording day named  Week{W}-Day{DD}  (day zero-padded to 2 digits;
    Week = ceil(day/7)). Multi-day recording sessions keep all their days in one
    folder, e.g.  Week1-Day05_06 ,  Week6-Day40_41_42 .

    Batch 1  (20231009 ... recording 1)
        - Its 655 EDFs are ALREADY in SV2A_2024 under B1_W{w}\Day_{tokens}.
          Each Day_ folder is MOVED (instant, same volume) to
          batch 1\Week{w}-Day{dd}; emptied B1_W* folders are removed.
        - 5 extra single-day "...Sel.edf" files (in WK-1\Hamza\...) are COPIED in,
          each routed to the day-folder that already covers its day
          (e.g. Day 5 -> Week1-Day05_06).
    Batch 2  (20240115 ... recording 2)  -> 670 EDFs COPIED from "EDF files\Week_*"
    Batch 3  (20240304 ... recording 3)  -> 563 EDFs COPIED from "EDF files\Week_*"
        Day is parsed from each filename (Day1-.. / D1-.. / Day5+6-..).

    No EDF basename collides within a batch; every filename parses to a day.
    No multi-day session crosses a week boundary (verified).

    Copies use robocopy (skip-if-identical => idempotent/resumable; /J unbuffered;
    /R:2 /W:5 retries). DRY-RUN by default; pass -Execute to act.

.EXAMPLE
    pwsh -File consolidate_sv2a_edf_byday.ps1            # preview
    pwsh -File consolidate_sv2a_edf_byday.ps1 -Execute   # do it
#>
[CmdletBinding()]
param(
    [string]$Base = "Z:\LU26D1055-epicenter\Data\KAHA recordings",
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'

$Dest = Join-Path $Base "SV2A_2024"
$Src = [ordered]@{
    "batch 1" = Join-Path $Base "20231009 SV2A Chronic KA Bl6  recording 1"
    "batch 2" = Join-Path $Base "20240115 Ling Chronic KABL6-recording 2"
    "batch 3" = Join-Path $Base "20240304 Ling Chronic KABL6-recording 3"
}
$mode = if ($Execute) { "EXECUTE" } else { "DRY-RUN" }
Write-Host "==================================================================="
Write-Host " SV2A EDF consolidation -> per-day folders   [$mode]"
Write-Host " Destination: $Dest"
Write-Host "==================================================================="

# --- helpers ---------------------------------------------------------------
function Get-DayFolderName {
    param([int[]]$Days)
    $s = @($Days | Sort-Object -Unique)
    $w1 = [int][math]::Ceiling($s[0] / 7.0)
    $w2 = [int][math]::Ceiling($s[-1] / 7.0)
    if ($w1 -ne $w2) { Write-Warning "Day span $($s -join ',') crosses weeks ($w1..$w2); using $w1" }
    $dd = ($s | ForEach-Object { '{0:D2}' -f $_ }) -join '_'
    "Week$w1-Day$dd"
}
function Get-DaysFromToken {
    param([string]$Token)   # e.g. "5_6", "5+6", "40-42", "8"
    [regex]::Matches($Token, '\d+') | ForEach-Object { [int]$_.Value }
}
function Get-DaysFromFileName {
    param([string]$Name)    # "Day5+6-20240119(1).edf" / "D18+19-..." / "Day1-..."
    if ($Name -match '(?i)^Day\s*([0-9+\-\s]+?)\s*-\s*\d{8}') { return Get-DaysFromToken $matches[1] }
    if ($Name -match '(?i)^D\s*([0-9+\-\s]+?)\s*-\s*\d{8}')    { return Get-DaysFromToken $matches[1] }
    return @()
}

$rcFlags = @('/J', '/R:2', '/W:5', '/NP', '/NDL', '/NJH', '/NJS')

# Copy a set of {File, Dst} plan rows via robocopy, grouped by (sourceDir,Dst).
function Invoke-CopyPlan {
    param([object[]]$Plan, [string]$Tag)
    $copied = 0; $failed = 0
    $groups = $Plan | Group-Object { "$($_.File.DirectoryName)|$($_.Dst)" }
    foreach ($g in $groups) {
        $srcDir = $g.Group[0].File.DirectoryName
        $dstDir = $g.Group[0].Dst
        $names = $g.Group | ForEach-Object { $_.File.Name }
        if (-not (Test-Path -LiteralPath $dstDir)) { New-Item -ItemType Directory -Path $dstDir -Force | Out-Null }
        & robocopy.exe @($srcDir, $dstDir) @names @rcFlags | Out-Null
        if ($LASTEXITCODE -ge 8) { $failed += $names.Count; Write-Warning "[$Tag] robocopy exit ${LASTEXITCODE}: $srcDir -> $dstDir" }
        else { $copied += $names.Count }
    }
    [pscustomobject]@{ Copied = $copied; Failed = $failed }
}

# ===========================================================================
# BATCH 1 : move existing B1_W{w}\Day_* folders -> batch 1\Week{w}-Day{dd}
# ===========================================================================
$b1Dst = Join-Path $Dest "batch 1"
Write-Host "`n--- batch 1 (reorganize in place) ---"

$weekDirs = Get-ChildItem -LiteralPath $Dest -Directory | Where-Object Name -like 'B1_W*' | Sort-Object Name
$dayCoverage = @{}   # day(int) -> target folder name (for routing the 5 stragglers)
$movePlan = @()      # rows: SourceFolder, TargetPath, TargetName
foreach ($w in $weekDirs) {
    foreach ($d in Get-ChildItem -LiteralPath $w.FullName -Directory) {
        $days = @(Get-DaysFromToken ($d.Name -replace '(?i)^Day_?', ''))
        if (-not $days) { Write-Warning "  cannot parse days from folder '$($d.Name)'"; continue }
        $tname = Get-DayFolderName $days
        foreach ($dn in $days) { $dayCoverage[$dn] = $tname }
        $movePlan += [pscustomobject]@{ Source = $d.FullName; Target = (Join-Path $b1Dst $tname); Name = $tname; Count = (Get-ChildItem -LiteralPath $d.FullName -File -Filter *.edf).Count }
    }
}
Write-Host ("  {0} existing day-folders -> batch 1\Week*-Day* ({1} edf)" -f $movePlan.Count, (($movePlan | Measure-Object Count -Sum).Sum))
$movePlan | Sort-Object Name | Select-Object -First 6 | ForEach-Object { Write-Host ("      {0}  ({1} edf)" -f $_.Name, $_.Count) }
Write-Host "      ..."
if ($Execute) {
    if (-not (Test-Path -LiteralPath $b1Dst)) { New-Item -ItemType Directory -Path $b1Dst | Out-Null }
    foreach ($m in $movePlan) {
        if (Test-Path -LiteralPath $m.Target) {
            # target exists (re-run): move files in, then drop empty source
            Get-ChildItem -LiteralPath $m.Source -File | ForEach-Object { Move-Item -LiteralPath $_.FullName -Destination $m.Target -Force }
            if (-not (Get-ChildItem -LiteralPath $m.Source -Recurse -File)) { Remove-Item -LiteralPath $m.Source -Recurse -Force }
        } else {
            Move-Item -LiteralPath $m.Source -Destination $m.Target
        }
    }
    foreach ($w in $weekDirs) {
        if (-not (Get-ChildItem -LiteralPath $w.FullName -Recurse -File -ErrorAction SilentlyContinue)) {
            Remove-Item -LiteralPath $w.FullName -Recurse -Force; Write-Host "  removed empty $($w.Name)"
        } else { Write-Warning "  $($w.Name) not empty; kept" }
    }
}

# 5 straggler "...Sel.edf" files in source 1 -> covering day folder (copy)
$placedNames = @()
if (Test-Path -LiteralPath $b1Dst) { $placedNames = @(Get-ChildItem -LiteralPath $b1Dst -Recurse -File -Filter *.edf | Select-Object -ExpandProperty Name) }
$stragglers = Get-ChildItem -LiteralPath $Src["batch 1"] -Recurse -File -Filter *.edf | Where-Object { $_.Name -notin $placedNames -and ($_.Name -notin ($movePlan | ForEach-Object { $null })) }
# Robust "not already placed" test: name not among files currently/after-move in batch 1.
$alreadyAll = @($placedNames) + @(Get-ChildItem -LiteralPath $Dest -Recurse -File -Filter *.edf | Select-Object -ExpandProperty Name)
$stragglers = Get-ChildItem -LiteralPath $Src["batch 1"] -Recurse -File -Filter *.edf | Where-Object { $_.Name -notin $alreadyAll } | Where-Object { $_.FullName -like '*\WK-*' }
$b1CopyPlan = foreach ($f in $stragglers) {
    $days = @(Get-DaysFromFileName $f.Name)
    if (-not $days) { Write-Warning "  straggler unparseable: $($f.Name)"; continue }
    $tname = if ($dayCoverage.ContainsKey($days[0])) { $dayCoverage[$days[0]] } else { Get-DayFolderName $days }
    [pscustomobject]@{ File = $f; Dst = (Join-Path $b1Dst $tname); Name = $tname }
}
$b1CopyPlan = @($b1CopyPlan)
Write-Host ("  {0} straggler file(s) to copy:" -f $b1CopyPlan.Count)
$b1CopyPlan | ForEach-Object { Write-Host ("      {0,-26} -> {1}" -f $_.File.Name, $_.Name) }
if ($Execute -and $b1CopyPlan.Count) {
    $r = Invoke-CopyPlan -Plan $b1CopyPlan -Tag 'b1-sel'
    Write-Host ("      copied {0}, failed {1}" -f $r.Copied, $r.Failed)
}

# ===========================================================================
# BATCH 2 & 3 : copy from "EDF files\Week_*" -> batch N\Week{w}-Day{dd}
# ===========================================================================
foreach ($name in @("batch 2", "batch 3")) {
    Write-Host "`n--- $name (copy from source) ---"
    $bDst = Join-Path $Dest $name
    $edfRoot = Join-Path $Src[$name] "EDF files"
    $files = @(Get-ChildItem -LiteralPath $edfRoot -Recurse -File -Filter *.edf)
    $plan = foreach ($f in $files) {
        $days = @(Get-DaysFromFileName $f.Name)
        if (-not $days) { Write-Warning "  unparseable: $($f.Name)"; continue }
        [pscustomobject]@{ File = $f; Dst = (Join-Path $bDst (Get-DayFolderName $days)); Name = (Get-DayFolderName $days) }
    }
    $plan = @($plan)
    $folders = $plan | Group-Object Name | Sort-Object { [int](($_.Name -replace '.*Day','') -split '_')[0] }
    Write-Host ("  {0} edf -> {1} day-folders" -f $plan.Count, $folders.Count)
    $folders | Select-Object -First 4 | ForEach-Object { Write-Host ("      {0}  ({1} edf)" -f $_.Name, $_.Count) }
    Write-Host "      ..."
    $folders | Select-Object -Last 2 | ForEach-Object { Write-Host ("      {0}  ({1} edf)" -f $_.Name, $_.Count) }
    if ($Execute) {
        if (-not (Test-Path -LiteralPath $bDst)) { New-Item -ItemType Directory -Path $bDst | Out-Null }
        $r = Invoke-CopyPlan -Plan $plan -Tag $name
        Write-Host ("  -> copied {0}, failed {1}" -f $r.Copied, $r.Failed)
    }
}

# ===========================================================================
# VERIFY
# ===========================================================================
Write-Host "`n=== verification ==="
foreach ($name in $Src.Keys) {
    $bDst = Join-Path $Dest $name
    $expected = if ($name -eq 'batch 1') { (Get-ChildItem -LiteralPath $Src[$name] -Recurse -File -Filter *.edf).Count }
    else { (Get-ChildItem -LiteralPath (Join-Path $Src[$name] 'EDF files') -Recurse -File -Filter *.edf).Count }
    if (Test-Path -LiteralPath $bDst) {
        $have = @(Get-ChildItem -LiteralPath $bDst -Recurse -File -Filter *.edf)
        $dayFolders = @(Get-ChildItem -LiteralPath $bDst -Directory)
        # files must sit exactly one level deep (inside a WeekX-DayX folder), none loose
        $loose = @(Get-ChildItem -LiteralPath $bDst -File -Filter *.edf).Count
        $deep = @($dayFolders | ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Directory }).Count
        $gb = [math]::Round(($have | Measure-Object Length -Sum).Sum / 1GB, 2)
        $ok = ($have.Count -eq $expected) -and ($loose -eq 0) -and ($deep -eq 0)
        Write-Host ("  [{0}] {1}: {2}/{3} edf, {4} day-folders, {5} GB (loose={6}, deeper-nested-dirs={7})" -f `
            $(if ($ok) { 'OK' } else { 'CHECK' }), $name, $have.Count, $expected, $dayFolders.Count, $gb, $loose, $deep)
    } else {
        Write-Host ("  [--] {0}: (dry-run, not created) expected {1} edf" -f $name, $expected)
    }
}
Write-Host "`nDone [$mode]."
if (-not $Execute) { Write-Host "Re-run with  -Execute  to perform the move/copy." }
