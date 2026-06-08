<#
.SYNOPSIS
    Collect the RAM_GDNF_2025 EDFs into the BENDR staging tree, mirroring the
    SV2A_2024 layout:  edf_data_for_bendr\RAM_GDNF_2025\batch {1,2,3}\W{ww}-D{dd}\

.DESCRIPTION
    Source recordings live under
        RAM_GDNF_2025\Batch_{1,2,3}_Recordings\Week_{1,2,3}\<varied EDF folder>\W{w}_D{d}\
    where the EDF-holding folder is inconsistently named ("EDF and Dat Files",
    "EDF and dat Files", "B3_W1_edf and dat files", nested "EDF and Videos" /
    "New folder" duplicate copies, ...). Rather than trust the folder layout, we
    route each EDF by its OWN filename, which always encodes batch/week/day:

        B{b}_W{w}_D{d}_DDMMYYYY(N).edf      e.g. B3_W3_D8_07022026(12).edf

    Each file goes to:
        edf_data_for_bendr\RAM_GDNF_2025\batch {b}\W{ww}-D{dd}\<same name>.edf
    (week and day zero-padded to 2 digits, taken verbatim from the filename —
    so B3_W3's non-contiguous "D8" lands in W03-D08 with no arithmetic surprise).

    Only *.edf are copied (no .dat / .xlsx / .mat / _ned_annotations.json side
    files) — matching the existing SV2A_2024 BENDR tree, which is EDF-only. The
    RAM annotations are left in place; collect them separately if ever needed for
    supervised fine-tuning.

    EXCLUDED:
      - anything under a "Test" folder (2 un-suffixed sample EDFs in
        Batch_1_Recordings\Test) — not real recordings.
    DE-DUPLICATED:
      - the 16 EDFs under nested "EDF and Videos" / "New folder" copies share
        their basename with the real file in the same day, so they route to the
        same target name. robocopy's native skip-identical (same size+timestamp)
        drops them. If two sources map to one target name with DIFFERENT sizes,
        that's a real conflict — the script flags it (CHECK) and skips the
        second rather than overwrite.

    Copies use robocopy (/J unbuffered for big EEG files; skip-identical =>
    idempotent/resumable; /R:2 /W:5 retries). DRY-RUN by default; -Execute to act.

.PARAMETER Base
    Root that holds RAM_GDNF_2025 and edf_data_for_bendr. Defaults to the Z:
    LU-share path. On the lab PC the share is mounted as R: — pass
    -Base "R:\LU26D1055-epicenter\Data\KAHA recordings" there.

.EXAMPLE
    pwsh -File collect_ram_gdnf_edf_byday.ps1            # preview
    pwsh -File collect_ram_gdnf_edf_byday.ps1 -Execute   # do it
#>
[CmdletBinding()]
param(
    [string]$Base = "Z:\LU26D1055-epicenter\Data\KAHA recordings",
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'

$SrcRoot = Join-Path $Base "RAM_GDNF_2025"
$Dest    = Join-Path $Base "edf_data_for_bendr\RAM_GDNF_2025"
$mode    = if ($Execute) { "EXECUTE" } else { "DRY-RUN" }

Write-Host "==================================================================="
Write-Host " RAM_GDNF_2025 EDF collection -> BENDR per-day tree   [$mode]"
Write-Host " Source:      $SrcRoot"
Write-Host " Destination: $Dest"
Write-Host "==================================================================="

if (-not (Test-Path -LiteralPath $SrcRoot)) { throw "Source not found: $SrcRoot" }

# Filename router: B{b}_W{w}_D{d}_<date>(N).edf  ->  batch {b}, W{ww}-D{dd}
$nameRe = '^B(?<b>[0-9]+)_W(?<w>[0-9]+)_D(?<d>[0-9]+)_'
$rcFlags = @('/J', '/R:2', '/W:5', '/NP', '/NDL', '/NJH', '/NJS')

# ---- build the copy plan ---------------------------------------------------
$batchDirs = Get-ChildItem -LiteralPath $SrcRoot -Directory |
    Where-Object Name -match '^Batch_[123]_Recordings$'

$plan = New-Object System.Collections.Generic.List[object]
$skippedTest = 0
$unparseable = @()

foreach ($bd in $batchDirs) {
    $edfs = Get-ChildItem -LiteralPath $bd.FullName -Recurse -File -Filter *.edf
    foreach ($f in $edfs) {
        # drop the Test-folder strays
        if ($f.FullName -split '[\\/]' | Where-Object { $_ -ieq 'Test' }) { $skippedTest++; continue }
        $m = [regex]::Match($f.Name, $nameRe, 'IgnoreCase')
        if (-not $m.Success) { $unparseable += $f.FullName; continue }
        $b  = [int]$m.Groups['b'].Value
        $w  = [int]$m.Groups['w'].Value
        $d  = [int]$m.Groups['d'].Value
        $day = 'W{0:D2}-D{1:D2}' -f $w, $d
        $dst = Join-Path (Join-Path $Dest ("batch {0}" -f $b)) $day
        $plan.Add([pscustomobject]@{ File = $f; Dst = $dst; Name = $f.Name; Batch = $b; Day = $day })
    }
}

if ($unparseable.Count) {
    Write-Warning "$($unparseable.Count) EDF(s) did not match B#_W#_D#_ and were skipped:"
    $unparseable | Select-Object -First 10 | ForEach-Object { Write-Warning "    $_" }
}

# ---- conflict detection: same target name, different size ------------------
# (byte-identical dups are fine — robocopy skips them; only differing sizes
#  under one name are a real ambiguity we must not silently overwrite)
$conflictKeys = @{}
foreach ($g in $plan | Group-Object { Join-Path $_.Dst $_.Name }) {
    $sizes = @($g.Group | ForEach-Object { $_.File.Length } | Sort-Object -Unique)
    if ($sizes.Count -gt 1) {
        $conflictKeys[$g.Name] = $true
        Write-Warning "CONFLICT (different sizes, will copy first only): $($g.Name)"
    }
}

# ---- summary of the plan ---------------------------------------------------
Write-Host ("`nPlanned: {0} EDF -> {1} unique target file(s) across {2} batch(es). Skipped {3} Test stray(s)." -f `
    $plan.Count, (@($plan | Group-Object { Join-Path $_.Dst $_.Name } ).Count), `
    (@($plan | Group-Object Batch).Count), $skippedTest)
foreach ($bg in $plan | Group-Object Batch | Sort-Object Name) {
    $days = @($bg.Group | Group-Object Day)
    Write-Host ("  batch {0}: {1} edf, {2} day-folders ({3} .. {4})" -f `
        $bg.Name, $bg.Group.Count, $days.Count, `
        (($days.Name | Sort-Object)[0]), (($days.Name | Sort-Object)[-1]))
}

# ---- execute ---------------------------------------------------------------
if ($Execute) {
    Write-Host "`n--- copying ---"
    $copied = 0; $failed = 0; $deduped = @{}
    # group by (sourceDir, targetDir) so robocopy gets a filelist per pair
    $groups = $plan | Group-Object { "$($_.File.DirectoryName)|$($_.Dst)" }
    foreach ($g in $groups) {
        $srcDir = $g.Group[0].File.DirectoryName
        $dstDir = $g.Group[0].Dst
        # within this src/dst pair, drop names we've already placed once and
        # names flagged as size-conflicts after their first placement
        $names = foreach ($row in $g.Group) {
            $key = Join-Path $row.Dst $row.Name
            if ($deduped.ContainsKey($key)) { continue }   # already queued/placed
            $deduped[$key] = $true
            $row.Name
        }
        $names = @($names)
        if (-not $names.Count) { continue }
        if (-not (Test-Path -LiteralPath $dstDir)) { New-Item -ItemType Directory -Path $dstDir -Force | Out-Null }
        & robocopy.exe @($srcDir, $dstDir) @names @rcFlags | Out-Null
        if ($LASTEXITCODE -ge 8) { $failed += $names.Count; Write-Warning "robocopy exit ${LASTEXITCODE}: $srcDir -> $dstDir" }
        else { $copied += $names.Count }
    }
    Write-Host ("  queued {0} unique target file(s); robocopy errors on {1}." -f $copied, $failed)
}

# ---- verify ----------------------------------------------------------------
Write-Host "`n=== verification ==="
foreach ($b in 1, 2, 3) {
    $bDst = Join-Path $Dest ("batch {0}" -f $b)
    $expected = @($plan | Where-Object Batch -eq $b | Group-Object { Join-Path $_.Dst $_.Name }).Count
    if (Test-Path -LiteralPath $bDst) {
        $have = @(Get-ChildItem -LiteralPath $bDst -Recurse -File -Filter *.edf)
        $dayFolders = @(Get-ChildItem -LiteralPath $bDst -Directory)
        $loose = @(Get-ChildItem -LiteralPath $bDst -File -Filter *.edf).Count
        $deep  = @($dayFolders | ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Directory }).Count
        $gb = [math]::Round(($have | Measure-Object Length -Sum).Sum / 1GB, 2)
        $ok = ($have.Count -eq $expected) -and ($loose -eq 0) -and ($deep -eq 0)
        Write-Host ("  [{0}] batch {1}: {2}/{3} edf, {4} day-folders, {5} GB (loose={6}, nested-dirs={7})" -f `
            $(if ($ok) { 'OK' } else { 'CHECK' }), $b, $have.Count, $expected, $dayFolders.Count, $gb, $loose, $deep)
    } else {
        Write-Host ("  [--] batch {0}: (dry-run, not created) expected {1} unique edf" -f $b, $expected)
    }
}
Write-Host "`nDone [$mode]."
if (-not $Execute) { Write-Host "Re-run with  -Execute  to perform the copy." }
