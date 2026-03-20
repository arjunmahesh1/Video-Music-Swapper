param(
    [Parameter(Mandatory = $true)]
    [string]$Candidate,

    [string]$MusicStem
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$original = Join-Path $repoRoot "video\Gatorade.mp4"
$transcript = Join-Path $PSScriptRoot "gatorade_transcript.txt"

if (-not (Test-Path $original)) {
    throw "Missing original video: $original"
}

if (-not (Test-Path $Candidate)) {
    throw "Missing candidate output: $Candidate"
}

if (-not (Test-Path $transcript)) {
    throw "Missing transcript file: $transcript"
}

$cmd = @(
    "voiceover_benchmark.py",
    "--original", $original,
    "--candidate", $Candidate,
    "--transcript", $transcript
)

if ($MusicStem) {
    if (-not (Test-Path $MusicStem)) {
        throw "Missing music stem: $MusicStem"
    }
    $cmd += @("--music-stem", $MusicStem)
}

Write-Host "Running benchmark..."
Write-Host "  original:   $original"
Write-Host "  candidate:  $Candidate"
Write-Host "  transcript: $transcript"
if ($MusicStem) {
    Write-Host "  music stem: $MusicStem"
}

python @cmd
