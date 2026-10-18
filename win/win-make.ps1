[CmdletBinding()]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]] $MakeArgs
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    if ($MakeArgs.Count -eq 0) {
        throw 'Usage: .\win\win-make.ps1 <make-target> [make-arguments]'
    }

    & docker compose up --detach --build book
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    & docker compose exec book make @MakeArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
