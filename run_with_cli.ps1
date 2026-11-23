# Runs PaperOrchestra through a logged-in CLI, without an LLM API key.
# Default: Claude plan login, Sonnet 5.5, education profile, Sage template.
# Token counts go to runs\cli_usage.jsonl.
# Example:
#   .\run_with_cli.ps1 --raw_materials_dir .\materials --output_dir .\runs\deneme
param(
    [ValidateSet("claude", "codex")]
    [string]$Cli = "claude",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PipelineArgs
)

$env:PAPER_ORCHESTRA_CLI = $Cli
$env:PAPER_ORCHESTRA_PROFILE = "education"
$env:PYTHONUTF8 = "1"   # the pipeline opens files without an explicit encoding

if ($PipelineArgs -notcontains "--latex_template_dir") {
    $PipelineArgs += @("--latex_template_dir", "$PSScriptRoot\templates\sage_education")
}

& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\paper_writing_cli.py" @PipelineArgs
exit $LASTEXITCODE
