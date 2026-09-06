#Requires -RunAsAdministrator

[CmdletBinding()]
param(
    [string]$ListenAddress = "100.122.105.65",
    [string]$AllowedRemoteAddress = "100.74.87.113",
    [string]$OrinAddress = "192.168.0.162",
    [string]$NezhaAddress = "192.168.0.163"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$mappings = @(
    [pscustomobject]@{
        Name = "T800 Orin SSH via Tailscale"
        ListenPort = 22162
        ConnectAddress = $OrinAddress
    },
    [pscustomobject]@{
        Name = "T800 Nezha SSH via Tailscale"
        ListenPort = 22163
        ConnectAddress = $NezhaAddress
    }
)

$localAddress = Get-NetIPAddress -AddressFamily IPv4 -IPAddress $ListenAddress `
    -ErrorAction SilentlyContinue
if (-not $localAddress) {
    throw "Listen address $ListenAddress is not assigned locally. Check Tailscale before changing portproxy."
}

Set-Service -Name iphlpsvc -StartupType Automatic
Start-Service -Name iphlpsvc

foreach ($mapping in $mappings) {
    & netsh interface portproxy delete v4tov4 `
        "listenaddress=$ListenAddress" `
        "listenport=$($mapping.ListenPort)" `
        protocol=tcp | Out-Null

    & netsh interface portproxy add v4tov4 `
        "listenaddress=$ListenAddress" `
        "listenport=$($mapping.ListenPort)" `
        "connectaddress=$($mapping.ConnectAddress)" `
        connectport=22 `
        protocol=tcp | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create portproxy listener $($mapping.ListenPort)."
    }

    # Remove every duplicate rule with this exact display name, then add one.
    Get-NetFirewallRule -DisplayName $mapping.Name -ErrorAction SilentlyContinue |
        Remove-NetFirewallRule

    New-NetFirewallRule `
        -DisplayName $mapping.Name `
        -Direction Inbound `
        -Action Allow `
        -Protocol TCP `
        -LocalAddress $ListenAddress `
        -LocalPort $mapping.ListenPort `
        -RemoteAddress $AllowedRemoteAddress `
        -Profile Any | Out-Null
}

# Portproxy rules can exist in the registry while no TCP socket is bound. Force
# IP Helper to reload the repaired rules before checking the actual listeners.
Restart-Service -Name iphlpsvc
Start-Sleep -Seconds 2

$listeners = @(
    Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object {
            $_.LocalAddress -eq $ListenAddress -and
            $_.LocalPort -in $mappings.ListenPort
        }
)

$missingPorts = @(
    $mappings.ListenPort | Where-Object { $_ -notin $listeners.LocalPort }
)

Write-Host "`nConfigured portproxy rules:"
& netsh interface portproxy show v4tov4

Write-Host "`nActive TCP listeners:"
$listeners |
    Sort-Object LocalPort |
    Format-Table LocalAddress, LocalPort, OwningProcess

Write-Host "`nRobot-side SSH reachability from Windows:"
foreach ($mapping in $mappings) {
    $reachable = Test-NetConnection $mapping.ConnectAddress -Port 22 `
        -InformationLevel Quiet
    [pscustomobject]@{
        Target = $mapping.ConnectAddress
        Port = 22
        Reachable = $reachable
    }
}

if ($missingPorts.Count -ne 0) {
    throw "No TCP listener for port(s): $($missingPorts -join ', '). Inspect iphlpsvc and the Tailscale IPv4 assignment."
}

Write-Host "`nBoth Windows listeners are active. Test SSH from $AllowedRemoteAddress;"
Write-Host "a local Test-NetConnection to $ListenAddress is not the acceptance test"
Write-Host "because the firewall intentionally allows only the remote Tailscale source."
