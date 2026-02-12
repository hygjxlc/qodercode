# 会务专家管理系统 - Windows 自动部署脚本
# 以管理员身份运行 PowerShell

param(
    [string]$MySQLRootPassword = "root123",
    [int]$BackendPort = 8080,
    [int]$FrontendPort = 3000
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  会务专家管理系统 - 自动部署脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查是否以管理员身份运行
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Warning "建议以管理员身份运行此脚本，以便安装软件"
}

# 函数：检查命令是否存在
function Test-Command($Command) {
    return [bool](Get-Command $Command -ErrorAction SilentlyContinue)
}

# 函数：下载文件
function Download-File($Url, $Output) {
    Write-Host "正在下载: $Url" -ForegroundColor Yellow
    Invoke-WebRequest -Uri $Url -OutFile $Output -UseBasicParsing
}

# 1. 检查并安装 MySQL
Write-Host "[1/7] 检查 MySQL..." -ForegroundColor Green
if (-not (Test-Command "mysql")) {
    Write-Host "MySQL 未安装，正在下载安装..." -ForegroundColor Yellow
    $mysqlInstaller = "$env:TEMP\mysql-installer.msi"
    Download-File "https://dev.mysql.com/get/Downloads/MySQLInstaller/mysql-installer-community-8.0.35.0.msi" $mysqlInstaller
    Write-Host "请手动运行安装程序: $mysqlInstaller" -ForegroundColor Red
    Write-Host "安装完成后重新运行此脚本" -ForegroundColor Red
    exit 1
} else {
    Write-Host "MySQL 已安装" -ForegroundColor Green
}

# 2. 检查并安装 Java
Write-Host "[2/7] 检查 Java..." -ForegroundColor Green
if (-not (Test-Command "java") -or (java -version 2>&1 | Select-String "version" | Select-String "17|18|19|20|21") -eq $null) {
    Write-Host "Java 17 未安装，正在下载..." -ForegroundColor Yellow
    $javaInstaller = "$env:TEMP\java17.msi"
    Download-File "https://download.oracle.com/java/17/latest/jdk-17_windows-x64_bin.msi" $javaInstaller
    Write-Host "正在安装 Java 17..." -ForegroundColor Yellow
    Start-Process -FilePath "msiexec.exe" -ArgumentList "/i", $javaInstaller, "/quiet", "/norestart" -Wait
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    Write-Host "Java 17 安装完成" -ForegroundColor Green
} else {
    Write-Host "Java 已安装" -ForegroundColor Green
}

# 3. 检查并安装 Maven
Write-Host "[3/7] 检查 Maven..." -ForegroundColor Green
if (-not (Test-Command "mvn")) {
    Write-Host "Maven 未安装，正在下载..." -ForegroundColor Yellow
    $mavenZip = "$env:TEMP\maven.zip"
    Download-File "https://dlcdn.apache.org/maven/maven-3/3.9.6/binaries/apache-maven-3.9.6-bin.zip" $mavenZip
    Expand-Archive -Path $mavenZip -DestinationPath "C:\apache-maven" -Force
    [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\apache-maven\apache-maven-3.9.6\bin", "Machine")
    $env:Path += ";C:\apache-maven\apache-maven-3.9.6\bin"
    Write-Host "Maven 安装完成" -ForegroundColor Green
} else {
    Write-Host "Maven 已安装" -ForegroundColor Green
}

# 4. 检查并安装 Node.js
Write-Host "[4/7] 检查 Node.js..." -ForegroundColor Green
if (-not (Test-Command "node")) {
    Write-Host "Node.js 未安装，正在下载..." -ForegroundColor Yellow
    $nodeInstaller = "$env:TEMP\node18.msi"
    Download-File "https://nodejs.org/dist/v18.19.0/node-v18.19.0-x64.msi" $nodeInstaller
    Write-Host "正在安装 Node.js..." -ForegroundColor Yellow
    Start-Process -FilePath "msiexec.exe" -ArgumentList "/i", $nodeInstaller, "/quiet", "/norestart" -Wait
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    Write-Host "Node.js 安装完成" -ForegroundColor Green
} else {
    Write-Host "Node.js 已安装" -ForegroundColor Green
}

# 5. 创建数据库
Write-Host "[5/7] 创建数据库..." -ForegroundColor Green
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$schemaPath = Join-Path $scriptPath "database\schema.sql"

try {
    # 尝试连接 MySQL 并创建数据库
    $mysqlCmd = "mysql -u root -p$MySQLRootPassword < `"$schemaPath`""
    Invoke-Expression $mysqlCmd 2>&1 | Out-Null
    Write-Host "数据库创建成功" -ForegroundColor Green
} catch {
    Write-Warning "数据库创建可能已存在或需要手动创建"
    Write-Host "请手动运行: mysql -u root -p < database\schema.sql" -ForegroundColor Yellow
}

# 6. 编译并启动后端
Write-Host "[6/7] 编译并启动后端..." -ForegroundColor Green
$backendPath = Join-Path $scriptPath "backend"
Set-Location $backendPath

Write-Host "正在编译后端..." -ForegroundColor Yellow
mvn clean package -DskipTests -q

if ($LASTEXITCODE -ne 0) {
    Write-Error "后端编译失败"
    exit 1
}

Write-Host "启动后端服务..." -ForegroundColor Yellow
$backendJar = Get-ChildItem "target\*.jar" | Select-Object -First 1
$backendJob = Start-Job -ScriptBlock {
    param($JarPath, $Port)
    java -jar $JarPath --server.port=$Port
} -ArgumentList $backendJar.FullName, $BackendPort

Start-Sleep -Seconds 10

# 检查后端是否启动成功
try {
    $response = Invoke-WebRequest -Uri "http://localhost:$BackendPort/actuator/health" -UseBasicParsing -TimeoutSec 5
    Write-Host "后端服务启动成功: http://localhost:$BackendPort" -ForegroundColor Green
} catch {
    Write-Host "后端服务可能正在启动中，请稍后检查" -ForegroundColor Yellow
}

# 7. 安装依赖并启动前端
Write-Host "[7/7] 安装依赖并启动前端..." -ForegroundColor Green
$frontendPath = Join-Path $scriptPath "frontend"
Set-Location $frontendPath

if (-not (Test-Path "node_modules")) {
    Write-Host "正在安装前端依赖..." -ForegroundColor Yellow
    npm install
}

Write-Host "启动前端服务..." -ForegroundColor Yellow
$frontendJob = Start-Job -ScriptBlock {
    param($Path, $Port)
    Set-Location $Path
    npm run serve -- --port=$Port
} -ArgumentList $frontendPath, $FrontendPort

Start-Sleep -Seconds 15

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  部署完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "访问地址:" -ForegroundColor Yellow
Write-Host "  前端页面: http://localhost:$FrontendPort" -ForegroundColor Green
Write-Host "  后端API:  http://localhost:$BackendPort" -ForegroundColor Green
Write-Host "  API文档:  http://localhost:$BackendPort/swagger-ui.html" -ForegroundColor Green
Write-Host ""
Write-Host "按 Ctrl+C 停止服务" -ForegroundColor Yellow
Write-Host ""

# 保持脚本运行
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    Write-Host "正在停止服务..." -ForegroundColor Yellow
    Stop-Job $backendJob -ErrorAction SilentlyContinue
    Stop-Job $frontendJob -ErrorAction SilentlyContinue
    Remove-Job $backendJob -ErrorAction SilentlyContinue
    Remove-Job $frontendJob -ErrorAction SilentlyContinue
}
