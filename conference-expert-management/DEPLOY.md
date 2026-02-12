# 会务专家管理系统 - 部署文档

## 系统要求

- Java 17+
- MySQL 8.0+
- Node.js 18+
- Maven 3.8+

## 一、数据库部署

### 1. 创建数据库
```bash
mysql -u root -p < database/schema.sql
```

### 2. 修改数据库配置
编辑 `backend/src/main/resources/application.yml`：
```yaml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/conference_expert_management?useSSL=false&serverTimezone=Asia/Shanghai
    username: your_username
    password: your_password
```

## 二、后端部署

### 1. 编译打包
```bash
cd backend
mvn clean package -DskipTests
```

### 2. 运行
```bash
# 开发模式
mvn spring-boot:run

# 生产模式
java -jar target/expert-management-1.0.0.jar
```

后端服务将在 http://localhost:8080 启动

### 3. API 文档
启动后访问：http://localhost:8080/swagger-ui.html

## 三、前端部署

### 1. 安装依赖
```bash
cd frontend
npm install
```

### 2. 开发模式运行
```bash
npm run serve
```
前端将在 http://localhost:3000 启动

### 3. 生产构建
```bash
npm run build
```
构建后的文件在 `dist/` 目录

### 4. 部署到 Nginx
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        root /path/to/frontend/dist;
        index index.html;
        try_files $uri $uri/ /index.html;
    }
    
    location /api {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 四、Docker 部署（可选）

### 1. 构建镜像
```bash
# 后端
cd backend
docker build -t expert-management-backend .

# 前端
cd frontend
docker build -t expert-management-frontend .
```

### 2. 运行容器
```bash
docker-compose up -d
```

## 五、系统访问

- 前端页面：http://localhost:3000
- 后端 API：http://localhost:8080
- API 文档：http://localhost:8080/swagger-ui.html

## 六、常见问题

### 1. 数据库连接失败
- 检查 MySQL 服务是否启动
- 检查数据库配置是否正确
- 确认数据库已创建

### 2. 跨域问题
- 后端已配置 CORS，允许所有来源
- 生产环境建议配置具体的允许域名

### 3. 前端代理失败
- 检查 `vue.config.js` 中的代理配置
- 确认后端服务已启动

## 七、联系方式

技术支持：support@example.com
