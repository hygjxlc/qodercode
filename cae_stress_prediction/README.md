# CAE Stress Prediction MLP

基于 OpenSpec v1 生成的神经网络模型，用于 CAE 结构应力预测。

## 模型信息

- **名称**: cae_stress_prediction_mlp
- **版本**: 1.0.0
- **领域**: CAE / 结构力学 / 仿真代理模型
- **输入**: 6 维特征 [长度, 宽度, 厚度, 弹性模量, 泊松比, 载荷]
- **输出**: 最大等效应力 (MPa)

## 项目结构

```
cae_stress_prediction/
├── model.py              # 神经网络模型定义
├── train.py              # 训练脚本
├── predict.py            # 预测脚本 (CLI)
├── api_server.py         # HTTP API 服务器
├── requirements.txt      # Python 依赖
├── openspec.yaml         # OpenSpec 配置文件
└── README.md            # 说明文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python train.py
```

训练完成后会生成：
- `best_model.pth` - 最佳模型权重
- `training_history.json` - 训练历史记录

### 3. 使用 CLI 预测

```bash
# 创建输入文件
echo '{"features": [100.0, 50.0, 5.0, 200.0, 0.3, 5000.0]}' > input.json

# 运行预测
python predict.py --input input.json --output result.json

# 查看结果
cat result.json
```

### 4. 启动 API 服务器

```bash
python api_server.py
```

访问：
- API 文档: http://localhost:5000
- 模型信息: http://localhost:5000/api/model/openspec
- 健康检查: http://localhost:5000/api/health

#### API 预测示例

```bash
curl -X POST http://localhost:5000/api/predict/stress \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [100.0, 50.0, 5.0, 200.0, 0.3, 5000.0],
      [150.0, 80.0, 8.0, 180.0, 0.25, 8000.0]
    ]
  }'
```

## 输入特征说明

| 索引 | 特征 | 单位 | 说明 |
|------|------|------|------|
| 0 | 长度 | mm | 结构长度 |
| 1 | 宽度 | mm | 结构宽度 |
| 2 | 厚度 | mm | 结构厚度 |
| 3 | 弹性模量 | GPa | 材料弹性模量 |
| 4 | 泊松比 | - | 材料泊松比 (0-0.5) |
| 5 | 载荷大小 | N | 施加的载荷 |

## 模型架构

```
Input (6) → Dense(256, SiLU) → Dropout(0.1) → 
Dense(128, SiLU) → Dense(64, SiLU) → Output(1)
```

## 训练配置

- **优化器**: Adam (lr=0.0005, weight_decay=1e-5)
- **损失函数**: SmoothL1Loss
- **评估指标**: MAE, MSE, RMSE, R²
- **批次大小**: 64
- **最大轮数**: 200

## 接口定义

### HTTP API

- `GET /api/model/openspec` - 获取模型配置
- `POST /api/predict/stress` - 应力预测
- `GET /api/health` - 健康检查

### CLI

```bash
python predict.py --input input.json --output result.json
```

## 许可证

MIT License
