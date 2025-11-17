# DrugFlow API Server

基于FastAPI的DrugFlow分子生成后端服务。

## 功能特性

- 接收PDB和SDF文件进行分子生成
- 异步任务处理
- 实时状态查询
- 结果文件下载
- 运行日志捕获

## API接口

### 1. 提交生成任务
**POST** `/generate`
- 参数：
  - `protein`: PDB格式的蛋白质文件
  - `ligand`: SDF格式的配体文件
- 返回：
  ```json
  {
    "status": "submitted",
    "task_id": "uuid",
    "message": "任务已提交，正在处理中"
  }
  ```

### 2. 查询任务状态
**GET** `/task_stat/{task_id}`
- 返回状态：
  - `running`: 正在运行中
  - `completed`: 运行完成
  - `failed`: 运行失败

### 3. 下载生成结果
**GET** `/download/{task_id}/samples`
- 下载生成的samples.sdf文件

### 4. 下载日志文件
**GET** `/download/{task_id}/log`
- 下载生成过程的日志文件

## 安装依赖

```bash
pip install -r server/requirements.txt
```

## 启动服务

### 方式一：使用模块启动（推荐）

```bash
python -m server.app --ip 0.0.0.0 --port 8000
```

### 方式二：直接运行脚本

```bash
python server/app.py --ip 0.0.0.0 --port 8000
```

### 方式三：使用uvicorn启动

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

## 访问API文档

启动服务后，访问以下地址查看交互式API文档：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 文件夹结构

```
tasks/
├── {task_id}/
│   ├── input/
│   │   ├── protein.pdb
│   │   └── ligand.sdf
│   └── output/
│       ├── samples.sdf (生成结果)
│       └── generate.log (运行日志)
```

## 注意事项

1. 确保项目根目录下存在 `checkpoints/drugflow.ckpt` 模型文件
2. 确保Python环境已安装项目所需的所有依赖
3. 生成过程可能需要较长时间，具体取决于硬件配置
4. 任务状态通过检查文件存在性和进程状态来判断
