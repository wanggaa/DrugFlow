import os
import uuid
import subprocess
import shutil
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
import psutil
import json
from typing import Dict

app = FastAPI(title="DrugFlow API", version="1.0.0")

# 基础路径配置
BASE_DIR = Path(__file__).parent.parent  # 指向项目根目录
SERVER_DIR = Path(__file__).parent  # server目录
TASKS_DIR = BASE_DIR / "tasks"
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "drugflow.ckpt"

# 确保任务目录存在
TASKS_DIR.mkdir(exist_ok=True)

# 存储任务进程信息的字典
task_processes: Dict[str, Dict] = {}

def create_task_folder(task_id: str) -> Path:
    """创建任务文件夹结构"""
    task_dir = TASKS_DIR / task_id
    input_dir = task_dir / "input"
    output_dir = task_dir / "output"
    
    task_dir.mkdir(exist_ok=True)
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    return task_dir

def save_uploaded_file(file: UploadFile, destination: Path):
    """保存上传的文件"""
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

def run_generate_process(task_id: str, pdb_file: Path, sdf_file: Path, output_dir: Path, n_samples: int = 10):
    """运行generate.py进程并捕获输出"""
    log_file = output_dir / "generate.log"
    
    # 构建命令
    cmd = [
        "python", "src/generate.py",
        "--protein", str(pdb_file),
        "--ref_ligand", str(sdf_file),
        "--checkpoint", str(CHECKPOINT_PATH),
        "--output", str(output_dir / "samples.sdf"),
        "--n_samples", str(n_samples),
        "--batch_size", "32"
    ]
    
    # 启动进程并重定向输出到日志文件
    process = subprocess.Popen(
        cmd,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        cwd=BASE_DIR
    )
    
    # 存储进程信息
    task_processes[task_id] = {
        "process": process,
        "pid": process.pid,
        "status": "running",
        "n_samples": n_samples
    }
    
    return process

def get_task_status(task_id: str) -> str:
    """获取任务状态"""
    task_dir = TASKS_DIR / task_id
    output_dir = task_dir / "output"
    samples_file = output_dir / "samples.sdf"
    
    # 检查任务目录是否存在
    if not task_dir.exists():
        return "not_found"
    
    # 检查进程状态
    if task_id in task_processes:
        process_info = task_processes[task_id]
        process = process_info["process"]
        
        # 检查进程是否仍在运行
        if process.poll() is None:
            return "running"
        else:
            # 进程已结束，检查输出文件
            if samples_file.exists():
                task_processes[task_id]["status"] = "completed"
                return "completed"
            else:
                task_processes[task_id]["status"] = "failed"
                return "failed"
    else:
        # 没有进程信息，检查文件状态
        if samples_file.exists():
            return "completed"
        else:
            return "failed"

@app.post("/generate")
async def generate_molecules(
    protein: UploadFile = File(..., description="PDB格式的蛋白质文件"),
    ligand: UploadFile = File(..., description="SDF格式的配体文件"),
    n_samples: int = Form(default=10, description="生成的分子样本数量")
):
    """接收PDB和SDF文件，创建任务并启动生成过程"""
    print(f"[API CALL] POST /generate - 收到文件上传请求: protein={protein.filename}, ligand={ligand.filename}, n_samples={n_samples}")
    
    # 验证文件类型
    if not protein.filename.endswith('.pdb'):
        print(f"[API ERROR] 蛋白质文件格式错误: {protein.filename}")
        raise HTTPException(status_code=400, detail="蛋白质文件必须是PDB格式")
    if not ligand.filename.endswith('.sdf'):
        print(f"[API ERROR] 配体文件格式错误: {ligand.filename}")
        raise HTTPException(status_code=400, detail="配体文件必须是SDF格式")
    
    # 验证n_samples参数
    if n_samples <= 0:
        print(f"[API ERROR] n_samples参数错误: {n_samples}")
        raise HTTPException(status_code=400, detail="n_samples必须大于0")
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    task_dir = create_task_folder(task_id)
    input_dir = task_dir / "input"
    output_dir = task_dir / "output"
    
    # 保存上传的文件
    pdb_file = input_dir / "protein.pdb"
    sdf_file = input_dir / "ligand.sdf"
    
    save_uploaded_file(protein, pdb_file)
    save_uploaded_file(ligand, sdf_file)
    
    # 启动生成进程
    try:
        process = run_generate_process(task_id, pdb_file, sdf_file, output_dir, n_samples)
        print(f"[TASK CREATED] 任务已创建: task_id={task_id}, 进程PID={process.pid}, n_samples={n_samples}")
        
        return {
            "status": "submitted",
            "task_id": task_id,
            "n_samples": n_samples,
            "message": "任务已提交，正在处理中"
        }
    except Exception as e:
        print(f"[TASK ERROR] 任务创建失败: task_id={task_id}, 错误={str(e)}")
        raise HTTPException(status_code=500, detail=f"启动生成过程失败: {str(e)}")

@app.get("/task_stat/{task_id}")
async def get_task_status_api(task_id: str):
    """获取任务状态"""
    print(f"[API CALL] GET /task_stat/{task_id} - 查询任务状态")
    status = get_task_status(task_id)
    
    if status == "not_found":
        print(f"[API ERROR] 任务不存在: {task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")
    
    status_messages = {
        "running": "正在运行中",
        "completed": "运行完成",
        "failed": "运行失败"
    }
    
    print(f"[TASK STATUS] 任务状态查询: task_id={task_id}, status={status}")
    return {
        "task_id": task_id,
        "status": status,
        "message": status_messages.get(status, "未知状态")
    }

@app.get("/download/{task_id}/samples")
async def download_samples(task_id: str):
    """下载生成的samples.sdf文件"""
    print(f"[API CALL] GET /download/{task_id}/samples - 下载生成结果")
    task_dir = TASKS_DIR / task_id
    samples_file = task_dir / "output" / "samples.sdf"
    
    if not task_dir.exists():
        print(f"[API ERROR] 任务不存在: {task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if not samples_file.exists():
        print(f"[API ERROR] 生成结果不存在: {task_id}")
        raise HTTPException(status_code=404, detail="生成结果尚未完成或不存在")
    
    print(f"[FILE DOWNLOAD] 下载samples.sdf: task_id={task_id}")
    return FileResponse(
        path=samples_file,
        filename=f"{task_id}_samples.sdf",
        media_type='chemical/x-mdl-sdfile'
    )

@app.get("/download/{task_id}/log")
async def download_log(task_id: str):
    """下载生成过程的日志文件"""
    print(f"[API CALL] GET /download/{task_id}/log - 下载日志文件")
    task_dir = TASKS_DIR / task_id
    log_file = task_dir / "output" / "generate.log"
    
    if not task_dir.exists():
        print(f"[API ERROR] 任务不存在: {task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if not log_file.exists():
        print(f"[API ERROR] 日志文件不存在: {task_id}")
        raise HTTPException(status_code=404, detail="日志文件不存在")
    
    print(f"[FILE DOWNLOAD] 下载generate.log: task_id={task_id}")
    return FileResponse(
        path=log_file,
        filename=f"{task_id}_generate.log",
        media_type='text/plain'
    )

@app.get("/")
async def root():
    """API根端点"""
    return {
        "message": "DrugFlow API",
        "version": "1.0.0",
        "endpoints": {
            "POST /generate": "提交生成任务（参数：protein, ligand, n_samples）",
            "GET /task_stat/{task_id}": "查询任务状态", 
            "GET /download/{task_id}/samples": "下载生成结果",
            "GET /download/{task_id}/log": "下载日志文件"
        }
    }

if __name__ == "__main__":
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="DrugFlow API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server IP address")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    args = parser.parse_args()
    
    print(f"Starting DrugFlow API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
