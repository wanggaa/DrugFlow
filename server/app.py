import os
import uuid
import subprocess
import shutil
import time
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import psutil
import json
from typing import Dict, Optional, Any
import datetime
import httpx

import argparse
import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import SDMolSupplier, MolToSmiles

def convert_sdf_to_smiles(input_sdf: Path, output_smi: Path, 
                         remove_hydrogens: bool = True,
                         canonicalize: bool = True,
                         isomeric_smiles: bool = True) -> None:
    """
    Convert SDF file to SMILES file.
    
    Args:
        input_sdf: Path to input SDF file
        output_smi: Path to output SMILES file
        remove_hydrogens: Whether to remove explicit hydrogens
        canonicalize: Whether to canonicalize SMILES
        isomeric_smiles: Whether to include stereochemistry in SMILES
    """
    if not input_sdf.exists():
        raise FileNotFoundError(f"Input SDF file not found: {input_sdf}")
    
    # Read molecules from SDF
    supplier = SDMolSupplier(str(input_sdf))
    
    smiles_list = []
    valid_count = 0
    error_count = 0
    
    lines = []
    
    for i, mol in enumerate(supplier):
        if mol is None:
            error_count += 1
            print(f"Warning: Failed to read molecule {i+1}", file=sys.stderr)
            continue
            
        try:
            # Process molecule
            if remove_hydrogens:
                mol = Chem.RemoveHs(mol)
            
            # Generate SMILES
            smiles = MolToSmiles(mol, 
                                 canonical=canonicalize,
                                 isomericSmiles=isomeric_smiles)
            
            smiles_list.append(smiles)
            
            line = smiles + ' ' + f'gen{i:06d}'
            lines.append(line)
            valid_count += 1
            
        except Exception as e:
            error_count += 1
            print(f"Warning: Failed to process molecule {i+1}: {e}", file=sys.stderr)
            continue
    
    # Write SMILES to file
    with open(output_smi, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    
    # Print summary
    print(f"Conversion complete:")
    print(f"  Input SDF: {input_sdf}")
    print(f"  Output SMILES: {output_smi}")
    print(f"  Total molecules processed: {valid_count + error_count}")
    print(f"  Successfully converted: {valid_count}")
    print(f"  Failed to process: {error_count}")


app = FastAPI(title="DrugFlow API", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应该限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头信息
)

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


async def send_callback(callback_config: Dict[str, Any], task_id: str, status: str, message: str = ""):
    """发送回调请求到指定URL，支持用户提供的格式"""
    if not callback_config:
        print(f"[CALLBACK ERROR] 回调配置为空: task_id={task_id}")
        return
    
    # 用户提供的格式: {"callback_config": {"callback_url": "...", "callback_method": "...", "callback_headers": {...}}}
    # 检查是否有callback_config键
    if "callback_config" not in callback_config:
        print(f"[CALLBACK ERROR] 回调配置格式错误，缺少callback_config键: task_id={task_id}, config={callback_config}")
        return
    
    config = callback_config["callback_config"]
    url = config.get("callback_url")
    if not url:
        print(f"[CALLBACK ERROR] 回调配置中没有callback_url: task_id={task_id}, config={config}")
        return
    
    # 只支持POST方法，直接使用POST
    headers = config.get("callback_headers", {"Content-Type": "application/json"})
    
    # 构建请求负载，task_id字段改为taskId
    payload = {
        "taskId": task_id,  # 接收端要求使用taskId字段
        "status": status,
        "message": message,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    try:
        # 使用默认超时30秒
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            print(f"[CALLBACK SUCCESS] 回调发送成功: task_id={task_id}, status={status}, url={url}")
    except Exception as e:
        print(f"[CALLBACK ERROR] 回调发送失败: task_id={task_id}, url={url}, error={str(e)}")


async def monitor_process(task_id: str, process: subprocess.Popen, output_dir: Path, callback_config: Optional[Dict[str, Any]] = None):
    """监控子进程并在完成后发送回调"""
    try:
        # 等待进程结束
        return_code = await asyncio.to_thread(process.wait)
        
        # 获取任务状态
        samples_file = output_dir / "samples.sdf"
        if samples_file.exists():
            status = "completed"
            message = "任务运行完成"
        else:
            status = "failed"
            message = f"进程退出，但输出文件不存在。返回码: {return_code}"
        
        # 更新任务状态
        if task_id in task_processes:
            task_processes[task_id]["status"] = status
            task_processes[task_id]["return_code"] = return_code
        
        # 发送回调
        await send_callback(callback_config, task_id, status, message)
        print(f"[PROCESS MONITOR] 进程监控结束: task_id={task_id}, status={status}, return_code={return_code}")
        
    except Exception as e:
        print(f"[PROCESS MONITOR ERROR] 进程监控异常: task_id={task_id}, error={str(e)}")
        # 即使监控出错也尝试发送回调
        await send_callback(callback_config, task_id, "failed", f"监控异常: {str(e)}")


def run_generate_process(task_id: str, pdb_file: Path, sdf_file: Path, output_dir: Path, n_samples: int = 10, callback_config=None):
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
        "--batch_size", "128"
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
        "n_samples": n_samples,
        "callback_config": callback_config,
        "monitor_task": None
    }
    
    # 启动后台监控任务
    monitor_task = asyncio.create_task(
        monitor_process(task_id, process, output_dir, callback_config)
    )
    task_processes[task_id]["monitor_task"] = monitor_task
    
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
    n_samples: int = Form(default=10, description="生成的分子样本数量"),
    callback_config: Optional[str] = Form(default=None, description="回调配置 JSON 字符串，格式: {\"callback_config\": {\"callback_url\": \"...\", \"callback_method\": \"POST\", \"callback_headers\": {...}}}")
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
    
    # 解析回调配置
    callback_config_dict = None
    if callback_config:
        try:
            callback_config_dict = json.loads(callback_config)
            # 验证必要的字段格式
            if not isinstance(callback_config_dict, dict) or "callback_config" not in callback_config_dict:
                raise HTTPException(status_code=400, detail="callback_config 必须包含 'callback_config' 键")
            
            inner_config = callback_config_dict["callback_config"]
            if not isinstance(inner_config, dict) or "callback_url" not in inner_config:
                raise HTTPException(status_code=400, detail="callback_config.callback_config 必须包含 'callback_url' 字段")
        except json.JSONDecodeError as e:
            print(f"[API ERROR] 回调配置 JSON 解析错误: {str(e)}")
            raise HTTPException(status_code=400, detail=f"回调配置 JSON 格式错误: {str(e)}")
        except Exception as e:
            print(f"[API ERROR] 回调配置验证错误: {str(e)}")
            raise HTTPException(status_code=400, detail=f"回调配置错误: {str(e)}")
    
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
        process = run_generate_process(task_id, pdb_file, sdf_file, output_dir, n_samples, callback_config_dict)
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

@app.get("/download/{task_id}/samples_sdf")
async def download_samples_sdf(task_id: str):
    """下载生成的samples.sdf文件"""
    print(f"[API CALL] GET /download/{task_id}/samples_sdf - 下载sdf生成结果")
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

@app.get("/download/{task_id}/samples_smi")
async def download_samples_smi(task_id: str):
    """下载生成的samples.smi文件"""
    print(f"[API CALL] GET /download/{task_id}/samples_smi - 下载smi生成结果")
    task_dir = TASKS_DIR / task_id
    samples_sdf_file = task_dir / "output" / "samples.sdf"
    samples_smi_file = task_dir / "output" / "samples.smi"
    
    
    if not task_dir.exists():
        print(f"[API ERROR] 任务不存在: {task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if not samples_sdf_file.exists():
        print(f"[API ERROR] 生成结果不存在: {task_id}")
        raise HTTPException(status_code=404, detail="生成结果尚未完成或不存在")
    
    print(f"[FILE TRANSFER] 转换samples.sdf: task_id={task_id}")
    convert_sdf_to_smiles(samples_sdf_file,samples_smi_file)
    
    print(f"[FILE DOWNLOAD] 下载samples.smi: task_id={task_id}")
    return FileResponse(
        path=samples_smi_file,
        filename=f"{task_id}_samples.smi",
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

@app.get("/status")
async def get_service_status():
    """获取后端服务运行状态"""
    
    # 检查关键文件是否存在
    checkpoint_exists = CHECKPOINT_PATH.exists()
    tasks_dir_exists = TASKS_DIR.exists()
    
    # 如果关键文件和目录都存在，则认为服务正常
    service_ok = checkpoint_exists and tasks_dir_exists
    
    status_info = {
        "status": "running" if service_ok else "error",
        "message": "服务正常运行" if service_ok else "服务异常，请检查关键文件",
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    return status_info

@app.get("/")
async def root():
    """API根端点"""
    return {
        "message": "DrugFlow API",
        "version": "1.0.0",
        "endpoints": {
            "GET /status": "获取服务状态",
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
