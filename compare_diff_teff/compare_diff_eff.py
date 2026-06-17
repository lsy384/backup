import os
import sys

# =======================================================
# 🚀 救命参数：破解并发死锁与线程爆炸 (必须放在所有的 import 之前)
# =======================================================
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # 禁用 NetCDF 底层文件锁 (解决卡死的核心)
os.environ["OMP_NUM_THREADS"] = "1"            # 限制 OpenMP 线程
os.environ["MKL_NUM_THREADS"] = "1"            # 限制 MKL 线程
os.environ["OPENBLAS_NUM_THREADS"] = "1"       # 限制 OpenBLAS 线程
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"     # 限制 Mac/Linux 向量计算线程
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import torch
import datetime
import numpy as np
import netCDF4 as nc
import time
import multiprocessing as mp
import concurrent.futures
from tqdm import tqdm

# Import the RTM model containing the added Teff schemes
from rtm import DifferentiableRTM

def worker_process(file_idx, file_path, total_files, output_dir):
    """
    单文件处理的独立 worker，自动分配 GPU 并重定向日志
    """
    # =======================================================
    # 🚀 在子进程内部强行限制 PyTorch 的 CPU 线程，把算力全部让给 GPU
    # =======================================================
    torch.set_num_threads(1)
    
    # 获取进程 ID 用于绑定 GPU 和日志文件
    current_process = mp.current_process()
    worker_id = current_process._identity[0] if current_process._identity else 1
    
    # 日志输出重定向
    log_dir = "log_files_compare"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"worker_{worker_id}.log")
    
    log_file = open(log_file_path, 'a', encoding='utf-8', buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = log_file
    sys.stderr = log_file

    try:
        # 动态分配 GPU (利用身份 ID 对 4 取模)
        # identity 通常从 1 开始，所以减 1 对 4 取模得到 0, 1, 2, 3
        device_id = (worker_id - 1) % 4
        device = torch.device(f"cuda:{device_id}")
        
        file_name = os.path.basename(file_path)
        out_file_name = file_name.replace('forward_inputs_', 'Teff_outputs_')
        out_file_path = os.path.join(output_dir, out_file_name)
        # ==========================================
        # 1. 检查文件是否已存在 (断点续传)
        # ==========================================
        if os.path.exists(out_file_path):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {out_file_name} already exists. Skipping...")
            # 直接返回 True，并附带跳过信息
            return True, file_path, f"Skipped: {out_file_name} exists"

        print(f"\n[{time.strftime('%H:%M:%S')}] [File {file_idx}/{total_files}] Loading {file_name} on GPU {device_id}...")
        t_file_start = time.time()
        
        # 初始化模型
        rtm_model = DifferentiableRTM().to(device)
        rtm_model.eval()

        # 物理与卫星常量设定 
        sat_fghz = 1.4 
        sat_theta = 40.0 * np.pi / 180.0
        f = sat_fghz * 1e9
        lam = rtm_model.C / f
        lamcm = lam * 100.0
        wtot = 0.0175 + 0.0276

        res_keys = [
            'r_H_wilheit', 'r_V_wilheit', 
            'T_eff_wilheit_H', 'T_eff_wilheit_V',
            'T_eff_lv_multi', 'T_eff_lv_two', 
            'T_eff_wigneron', 'T_eff_holmes2006', 'T_eff_wigneron2008',
            'depth_90_lv_multi'
        ]

        # 恢复老代码的大吞吐分块策略
        time_chunk_size = 8784  
        batch_size = 15000

        # ==========================================
        # 2. 读取文件与准备数据
        # ==========================================
        ds_in = nc.Dataset(file_path, 'r')
        ds_in.set_auto_mask(False)  # 强行关闭自动 Mask 生成
        
        patchtype = ds_in.variables['patchtype'][:]
        mask = patchtype < 3
        if not np.any(mask):
            ds_in.close()
            return True, file_path, f"No valid patches in {file_name}"
            
        valid_idx = np.where(mask)[0]
        n_patches = len(valid_idx)
        total_time = ds_in.variables['time'].shape[0]
        
        lon_np = ds_in.variables['lon'][valid_idx]
        lat_np = ds_in.variables['lat'][valid_idx]
        patchclass_np = ds_in.variables['patchclass'][valid_idx]
        
        # 引入新代码的属性读取
        wf_clay_np = ds_in.variables['wf_clay'][valid_idx, :]
        wf_sand_np = ds_in.variables['wf_sand'][valid_idx, :]  # 新增：读取砂粒质量百分比
        
        if 'BD_all' in ds_in.variables:
            bd_all_np = ds_in.variables['BD_all'][valid_idx, :]
        elif 'bden_soi' in ds_in.variables:
            bd_all_np = ds_in.variables['bden_soi'][valid_idx, :]
        else:
            bd_all_np = np.full_like(wf_clay_np, 1400.0)
        
        clay_surf_pct_np = (wf_clay_np[:, 0]*0.0175 + wf_clay_np[:, 1]*0.0276) / wtot
        clay_surf_np = clay_surf_pct_np / 100.0 
        
        sand_surf_pct_np = (wf_sand_np[:, 0]*0.0175 + wf_sand_np[:, 1]*0.0276) / wtot  # 新增：计算表层砂粒比例(%)
        bd_surf_np = (bd_all_np[:, 0]*0.0175 + bd_all_np[:, 1]*0.0276) / wtot / 1000.0

        # 引入新代码的 GPU Tensor 缓存
        base_clay_all = torch.tensor(wf_clay_np, dtype=torch.float32, device=device)
        base_clay_surf_pct = torch.tensor(clay_surf_pct_np, dtype=torch.float32, device=device)
        base_clay_surf = torch.tensor(clay_surf_np, dtype=torch.float32, device=device)
        base_sand_all = torch.tensor(wf_sand_np, dtype=torch.float32, device=device) 
        base_sand_surf_pct = torch.tensor(sand_surf_pct_np, dtype=torch.float32, device=device)
        base_bd_surf = torch.tensor(bd_surf_np, dtype=torch.float32, device=device)

        ds_out = nc.Dataset(out_file_path, 'w')
        ds_out.createDimension('time', total_time)
        ds_out.createDimension('patch', n_patches)
        
        v_time = ds_out.createVariable('time', 'f8', ('time',))
        v_lon = ds_out.createVariable('lon', 'f8', ('patch',))
        v_lat = ds_out.createVariable('lat', 'f8', ('patch',))
        v_pclass = ds_out.createVariable('patchclass', 'i4', ('patch',))
        
        v_time[:] = ds_in.variables['time'][:]
        v_lon[:] = lon_np
        v_lat[:] = lat_np
        v_pclass[:] = patchclass_np
        
        out_vars = {}
        for key in res_keys:
            out_vars[key] = ds_out.createVariable(key, 'f4', ('time', 'patch'), zlib=True)

        # ==========================================
        # 3. GPU 推理核心循环 (修复 VRAM 爆炸版)
        # ==========================================
        with torch.no_grad():
            for t_start in range(0, total_time, time_chunk_size):
                t_end = min(t_start + time_chunk_size, total_time)
                cur_T = t_end - t_start
                
                # 从 NetCDF 连续读取大块 (维持老代码的极速 I/O)
                t_soisno_raw = ds_in.variables['t_soisno'][:]
                wliq_soisno_raw = ds_in.variables['wliq_soisno'][:]
                wice_soisno_raw = ds_in.variables['wice_soisno'][:]
                
                t_soisno_chunk = t_soisno_raw[t_start:t_end, valid_idx, 5:]
                wliq_soisno_chunk = wliq_soisno_raw[t_start:t_end, valid_idx, 5:]
                wice_soisno_chunk = wice_soisno_raw[t_start:t_end, valid_idx, 5:]
                
                del t_soisno_raw, wliq_soisno_raw, wice_soisno_raw
                
                # 🚀 修复 1：仅在 CPU 端创建 Full Tensor，绝对不要加 device=device!
                t_soi_full = torch.tensor(t_soisno_chunk, dtype=torch.float32).reshape(-1, 10)
                wliq_soi_full = torch.tensor(wliq_soisno_chunk, dtype=torch.float32).reshape(-1, 10)
                wice_soi_full = torch.tensor(wice_soisno_chunk, dtype=torch.float32).reshape(-1, 10)
                
                total_samples = cur_T * n_patches
                
                # 🚀 修复 2：彻底删除了之前这里所有的 expand().reshape() 代码
                
                chunk_res = {k: [] for k in res_keys}

                for i in range(0, total_samples, batch_size):
                    end_idx = min(i + batch_size, total_samples)
                    cur_b = end_idx - i
                    
                    # 🚀 修复 3：仅将当前 Batch 的数据推入 5090，non_blocking 加速传输
                    t_soi = t_soi_full[i:end_idx].to(device, non_blocking=True)
                    wliq_soi = wliq_soi_full[i:end_idx].to(device, non_blocking=True)
                    wice_soi = wice_soi_full[i:end_idx].to(device, non_blocking=True)
                    
                    # 🚀 修复 4：利用取模动态获取当前 Batch 对应的 Patch 静态属性，免除显存爆炸
                    patch_indices = torch.arange(i, end_idx, device=device) % n_patches
                    
                    clay_all = base_clay_all[patch_indices]
                    clay_surf_pct = base_clay_surf_pct[patch_indices]
                    clay_surf = base_clay_surf[patch_indices]
                    sand_all = base_sand_all[patch_indices]
                    sand_surf_pct = base_sand_surf_pct[patch_indices]
                    bd_surf = base_bd_surf[patch_indices]
                    
                    # 生成当前 Batch 常量
                    t_lam = torch.full((cur_b,), lam, dtype=torch.float32, device=device)
                    t_theta = torch.full((cur_b,), sat_theta, dtype=torch.float32, device=device)
                    dz_soi = rtm_model.dz_soi.unsqueeze(0).expand(cur_b, -1)
                    
                    # ----------------------------------------------------
                    # 以下保留新代码的完整物理逻辑：严格转换与介电常数沙漠校正
                    # ----------------------------------------------------
                    denh2o = 1000.0
                    denice = 917.0
                    liq_all = wliq_soi / (dz_soi * denh2o)
                    ice_all = wice_soi / (dz_soi * denice)
                    wc_all = liq_all  # M09计算时输入液态水部分
                    
                    liq_surf = (wliq_soi[:, 0] + wliq_soi[:, 1]) / (wtot * denh2o)
                    ice_surf = (wice_soi[:, 0] + wice_soi[:, 1]) / (wtot * denice)
                    wc_surf = liq_surf
                    
                    t_surf = ((t_soi[:, 0]*0.0175 + t_soi[:, 1]*0.0276) / wtot)
                    t_deep = ((t_soi[:, 6]*(0.8289-0.5) + t_soi[:, 7]*(1.0-0.8289)) / 0.5)

                    eps_nd = rtm_model.diel_soil_M09(wc_all, t_soi - rtm_model.tfrz, clay_all, f)
                    eps_surf_nd = rtm_model.diel_soil_M09(wc_surf, t_surf - rtm_model.tfrz, clay_surf_pct, f)

                    ffrz_all = torch.where((liq_all + ice_all) <= 0.0, 
                                           torch.zeros_like(liq_all),
                                           ice_all / torch.clamp(liq_all + ice_all, min=1e-6))
                    ffrz_surf = torch.where((liq_surf + ice_surf) <= 0.0, 
                                            torch.zeros_like(liq_surf), 
                                            ice_surf / torch.clamp(liq_surf + ice_surf, min=1e-6))
                    
                    eps_f_all = torch.complex(torch.full_like(liq_all, 5.0), torch.full_like(liq_all, 0.5))
                    eps_f_surf = torch.complex(torch.full_like(liq_surf, 5.0), torch.full_like(liq_surf, 0.5))
                    
                    eps_mixed = eps_nd * (1. - ffrz_all) + eps_f_all * ffrz_all
                    eps_surf_mixed = eps_surf_nd * (1. - ffrz_surf) + eps_f_surf * ffrz_surf

                    # 沙漠掩膜校正
                    is_desert = (liq_surf < 0.02) & (sand_surf_pct > 90.0)
                    is_desert_all = (liq_all < 0.02) & (sand_all > 90.0)
                    
                    eps_d_real = 2.53 + (2.79 - 2.53)/(1 + (sat_fghz/0.27)**2)
                    eps_d_imag = (2.79 - 2.53)*(sat_fghz/0.27)/(1 + (sat_fghz/0.27)**2) + 0.002
                    
                    eps_soil_d_all = torch.complex(torch.full_like(liq_all, eps_d_real), 
                                                   torch.full_like(liq_all, eps_d_imag))
                    eps_soil_d_surf = torch.complex(torch.full_like(liq_surf, eps_d_real), 
                                                    torch.full_like(liq_surf, eps_d_imag))
                                                    
                    eps = torch.where(is_desert_all, eps_soil_d_all, eps_mixed)
                    eps_surf = torch.where(is_desert, eps_soil_d_surf, eps_surf_mixed)

                    # ----------------------------------------------------
                    # RTM 计算保持不变
                    # ----------------------------------------------------
                    r_h, r_v, teff_wilheit_h, teff_wilheit_v = rtm_model.eff_soil_temp_Wilheit(dz_soi, t_soi, eps, t_theta, lamcm)
                    teff_lv_multi, weights_lv = rtm_model.eff_soil_temp_Lv_multi(dz_soi, t_soi, eps, t_lam, return_weights=True)
                    
                    cum_dz = torch.cumsum(dz_soi, dim=1)                 
                    cum_weights = torch.cumsum(weights_lv, dim=1)        
                    
                    idx_90 = torch.argmax((cum_weights >= 0.9).int(), dim=1, keepdim=True)
                    cw_i = torch.gather(cum_weights, 1, idx_90).squeeze(1)
                    dz_i = torch.gather(dz_soi, 1, idx_90).squeeze(1)
                    
                    pad_cw = torch.cat([torch.zeros_like(cum_weights[:, :1]), cum_weights], dim=1)
                    pad_cdz = torch.cat([torch.zeros_like(cum_dz[:, :1]), cum_dz], dim=1)
                    
                    cw_prev = torch.gather(pad_cw, 1, idx_90).squeeze(1)
                    cdz_prev = torch.gather(pad_cdz, 1, idx_90).squeeze(1)
                    
                    weight_diff = torch.clamp(cw_i - cw_prev, min=1e-8) 
                    depth_90 = cdz_prev + (0.9 - cw_prev) / weight_diff * dz_i

                    teff_lv_two = rtm_model.eff_soil_temp_Lv_two(wtot, t_surf, t_deep, eps_surf, t_lam)
                    teff_wigneron = rtm_model.eff_soil_temp_Wigneron2001(wc_surf, t_surf, t_deep)
                    teff_holmes2006 = rtm_model.eff_soil_temp_Holmes2006(eps_surf, t_surf, t_deep)
                    teff_wigneron2008 = rtm_model.eff_soil_temp_Wigneron2008(wc_surf, t_surf, t_deep, clay_surf, bd_surf)

                    chunk_res['r_H_wilheit'].append(r_h.cpu().numpy())
                    chunk_res['r_V_wilheit'].append(r_v.cpu().numpy())
                    chunk_res['T_eff_wilheit_H'].append(teff_wilheit_h.cpu().numpy())
                    chunk_res['T_eff_wilheit_V'].append(teff_wilheit_v.cpu().numpy())
                    chunk_res['T_eff_lv_multi'].append(teff_lv_multi.cpu().numpy())
                    chunk_res['T_eff_lv_two'].append(teff_lv_two.cpu().numpy())
                    chunk_res['T_eff_wigneron'].append(teff_wigneron.cpu().numpy())
                    chunk_res['T_eff_holmes2006'].append(teff_holmes2006.cpu().numpy())
                    chunk_res['T_eff_wigneron2008'].append(teff_wigneron2008.cpu().numpy())
                    chunk_res['depth_90_lv_multi'].append(depth_90.cpu().numpy()) 

                for key in res_keys:
                    concat_res = np.concatenate(chunk_res[key])
                    out_vars[key][t_start:t_end, :] = concat_res.reshape(cur_T, n_patches)
        
        ds_in.close()
        ds_out.close()
        print(f"✅ [GPU {device_id} | {time.strftime('%H:%M:%S')}] Finished {out_file_name} | Cost: {time.time() - t_file_start:.2f} s")
        return True, file_path, f"Processed {file_name} -> GPU {device_id}"
        
    except Exception as e:
        print(f"❌ [GPU {device_id}] Error processing {file_path}: {e}")
        return False, file_path, f"Error in {os.path.basename(file_path)}: {e}"
        
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

if __name__ == "__main__":
    # 强制在子进程中重新分配 CUDA 内存池以防止死锁
    mp.set_start_method('spawn', force=True)

    nc_dir = '/home/liusy/CoLM/outputs/global_veg_wigneron/forward_inputs_folder'
    output_dir = '/home/liusy/research_lists/2026-06-01_research_list/compare_diff_teff_right_now/results_over_1000'
    os.makedirs(output_dir, exist_ok=True)

    nc_files = sorted(glob.glob(os.path.join(nc_dir, 'forward_inputs_worker*.nc')))
    total_files = len(nc_files)
    
    MAX_WORKERS = 4  # 4进程对应 4张 5090
    
    print(f"\n=======================================================")
    print(f"🚀 开始多卡高并发提取，共发现 {total_files} 个 NetCDF 文件")
    print(f"分配策略: {MAX_WORKERS} 个并发 Worker 均匀分布在 4 张 5090 显卡上")
    print(f"日志状态: 进程内计算细节将被静默记录到 `log_files_compare` 目录")
    print(f"=======================================================\n")

    t_global_start = time.time()

    # 启用 ProcessPoolExecutor 强力调度
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 将任务提交入池
        futures = {
            executor.submit(worker_process, idx, file_path, total_files, output_dir): file_path 
            for idx, file_path in enumerate(nc_files, 1)
        }
        
        # tqdm 在主线程接管进度，展示极其整洁的 UI
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="🚀 全局计算进度"):
            success, file_path, msg = future.result()
            
            # 只有遇到错误时，才打破进度条在主终端报错提醒你
            if not success:
                tqdm.write(f"❌ 警告: {msg}")

    print(f"\n🎉 All files processed successfully! Total Elapsed Time: {(time.time() - t_global_start) / 60:.2f} mins.")
    
      
    
    
