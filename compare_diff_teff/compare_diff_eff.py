import os
import glob
import torch
import datetime
import numpy as np
import pandas as pd
import netCDF4 as nc

# Import the RTM model containing the added Teff schemes
from rtm import DifferentiableRTM
import time  # <--- 修改处：引入时间模块

def main():
    output_dir = '/home/liusy/research_lists/2026-06-01_research_list/compare_diff_teff/results'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rtm_model = DifferentiableRTM().to(device)
    rtm_model.eval()  # 设置为评估模式

    # 物理与卫星常量设定
    sat_fghz = 1.4 
    sat_theta = 40.0 * np.pi / 180.0
    f = sat_fghz * 1e9
    lam = rtm_model.C / f
    lamcm = lam * 100.0

    nc_dir = '/home/liusy/CoLM/outputs/global_veg_wigneron/forward_inputs_folder'
    nc_files = sorted(glob.glob(os.path.join(nc_dir, 'forward_inputs_worker*.nc')))
    
    print(f"Found {len(nc_files)} NetCDF files. Initializing static data...")

    # ==========================================
    # STEP 1: 只在最开始读取并聚合所有的静态数据
    # ==========================================
    datasets = []
    valid_masks = []
    all_lon, all_lat, all_patchclass = [], [], []
    all_wf_clay, all_bd_all = [], []

    for file_path in nc_files:
        try:
            ds = nc.Dataset(file_path, 'r')
            patchtype = ds.variables['patchtype'][:]
            mask = patchtype < 3
            
            if not np.any(mask):
                ds.close()
                continue
            
            datasets.append(ds)
            valid_masks.append(mask)
            
            all_lon.append(ds.variables['lon'][mask])
            all_lat.append(ds.variables['lat'][mask])
            all_patchclass.append(ds.variables['patchclass'][mask])
            all_wf_clay.append(ds.variables['wf_clay'][mask, :])
            
            if 'BD_all' in ds.variables:
                all_bd_all.append(ds.variables['BD_all'][mask, :])
            elif 'bden_soi' in ds.variables:
                all_bd_all.append(ds.variables['bden_soi'][mask, :])
            else:
                all_bd_all.append(np.full_like(ds.variables['wf_clay'][mask, :], 1400.0))
        except Exception as e:
            print(f"Error initializing {file_path}: {e}")

    # 拼接全局静态 Numpy 数组
    lon_np = np.concatenate(all_lon)
    lat_np = np.concatenate(all_lat)
    patchclass_np = np.concatenate(all_patchclass)
    wf_clay_np = np.concatenate(all_wf_clay)
    bd_all_np = np.concatenate(all_bd_all)
    
    n_samples = len(lon_np)
    print(f"Global valid patches loaded: {n_samples}.")

    # ==========================================
    # STEP 2: 预计算表层整合参数，并常驻 GPU (节省每小时计算开销)
    # ==========================================
    wtot = 0.0175 + 0.0276
    
    # 预计算表层的粘土与容重 (合并层1和层2)
    clay_surf_pct_np = (wf_clay_np[:, 0]*0.0175 + wf_clay_np[:, 1]*0.0276) / wtot
    clay_surf_np = clay_surf_pct_np / 100.0 # Fraction format
    bd_surf_np = (bd_all_np[:, 0]*0.0175 + bd_all_np[:, 1]*0.0276) / wtot / 1000.0

    # 将不变的土壤属性推入 GPU
    clay_all_gpu = torch.tensor(wf_clay_np, dtype=torch.float32, device=device)
    clay_surf_pct_gpu = torch.tensor(clay_surf_pct_np, dtype=torch.float32, device=device)
    clay_surf_gpu = torch.tensor(clay_surf_np, dtype=torch.float32, device=device)
    bd_surf_gpu = torch.tensor(bd_surf_np, dtype=torch.float32, device=device)
    
    t_lam_gpu = torch.full((batch_size := 300000,), lam, dtype=torch.float32, device=device)
    t_theta_gpu = torch.full((batch_size,), sat_theta, dtype=torch.float32, device=device)

    # 确定时间轴
    start_dt = datetime.datetime(2016, 1, 1, 0, 0)
    total_time_steps = 8784  # 针对 2016 闰年

    print("Starting hourly forward simulations...")

    # ==========================================
    # STEP 3: 时间循环 (避免重新打开文件，直接流式读取)
    # ==========================================
    t0 =time.time()
    try:
        with torch.no_grad(): # 全局禁用计算图
            for time_idx in range(total_time_steps):
                current_dt = start_dt + datetime.timedelta(hours=time_idx)
                time_format_str = current_dt.strftime("%Y-%m-%d-%H")
                
                # 提取当前时刻的动态数据 (只提取后10个土壤层 5:15)
                all_t_soisno, all_wliq_soisno = [], []
                for ds, mask in zip(datasets, valid_masks):
                    # 为了规避不同 netcdf4 库的高级切片 Bug，先截取维度，再用 mask 过滤
                    t_slice = ds.variables['t_soisno'][time_idx, :, 5:]
                    wliq_slice = ds.variables['wliq_soisno'][time_idx, :, 5:]
                    all_t_soisno.append(t_slice[mask])
                    all_wliq_soisno.append(wliq_slice[mask])
                    
                t_soi_np = np.concatenate(all_t_soisno)
                wliq_soi_np = np.concatenate(all_wliq_soisno)
                
                # 一次性将当前时间的动态数据推入 GPU
                t_soi_full = torch.tensor(t_soi_np, dtype=torch.float32, device=device)
                wliq_soi_full = torch.tensor(wliq_soi_np, dtype=torch.float32, device=device)

                # 用于收集当前时刻的结果
                res_dict = {
                    'r_H_wilheit': [], 'r_V_wilheit': [],
                    'T_eff_wilheit_H': [], 'T_eff_wilheit_V': [],
                    'T_eff_lv_multi': [], 'T_eff_lv_two': [],
                    'T_eff_wigneron': [], 'T_eff_holmes2006': [], 'T_eff_wigneron2008': []
                }

                # 在 GPU 上做 Batch 推理防爆显存
                for i in range(0, n_samples, batch_size):
                    end_idx = min(i + batch_size, n_samples)
                    cur_b = end_idx - i
                    
                    t_soi = t_soi_full[i:end_idx]
                    wliq_soi = wliq_soi_full[i:end_idx]
                    clay_all = clay_all_gpu[i:end_idx]
                    clay_surf_pct = clay_surf_pct_gpu[i:end_idx]
                    clay_surf = clay_surf_gpu[i:end_idx]
                    bd_surf = bd_surf_gpu[i:end_idx]
                    
                    t_lam = t_lam_gpu[:cur_b]
                    t_theta = t_theta_gpu[:cur_b]

                    dz_soi = rtm_model.dz_soi.unsqueeze(0).expand(cur_b, -1)
                    wc_all = wliq_soi / (dz_soi * 100.0)
                    
                    # 动态求解温度与水分的表层及深层整体参数
                    t_surf = ((t_soi[:, 0]*0.0175 + t_soi[:, 1]*0.0276) / wtot)
                    wc_surf = (wliq_soi[:, 0] + wliq_soi[:, 1]) / (wtot * 1000.0)
                    t_deep = ((t_soi[:, 6]*(0.8289-0.5) + t_soi[:, 7]*(1.0-0.8289)) / 0.5)

                    # 统一计算各层复介电常数及表层整体复介电常数
                    eps = rtm_model.diel_soil_M09(wc_all, t_soi - rtm_model.tfrz, clay_all, f)
                    eps_surf = rtm_model.diel_soil_M09(wc_surf, t_surf - rtm_model.tfrz, clay_surf_pct, f)

                    # ====== 执行各种 Teff 方案 ======
                    r_h, r_v, teff_wilheit_h, teff_wilheit_v = rtm_model.eff_soil_temp_Wilheit(dz_soi, t_soi, eps, t_theta, lamcm)
                    teff_lv_multi = rtm_model.eff_soil_temp_Lv_multi(dz_soi, t_soi, eps, t_lam)
                    
                    # Lv Two-layer (传入整合后的表层参数)
                    teff_lv_two = rtm_model.eff_soil_temp_Lv_two(wtot, t_surf, t_deep, eps_surf, t_lam)
                    
                    teff_wigneron = rtm_model.eff_soil_temp_Wigneron2001(wc_surf, t_surf, t_deep)
                    
                    # Holmes & Wigneron 2008 
                    teff_holmes2006 = rtm_model.eff_soil_temp_Holmes2006(eps_surf, t_surf, t_deep)
                    teff_wigneron2008 = rtm_model.eff_soil_temp_Wigneron2008(wc_surf, t_surf, t_deep, clay_surf, bd_surf)

                    # 将计算结果退回 CPU 并记录
                    res_dict['r_H_wilheit'].append(r_h.cpu().numpy())
                    res_dict['r_V_wilheit'].append(r_v.cpu().numpy())
                    res_dict['T_eff_wilheit_H'].append(teff_wilheit_h.cpu().numpy())
                    res_dict['T_eff_wilheit_V'].append(teff_wilheit_v.cpu().numpy())
                    res_dict['T_eff_lv_multi'].append(teff_lv_multi.cpu().numpy())
                    res_dict['T_eff_lv_two'].append(teff_lv_two.cpu().numpy())
                    res_dict['T_eff_wigneron'].append(teff_wigneron.cpu().numpy())
                    res_dict['T_eff_holmes2006'].append(teff_holmes2006.cpu().numpy())
                    res_dict['T_eff_wigneron2008'].append(teff_wigneron2008.cpu().numpy())

                # === 生成当小时的 DataFrame 并求误差统计 ===
                df_data = {
                    'patch_lon': lon_np,
                    'patch_lat': lat_np,
                    'patchclass': patchclass_np
                }
                for key in res_dict:
                    df_data[key] = np.concatenate(res_dict[key])
                final_df = pd.DataFrame(df_data)
                
                # 保存空间像素 CSV
                csv_filename = f"Teff_compare_{time_format_str}.csv"
                final_df.to_csv(os.path.join(output_dir, csv_filename), index=False)

                # === 误差统计清单 ===
                plot_configs = [
                    {"col": "T_eff_lv_multi", "ref": "T_eff_wilheit_H", "title": "Lv Multi-layer - Wilheit (H)", "is_temp": True},
                    {"col": "T_eff_lv_multi", "ref": "T_eff_wilheit_V", "title": "Lv Multi-layer - Wilheit (V)", "is_temp": True},
                    {"col": "T_eff_lv_two", "ref": "T_eff_wilheit_H", "title": "Lv Two-layer - Wilheit (H)", "is_temp": True},
                    {"col": "T_eff_lv_two", "ref": "T_eff_wilheit_V", "title": "Lv Two-layer - Wilheit (V)", "is_temp": True},
                    {"col": "T_eff_wigneron", "ref": "T_eff_wilheit_H", "title": "Wigneron 2001 - Wilheit (H)", "is_temp": True},
                    {"col": "T_eff_wigneron", "ref": "T_eff_wilheit_V", "title": "Wigneron 2001 - Wilheit (V)", "is_temp": True},
                    {"col": "T_eff_wigneron2008", "ref": "T_eff_wilheit_H", "title": "Wigneron 2008 - Wilheit (H)", "is_temp": True},
                    {"col": "T_eff_wigneron2008", "ref": "T_eff_wilheit_V", "title": "Wigneron 2008 - Wilheit (V)", "is_temp": True},
                    {"col": "T_eff_holmes2006", "ref": "T_eff_wilheit_H", "title": "Holmes 2006 - Wilheit (H)", "is_temp": True},
                    {"col": "T_eff_holmes2006", "ref": "T_eff_wilheit_V", "title": "Holmes 2006 - Wilheit (V)", "is_temp": True},
                    {"col": "T_eff_wilheit_H", "ref": "T_eff_wilheit_V", "title": "Wilheit (H) - Wilheit (V)", "is_temp": True},
                    {"col": "r_H_wilheit", "ref": "r_V_wilheit", "title": "Wilheit r_H - Wilheit r_V", "is_temp": False}
                ]

                stats_list = []
                for cfg in plot_configs:
                    diff = final_df[cfg["col"]] - final_df[cfg["ref"]]
                    unit_str = "K" if cfg["is_temp"] else "-"
                    stats_list.append({
                        "Scheme Profile": cfg["title"],
                        "Total Patches": len(diff),
                        "Unit": unit_str,
                        "MBE": round(float(diff.mean()), 4),
                        "SD": round(float(diff.std()), 4),
                        "RMSE": round(float(np.sqrt((diff ** 2).mean())), 4),
                        "Min Diff": round(float(diff.min()), 4),
                        "Max Diff": round(float(diff.max()), 4)
                    })
                
                # 保存统计摘要 CSV
                summary_df = pd.DataFrame(stats_list)
                stats_csv_filename = f"Teff_stats_summary_{time_format_str}.csv"
                summary_df.to_csv(os.path.join(output_dir, stats_csv_filename), index=False)
                
                print(f"[{time_format_str}] Processed & Saved (Time index: {time_idx}) | spend {time.time() - t0:.2f} seconds")

    finally:
        # 安全地关闭所有 NetCDF 文件句柄
        for ds in datasets:
            ds.close()
        print("All NetCDF datasets closed successfully. Mission complete.")

if __name__ == "__main__":
    main()






