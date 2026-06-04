import os
import glob
import torch
import datetime
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import calendar
import time
from concurrent.futures import ProcessPoolExecutor

# Import the RTM model containing the added Teff schemes
from rtm import DifferentiableRTM

# ==========================================
# 提取成独立的函数，以便使用多进程并行读取
# ==========================================
def read_single_nc(args):
    file_path, time_idx = args
    try:
        dataset = nc.Dataset(file_path, 'r')
        patchtype = dataset.variables['patchtype'][:]
        
        # Filter non-water patches
        valid_mask = patchtype < 3
        if not np.any(valid_mask):
            dataset.close()
            return None
            
        lon = dataset.variables['lon'][:][valid_mask]
        lat = dataset.variables['lat'][:][valid_mask]
        patchclass = dataset.variables['patchclass'][:][valid_mask]
        
        t_soisno = dataset.variables['t_soisno'][time_idx, valid_mask, :]
        wliq_soisno = dataset.variables['wliq_soisno'][time_idx, valid_mask, :]
        wf_clay = dataset.variables['wf_clay'][valid_mask, :]
        wf_sand = dataset.variables['wf_sand'][valid_mask, :]  # 读取沙粒比例
        
        if 'BD_all' in dataset.variables:
            bd_all = dataset.variables['BD_all'][valid_mask, :]
        elif 'bden_soi' in dataset.variables:
            bd_all = dataset.variables['bden_soi'][valid_mask, :]
        else:
            bd_all = np.full_like(wf_clay, 1400.0) 
            
        dataset.close()
        
        return {
            'lon': lon, 'lat': lat, 'patchclass': patchclass,
            't_soisno': t_soisno, 'wliq_soisno': wliq_soisno,
            'wf_clay': wf_clay, 'wf_sand': wf_sand, 'bd_all': bd_all
        }
    except Exception as e:
        return None

def main(target_year, target_month, target_day, target_hour, output_dir):
    # ==========================================
    # TIME SETTING 
    # ==========================================
    target_dt = datetime.datetime(target_year, target_month, target_day, target_hour)
    start_of_year = datetime.datetime(target_year, 1, 1)
    time_idx = int((target_dt - start_of_year).total_seconds() // 3600)
    
    time_format_str = f"{target_year:04d}-{target_month:02d}-{target_day:02d}-{target_hour:02d}"
    print(f"\nTarget Datetime: {target_dt} | Time Index in dataset: {time_idx}")

    nc_dir = '/home/liusy/CoLM/outputs/global_veg_wigneron/forward_inputs_folder'
    nc_files = glob.glob(os.path.join(nc_dir, 'forward_inputs_worker*.nc'))
    print(f"Found {len(nc_files)} NetCDF files. Reading in parallel...")

    # ==================================================
    # 1. 采用多进程并行读取所有 NetCDF 文件
    # ==================================================
    t_io_start = time.time()
    results = []
    
    max_workers = min(os.cpu_count(), 16) 
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        args_list = [(fp, time_idx) for fp in nc_files]
        for res in executor.map(read_single_nc, args_list):
            if res is not None:
                results.append(res)
                
    print(f"I/O read completed in {time.time() - t_io_start:.2f} seconds.")

    if not results:
        print("No valid data found in this time slice.")
        return

    # ==================================================
    # 2. 将数据合并为大的 numpy 数组
    # ==================================================
    lon_all = np.concatenate([r['lon'] for r in results])
    lat_all = np.concatenate([r['lat'] for r in results])
    patchclass_all = np.concatenate([r['patchclass'] for r in results])
    
    t_soisno_all = np.concatenate([r['t_soisno'] for r in results])
    wliq_soisno_all = np.concatenate([r['wliq_soisno'] for r in results])
    wf_clay_all = np.concatenate([r['wf_clay'] for r in results])
    wf_sand_all = np.concatenate([r['wf_sand'] for r in results]) 
    bd_all_all = np.concatenate([r['bd_all'] for r in results])
    
    num_patches = len(lon_all)
    print(f"Total valid patches to compute: {num_patches}")

    # ==================================================
    # 3. GPU 初始化与物理常量
    # ==================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rtm_model = DifferentiableRTM().to(device)

    sat_fghz = 1.4 
    sat_theta = 40.0 * np.pi / 180.0
    f = sat_fghz * 1e9
    lam = rtm_model.C / f
    lamcm = lam * 100.0

    # ==================================================
    # 4. 大 Batch 喂给 GPU 并进行变量反推计算
    # ==================================================
    t_compute_start = time.time()
    batch_size = 250000  
    
    # 初始化按严格指定的列顺序存储的字典容器 (新增 t_soi_max 位于 t_deep 与 eps_surf_real 之间)
    out_dict = {
        'T_eff_wilheit_H': [], 'T_eff_wilheit_V': [], 'T_eff_lv_multi': [], 'T_eff_lv_two': [],
        'T_eff_wigneron': [], 'T_eff_holmes2006': [], 'T_eff_wigneron2008': [], 'depth_90_lv_multi': [],
        't_surf': [], 'wc_surf': [], 'clay_surf': [], 'mvt_surf': [], 'sand_surf': [], 'bd_surf': [], 
        't_deep': [], 
        't_soi_max': [], 't_soi_min': [], # <--- 新增：10层土壤最大温度位置
        'eps_surf_real': [], 'eps_surf_imag': [],
        'C_lv_two': [], 'C_holmes2006': [], 'C_inverted_wilheit': [], 'b_param_inverted': [], 'dz_surf_inverted': []
    }
    
    # 动态加入 10层土壤温度键，会自动排在 dz_surf_inverted 之后
    for layer in range(1, 11):
        out_dict[f't_soi_layer_{layer}'] = []

    for i in range(0, num_patches, batch_size):
        end_idx = min(i + batch_size, num_patches)
        
        t_soi = torch.tensor(t_soisno_all[i:end_idx, 5:], dtype=torch.float32, device=device)
        wliq_soi = torch.tensor(wliq_soisno_all[i:end_idx, 5:], dtype=torch.float32, device=device)
        clay_all = torch.tensor(wf_clay_all[i:end_idx, :], dtype=torch.float32, device=device)
        sand_all = torch.tensor(wf_sand_all[i:end_idx, :], dtype=torch.float32, device=device)
        bd_all_tensor = torch.tensor(bd_all_all[i:end_idx, :], dtype=torch.float32, device=device)

        dz_soi = rtm_model.dz_soi.unsqueeze(0).expand(t_soi.shape[0], -1)
        wc_all = wliq_soi / (dz_soi * 100.0)
        
        # wtot = 0.0175 + 0.0276
        # t_surf = ((t_soi[:, 0]*0.0175 + t_soi[:, 1]*0.0276) / wtot)
        # wc_surf = (wliq_soi[:, 0] + wliq_soi[:, 1]) / (wtot * 1000.0)
        # t_deep = ((t_soi[:, 6]*(0.8289-0.5) + t_soi[:, 7]*(1.0-0.8289)) / 0.5)
        wtot = 0.0175
        t_surf = t_soi[:, 0]
        wc_surf = wliq_soi[:, 0] / (wtot * 1000.0)   
        t_deep = ((t_soi[:, 6]*(0.8289-0.5) + t_soi[:, 7]*(1.0-0.8289)) / 0.5) # 改成t_soi[:, -1]没啥变化
        bd_surf = bd_all_tensor[:, 0] / 1000.0    
        
        # 计算10层土壤的最大温度 (沿维度1求最大值)
        t_soi_max = torch.max(t_soi, dim=1)[0]
        # 计算10层土壤的最小温度 (沿维度1求最小值)
        t_soi_min = torch.min(t_soi, dim=1)[0]

        # clay_surf_pct = (clay_all[:, 0]*0.0175 + clay_all[:, 1]*0.0276) / wtot 
        # sand_surf_pct = (sand_all[:, 0]*0.0175 + sand_all[:, 1]*0.0276) / wtot  
        # clay_surf = clay_surf_pct / 100.0 
        clay_surf_pct = clay_all[:, 0] 
        clay_surf = clay_surf_pct / 100.0
        sand_surf_pct = sand_all[:, 0]
        # 计算 M09 方案的最大结合水含量 (mvt)
        mvt_surf = 0.02863 + 0.30673e-2 * clay_surf_pct
        
        eps = rtm_model.diel_soil_M09(wc_all, t_soi - rtm_model.tfrz, clay_all, f)
        eps_surf = rtm_model.diel_soil_M09(wc_surf, t_surf - rtm_model.tfrz, clay_surf_pct, f)
        
        t_lam = torch.full((t_soi.shape[0],), lam, dtype=torch.float32, device=device)
        t_theta = torch.full((t_soi.shape[0],), sat_theta, dtype=torch.float32, device=device)

        with torch.no_grad():
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
            
            teff_lv_two, C_lv_two = rtm_model.eff_soil_temp_Lv_two(wtot, t_surf, t_deep, eps_surf, t_lam, return_C=True)
            teff_wigneron = rtm_model.eff_soil_temp_Wigneron2001(wc_surf, t_surf, t_deep)
            
            teff_holmes2006, C_holmes2006 = rtm_model.eff_soil_temp_Holmes2006(wc_surf, eps_surf, t_surf, t_deep, return_C=True)
            teff_wigneron2008 = rtm_model.eff_soil_temp_Wigneron2008(wc_surf, t_surf, t_deep, clay_surf, bd_surf)

            # 基于 T_eff_wilheit_H 反推 C 值与修正后的 dz_surf (Δx)
            delta_T = t_surf - t_deep
            delta_T_sign = torch.where(delta_T >= 0, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))
            delta_T_safe = torch.where(delta_T.abs() < 1e-6, delta_T_sign * 1e-6, delta_T)
            C_inverted = (teff_wilheit_h - t_deep) / delta_T_safe
            
            C_inverted_clamped = torch.clamp(C_inverted, min=1e-5, max=1.0 - 1e-5)
            B1_inverted = -torch.log(1.0 - C_inverted_clamped)
            
            eps_surf_r = eps_surf.real
            eps_surf_i = torch.abs(eps_surf.imag)
            
            # --- 新增 logic: 依据 Holmes 公式 C = (eps_ratio / eps0_param)**b_param 反推 b_param ---
            eps0_param = 0.08
            eps_ratio = eps_surf_i / eps_surf_r
            eps_ratio_norm = torch.clamp(eps_ratio / eps0_param, min=1e-5)
            log_denom = torch.log(eps_ratio_norm)
            # 防止对数项为 0 导致被除数为零出现 nan/inf
            log_denom_safe = torch.where(log_denom.abs() < 1e-8, torch.tensor(1e-8, device=device), log_denom)
            b_param_inverted = torch.log(C_inverted_clamped) / log_denom_safe
            # --------------------------------------------------------------------------------
            
            factor = (4.0 * np.pi / t_lam) * (eps_surf_i / (2.0 * torch.sqrt(eps_surf_r)))
            factor_safe = torch.where(factor < 1e-8, torch.tensor(1e-8, device=device), factor)
            
            dz_surf_inverted = B1_inverted / factor_safe

        # 把结果安全拉回 CPU 并追加记录
        out_dict['T_eff_wilheit_H'].append(teff_wilheit_h.cpu().numpy())
        out_dict['T_eff_wilheit_V'].append(teff_wilheit_v.cpu().numpy())
        out_dict['T_eff_lv_multi'].append(teff_lv_multi.cpu().numpy())
        out_dict['T_eff_lv_two'].append(teff_lv_two.cpu().numpy())
        out_dict['T_eff_wigneron'].append(teff_wigneron.cpu().numpy())
        out_dict['T_eff_holmes2006'].append(teff_holmes2006.cpu().numpy())
        out_dict['T_eff_wigneron2008'].append(teff_wigneron2008.cpu().numpy())
        out_dict['depth_90_lv_multi'].append(depth_90.cpu().numpy())
        
        out_dict['t_surf'].append(t_surf.cpu().numpy())
        out_dict['wc_surf'].append(wc_surf.cpu().numpy())
        out_dict['clay_surf'].append(clay_surf_pct.cpu().numpy())  
        out_dict['mvt_surf'].append(mvt_surf.cpu().numpy())  
        out_dict['sand_surf'].append(sand_surf_pct.cpu().numpy())  
        out_dict['bd_surf'].append((bd_surf * 1000.0).cpu().numpy()) 
        out_dict['t_deep'].append(t_deep.cpu().numpy())
        
        # 将最大土壤温度存入对应插槽
        out_dict['t_soi_max'].append(t_soi_max.cpu().numpy())
        out_dict['t_soi_min'].append(t_soi_min.cpu().numpy())
        
        out_dict['eps_surf_real'].append(eps_surf.real.cpu().numpy())
        out_dict['eps_surf_imag'].append(eps_surf.imag.cpu().numpy())
        
        out_dict['C_lv_two'].append(C_lv_two.cpu().numpy())
        out_dict['C_holmes2006'].append(C_holmes2006.cpu().numpy())
        out_dict['C_inverted_wilheit'].append(C_inverted.cpu().numpy())
        out_dict['b_param_inverted'].append(b_param_inverted.cpu().numpy())
        out_dict['dz_surf_inverted'].append(dz_surf_inverted.cpu().numpy())
        
        # 循环保存 10 层土壤的独立温度
        for layer in range(10):
            out_dict[f't_soi_layer_{layer+1}'].append(t_soi[:, layer].cpu().numpy())

    print(f"GPU Computing completed in {time.time() - t_compute_start:.2f} seconds.")

    # ==================================================
    # 5. 按照严格指定的列顺序组装 DataFrame 并保存
    # ==================================================
    final_data = {
        'patch_lon': lon_all,
        'patch_lat': lat_all,
        'patchclass': patchclass_all,
    }
    for key in out_dict:
        final_data[key] = np.concatenate(out_dict[key])
        
    final_df = pd.DataFrame(final_data)

    csv_filename = f"Teff_compare_{time_format_str}.csv"
    final_df.to_csv(os.path.join(output_dir, csv_filename), index=False)
    print(f"Calculation finished successfully. Saved data table to: {csv_filename}")

    # ==================================================
    # 【计算差异】用于样本条件筛选
    # ==================================================
    diff_lv_multi_abs = np.abs(final_df['T_eff_lv_multi'] - final_df['T_eff_wilheit_H'])
    diff_holmes_abs = np.abs(final_df['T_eff_holmes2006'] - final_df['T_eff_wilheit_H'])

    # 1. 筛选出大误差样本 (Holmes 绝对差异大于 5K)
    large_loss_threshold = 2.0  
    large_loss_mask = (diff_holmes_abs > large_loss_threshold)
    large_loss_df = final_df[large_loss_mask]
    
    large_loss_csv_filename = f"Teff_compare_large_loss_{time_format_str}.csv"
    large_loss_df.to_csv(os.path.join(output_dir, large_loss_csv_filename), index=False)
    print(f"Filtered large loss samples (Diff > {large_loss_threshold}K). Total patches: {len(large_loss_df)}. Saved to: {large_loss_csv_filename}")

    # 2. 新增逻辑：筛选出小误差样本 (任意一个模型的绝对差异小于 1K)
    little_loss_threshold = 1.0  
    little_loss_mask = (diff_holmes_abs <= little_loss_threshold) #& (diff_lv_multi_abs < little_loss_threshold)
    little_loss_df = final_df[little_loss_mask]
    
    little_loss_csv_filename = f"Teff_compare_little_loss_{time_format_str}.csv"
    little_loss_df.to_csv(os.path.join(output_dir, little_loss_csv_filename), index=False)
    print(f"Filtered little loss samples (Diff < {little_loss_threshold}K). Total patches: {len(little_loss_df)}. Saved to: {little_loss_csv_filename}\n")

    # ==================================================
    # GRID PLOT LAYOUT (保持5行2列原有空间差异图不变)
    # ==================================================
    fig, axs = plt.subplots(5, 2, figsize=(20, 30))
    plot_configs = [
        {"col": "T_eff_lv_multi", "ref": "T_eff_wilheit_H", "title": "Lv Multi-layer - Wilheit (H)", "type": "diff", "is_temp": True, "vmin": -20.0, "vmax": 20.0},
        {"col": "T_eff_lv_two", "ref": "T_eff_wilheit_H", "title": "Lv Two-layer - Wilheit (H)", "type": "diff", "is_temp": True, "vmin": -20.0, "vmax": 20.0},
        {"col": "T_eff_wigneron", "ref": "T_eff_wilheit_H", "title": "Wigneron 2001 - Wilheit (H)", "type": "diff", "is_temp": True, "vmin": -20.0, "vmax": 20.0},
        {"col": "T_eff_wigneron2008", "ref": "T_eff_wilheit_H", "title": "Wigneron 2008 - Wilheit (H)", "type": "diff", "is_temp": True, "vmin": -20.0, "vmax": 20.0},
        {"col": "T_eff_holmes2006", "ref": "T_eff_wilheit_H", "title": "Holmes 2006 - Wilheit (H)", "type": "diff", "is_temp": True, "vmin": -20.0, "vmax": 20.0},
        
        {"col": "depth_90_lv_multi", "title": "90% Penetration Depth (Lv Multi)", "type": "value", "unit": "m", "cmap": "viridis", "vmin": 0.0, "vmax": 0.2, "extend": "max"},
        {"col": "C_holmes2006", "title": "C Value (Holmes 2006)", "type": "value", "unit": "-", "cmap": "plasma", "vmin": 0.0, "vmax": 1.0, "extend": "neither"},
        {"col": "C_lv_two", "title": "C Value (Lv Two-layer)", "type": "value", "unit": "-", "cmap": "plasma", "vmin": 0.0, "vmax": 1.0, "extend": "neither"},
        
        {"col": "C_inverted_wilheit", "title": "Inverted C Value (Wilheit H)", "type": "value", "unit": "-", "cmap": "plasma", "vmin": 0.0, "vmax": 1.0, "extend": "both"},
        {"col": "dz_surf_inverted", "title": "Inverted dz_surf (Wilheit H)", "type": "value", "unit": "m", "cmap": "viridis", "vmin": 0.0, "vmax": 0.2, "extend": "max"}
    ]

    # 为了增加条件过滤统计，在后台提取当前的总表面湿度数组
    threshold_wc = 0.03
    wc_surf_np = final_df['wc_surf'].values
    mask_wet = wc_surf_np >= threshold_wc
    mask_dry = wc_surf_np < threshold_wc

    stats_list = []

    for i, cfg in enumerate(plot_configs):
        row_idx = i // 2
        col_idx = i % 2
        ax = axs[row_idx, col_idx]
        
        # 准备待统计的基础原始物理量
        if cfg["type"] == "diff":
            base_val = (final_df[cfg["col"]] - final_df[cfg["ref"]]).values
            unit_str = "K" if cfg["is_temp"] else "-"
            is_diff_mode = True
        else:
            base_val = final_df[cfg["col"]].values
            unit_str = cfg["unit"]
            is_diff_mode = False

        # 循环生成三个湿度状态的数据切片 (Total, >=threshold_wc, <threshold_wc)
        conditions = [
            {"label": "Total", "mask": np.ones_like(base_val, dtype=bool)},
            {"label": f"wc>={threshold_wc}", "mask": mask_wet},
            {"label": f"wc<{threshold_wc}", "mask": mask_dry}
        ]
        
        for cond in conditions:
            sub_val = base_val[cond["mask"]]
            
            # 若切片内没有任何有效元素，填入占位符
            if len(sub_val) == 0 or np.all(np.isnan(sub_val)):
                stats_list.append({
                    "Scheme Profile": f"{cfg['title']}", "Condition": cond["label"], "Total Patches": 0, "Unit": unit_str,
                    "Mean/MBE": "-", "SD": "-", "RMSE": "-", "Min": "-", 
                    "mu-3sigma": "-", "mu-2sigma": "-", "mu-sigma": "-", "50% Q(Med)": "-", 
                    "mu+sigma": "-", "mu+2sigma": "-", "mu+3sigma": "-", "Max": "-"
                })
                continue
                
            mean_val = np.nanmean(sub_val)
            std_dev = np.nanstd(sub_val)
            rmse_val = np.sqrt(np.nanmean(sub_val ** 2)) if is_diff_mode else "-"
            min_val = np.nanmin(sub_val)
            max_val = np.nanmax(sub_val)
            
            # 丰富 1/2/3-Sigma 的多级分界计算
            m_3sig = mean_val - 3.0 * std_dev
            m_2sig = mean_val - 2.0 * std_dev
            m_1sig = mean_val - 1.0 * std_dev
            
            p_1sig = mean_val + 1.0 * std_dev
            p_2sig = mean_val + 2.0 * std_dev
            p_3sig = mean_val + 3.0 * std_dev
            
            q50 = np.nanpercentile(sub_val, 50)
            
            stats_list.append({
                "Scheme Profile": cfg["title"],
                "Condition": cond["label"],
                "Total Patches": len(sub_val),
                "Unit": unit_str,
                "Mean/MBE": round(mean_val, 4),
                "SD": round(std_dev, 4),
                "RMSE": round(rmse_val, 4) if is_diff_mode else "-",
                "Min": round(min_val, 4),
                "μ-3σ": round(m_3sig, 4),
                "μ-2σ": round(m_2sig, 4),
                "μ-σ": round(m_1sig, 4),
                "50% Q(Med)": round(q50, 4),
                "μ+σ": round(p_1sig, 4),
                "μ+2σ": round(p_2sig, 4),
                "μ+3σ": round(p_3sig, 4),
                "Max": round(max_val, 4)
            })

        # 地图绘图仍绘制全体(Total)情况，保持全局视场完整
        if is_diff_mode:
            sc = ax.scatter(final_df['patch_lon'], final_df['patch_lat'], 
                            c=base_val, cmap='coolwarm', s=1, vmin=cfg["vmin"], vmax=cfg["vmax"])
            # 在图题中展示全体的 RMSE 和 Bias 表现
            total_rmse = np.sqrt(np.nanmean(base_val ** 2))
            total_bias = np.nanmean(base_val)
            ax.set_title(f"{cfg['title']}\n(Total RMSE: {total_rmse:.4f}{unit_str}, Bias: {total_bias:.4f}{unit_str})")
            cb_label = 'T_eff Difference (K)' if cfg["is_temp"] else 'Reflectivity Difference (-)'
            plt.colorbar(sc, ax=ax, label=cb_label, extend='both')
        else:
            sc = ax.scatter(final_df['patch_lon'], final_df['patch_lat'], 
                            c=base_val, cmap=cfg["cmap"], s=1, vmin=cfg["vmin"], vmax=cfg["vmax"])
            total_mean = np.nanmean(base_val)
            ax.set_title(f"{cfg['title']}\n(Total Mean: {total_mean:.4f} {unit_str})")
            plt.colorbar(sc, ax=ax, label=f'{cfg["title"]} ({unit_str})', extend=cfg.get("extend", "neither"))

        ax.set_ylabel('Latitude')
        ax.set_xlabel('Longitude')

    summary_df = pd.DataFrame(stats_list)
    
    # 调整终端打印列的布局排版
    print("\n" + "="*185)
    print(f" MULTI-CONDITION ERROR DISTRIBUTION & VARIABLE SUMMARY TABLE ({time_format_str})")
    print("="*185)
    print(summary_df.to_string(index=False))
    print("="*185 + "\n")
    
    stats_csv_filename = f"Teff_stats_summary_{time_format_str}.csv"
    summary_df.to_csv(os.path.join(output_dir, stats_csv_filename), index=False)
    print(f"Multi-condition statistics summary table successfully saved to CSV: {stats_csv_filename}")
    
    # plt.tight_layout()
    # plot_filename = f"Teff_diff_map_{time_format_str}.png"
    # plt.savefig(os.path.join(output_dir, plot_filename), dpi=300)
    # plt.close()
    # print(f"Spatial discrepancy profile plot successfully generated: {plot_filename}")

if __name__ == "__main__":
    output_dir = '/home/liusy/research_lists/2026-06-01_research_list/compare_diff_teff/results_try_2'
    t0 = time.time()
    for month in range(6, 9):
        main(2016, month, 1, 0, output_dir=output_dir)
        print(f"Month {month} completed in {time.time()-t0:.2f} seconds")
