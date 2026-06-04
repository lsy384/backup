import os
import glob
import torch
import datetime
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import calendar
# Import the RTM model containing the added Teff schemes
from rtm import DifferentiableRTM
import time

def main(target_year, target_month, target_day, target_hour, output_dir):
    # ==========================================
    # TIME SETTING (Modify here for target slice)
    # ==========================================

    target_dt = datetime.datetime(target_year, target_month, target_day, target_hour)
    start_of_year = datetime.datetime(target_year, 1, 1)
    time_idx = int((target_dt - start_of_year).total_seconds() // 3600)
    
    # Strictly format time representation as requested (YYYY-MM-DD-HH)
    time_format_str = f"{target_year:04d}-{target_month:02d}-{target_day:02d}-{target_hour:02d}"
    print(f"Target Datetime: {target_dt} | Time Index in dataset: {time_idx}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rtm_model = DifferentiableRTM().to(device)

    # Satellite and physics configurations (1.4 GHz)
    sat_fghz = 1.4 
    sat_theta = 40.0 * np.pi / 180.0
    f = sat_fghz * 1e9
    lam = rtm_model.C / f
    lamcm = lam * 100.0

    nc_dir = '/home/liusy/CoLM/outputs/global_veg_wigneron/forward_inputs_folder'
    nc_files = glob.glob(os.path.join(nc_dir, 'forward_inputs_worker*.nc'))
    
    all_results = []

    print(f"Found {len(nc_files)} NetCDF files. Processing...")

    for file_path in nc_files:
        try:
            dataset = nc.Dataset(file_path, 'r')
            
            # Extract static variables
            lon = dataset.variables['lon'][:]
            lat = dataset.variables['lat'][:]
            patchclass = dataset.variables['patchclass'][:]
            patchtype = dataset.variables['patchtype'][:]
            
            # Filter non-water patches
            valid_mask = patchtype < 3
            if not np.any(valid_mask):
                dataset.close()
                continue
                
            lon = lon[valid_mask]
            lat = lat[valid_mask]
            patchclass = patchclass[valid_mask]
            
            t_soisno = dataset.variables['t_soisno'][time_idx, valid_mask, :]
            wliq_soisno = dataset.variables['wliq_soisno'][time_idx, valid_mask, :]
            wf_clay = dataset.variables['wf_clay'][valid_mask, :]
            
            # 尝试读取容重，若当前NC文件无该变量则给一个典型的默认值 1400.0 kg/m3
            if 'BD_all' in dataset.variables:
                bd_all = dataset.variables['BD_all'][valid_mask, :]
            elif 'bden_soi' in dataset.variables:
                bd_all = dataset.variables['bden_soi'][valid_mask, :]
            else:
                bd_all = np.full_like(wf_clay, 1400.0) 
                
            dataset.close()

            t_soi = torch.tensor(t_soisno[:, 5:], dtype=torch.float32, device=device)
            wliq_soi = torch.tensor(wliq_soisno[:, 5:], dtype=torch.float32, device=device)
            clay_all = torch.tensor(wf_clay, dtype=torch.float32, device=device)
            bd_all_tensor = torch.tensor(bd_all, dtype=torch.float32, device=device)

            dz_soi = rtm_model.dz_soi.unsqueeze(0).expand(t_soi.shape[0], -1)
            wc_all = wliq_soi / (dz_soi * 100.0)
            
            # Calculate surface and deep parameters
            wtot = 0.0175 + 0.0276
            t_surf = ((t_soi[:, 0]*0.0175 + t_soi[:, 1]*0.0276) / wtot)
            wc_surf = (wliq_soi[:, 0] + wliq_soi[:, 1]) / (wtot * 1000.0)
            t_deep = ((t_soi[:, 6]*(0.8289-0.5) + t_soi[:, 7]*(1.0-0.8289)) / 0.5)
            
            # 【新增】为 Wigneron 2008 准备物理参数
            # 表层参数准备
            clay_surf_pct = (clay_all[:, 0]*0.0175 + clay_all[:, 1]*0.0276) / wtot  # 黏土百分比 %
            clay_surf = clay_surf_pct / 100.0  # Fraction (专门给 Wigneron 2008 用)
            bd_surf = (bd_all_tensor[:, 0]*0.0175 + bd_all_tensor[:, 1]*0.0276) / wtot / 1000.0
            # 计算各层介电常数
            eps = rtm_model.diel_soil_M09(wc_all, t_soi - rtm_model.tfrz, clay_all, f)
            # 计算表层（第一层和第二层作为整体）介电常数
            eps_surf = rtm_model.diel_soil_M09(wc_surf, t_surf - rtm_model.tfrz, clay_surf_pct, f)
            # Compute the four Teff schemes
            t_lam = torch.full((t_soi.shape[0],), lam, dtype=torch.float32, device=device)
            t_theta = torch.full((t_soi.shape[0],), sat_theta, dtype=torch.float32, device=device)

            # Unpack all variables
            r_h, r_v, teff_wilheit_h, teff_wilheit_v = rtm_model.eff_soil_temp_Wilheit(dz_soi, t_soi, eps, t_theta, lamcm)
            
            # 提取 lv_multi 的权重以计算 90% 穿透深度
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
            
            # Lv Two-layer (多返回 C 值)
            teff_lv_two, C_lv_two = rtm_model.eff_soil_temp_Lv_two(wtot, t_surf, t_deep, eps_surf, t_lam, return_C=True)
            teff_wigneron = rtm_model.eff_soil_temp_Wigneron2001(wc_surf, t_surf, t_deep)
            
            # Holmes 2006 (多返回 C 值)
            teff_holmes2006, C_holmes2006 = rtm_model.eff_soil_temp_Holmes2006(eps_surf, t_surf, t_deep, return_C=True)
            teff_wigneron2008 = rtm_model.eff_soil_temp_Wigneron2008(wc_surf, t_surf, t_deep, clay_surf, bd_surf)

            df_batch = pd.DataFrame({
                'patch_lon': lon,
                'patch_lat': lat,
                'patchclass': patchclass,
                'r_H_wilheit': r_h.cpu().numpy(),
                'r_V_wilheit': r_v.cpu().numpy(),
                'T_eff_wilheit_H': teff_wilheit_h.cpu().numpy(),
                'T_eff_wilheit_V': teff_wilheit_v.cpu().numpy(),
                'T_eff_lv_multi': teff_lv_multi.cpu().numpy(),
                'T_eff_lv_two': teff_lv_two.cpu().numpy(),
                'T_eff_wigneron': teff_wigneron.cpu().numpy(),
                'T_eff_holmes2006': teff_holmes2006.cpu().numpy(),       
                'T_eff_wigneron2008': teff_wigneron2008.cpu().numpy(),   
                'depth_90_lv_multi': depth_90.cpu().numpy(),            # 【保留】
                'C_lv_two': C_lv_two.cpu().numpy(),                     # 【新增】
                'C_holmes2006': C_holmes2006.cpu().numpy()              # 【新增】
            })
            all_results.append(df_batch)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    final_df = pd.concat(all_results, ignore_index=True)
    
    # Save CSV using the specified format YYYY-MM-DD-HH
    csv_filename = f"Teff_compare_{time_format_str}.csv"
    final_df.to_csv(os.path.join(output_dir, csv_filename), index=False)
    print(f"Calculation finished successfully. Saved data table to: {csv_filename}\n")

    # ==================================================
    # GRID PLOT LAYOUT (4 rows, 2 columns)
    # ==================================================
    fig, axs = plt.subplots(4, 2, figsize=(20, 24))
    plot_configs = [
        # 1. 保留的前5副差异图（针对 Wilheit H），统一范围 -20 到 20
        {"col": "T_eff_lv_multi", "ref": "T_eff_wilheit_H", "title": "Lv Multi-layer - Wilheit (H)", "type": "diff", "is_temp": True, "vmin": -20.0, "vmax": 20.0},
        {"col": "T_eff_lv_two", "ref": "T_eff_wilheit_H", "title": "Lv Two-layer - Wilheit (H)", "type": "diff", "is_temp": True, "vmin": -20.0, "vmax": 20.0},
        {"col": "T_eff_wigneron", "ref": "T_eff_wilheit_H", "title": "Wigneron 2001 - Wilheit (H)", "type": "diff", "is_temp": True, "vmin": -20.0, "vmax": 20.0},
        {"col": "T_eff_wigneron2008", "ref": "T_eff_wilheit_H", "title": "Wigneron 2008 - Wilheit (H)", "type": "diff", "is_temp": True, "vmin": -20.0, "vmax": 20.0},
        {"col": "T_eff_holmes2006", "ref": "T_eff_wilheit_H", "title": "Holmes 2006 - Wilheit (H)", "type": "diff", "is_temp": True, "vmin": -20.0, "vmax": 20.0},
        
        # 2. 新增的3副绝对数值图
        # 第六副：深度，平均0.0628，极值到1。设置 vmax=0.2 搭配 extend='max' 显示最佳。
        {"col": "depth_90_lv_multi", "title": "90% Penetration Depth (Lv Multi)", "type": "value", "unit": "m", "cmap": "viridis", "vmin": 0.0, "vmax": 0.2, "extend": "max"},
        
        # 第七、第八副：C值 0到1
        {"col": "C_holmes2006", "title": "C Value (Holmes 2006)", "type": "value", "unit": "-", "cmap": "plasma", "vmin": 0.0, "vmax": 1.0, "extend": "neither"},
        {"col": "C_lv_two", "title": "C Value (Lv Two-layer)", "type": "value", "unit": "-", "cmap": "plasma", "vmin": 0.0, "vmax": 1.0, "extend": "neither"}
    ]

    stats_list = []

    for i, cfg in enumerate(plot_configs):
        row_idx = i // 2
        col_idx = i % 2
        ax = axs[row_idx, col_idx]
        
        if cfg["type"] == "diff":
            val = final_df[cfg["col"]] - final_df[cfg["ref"]]
            mean_val = val.mean()
            std_dev = val.std()
            rmse_val = np.sqrt((val ** 2).mean())
            min_val = val.min()
            max_val = val.max()
            unit_str = "K" if cfg["is_temp"] else "-"
            
            stats_list.append({
                "Scheme Profile": cfg["title"],
                "Total Patches": len(val),
                "Unit": unit_str,
                "Mean/MBE": round(mean_val, 4),
                "SD": round(std_dev, 4),
                "RMSE": round(rmse_val, 4),
                "Min": round(min_val, 4),
                "Max": round(max_val, 4)
            })
            
            sc = ax.scatter(final_df['patch_lon'], final_df['patch_lat'], 
                            c=val, cmap='coolwarm', s=1, vmin=cfg["vmin"], vmax=cfg["vmax"])
            ax.set_title(f"{cfg['title']}\n(RMSE: {rmse_val:.4f}{unit_str}, Bias: {mean_val:.4f}{unit_str})")
            cb_label = 'T_eff Difference (K)' if cfg["is_temp"] else 'Reflectivity Difference (-)'
            # 对称误差图加上 extend='both' 显示超出 -20 和 20 的极值
            plt.colorbar(sc, ax=ax, label=cb_label, extend='both')
            
        else:
            # 对于绝对数值的物理量（深度，C值）
            val = final_df[cfg["col"]]
            mean_val = val.mean()
            std_dev = val.std()
            min_val = val.min()
            max_val = val.max()
            unit_str = cfg["unit"]
            
            stats_list.append({
                "Scheme Profile": cfg["title"],
                "Total Patches": len(val),
                "Unit": unit_str,
                "Mean/MBE": round(mean_val, 4),
                "SD": round(std_dev, 4),
                "RMSE": "-",
                "Min": round(min_val, 4),
                "Max": round(max_val, 4)
            })
            
            sc = ax.scatter(final_df['patch_lon'], final_df['patch_lat'], 
                            c=val, cmap=cfg["cmap"], s=1, vmin=cfg["vmin"], vmax=cfg["vmax"])
            ax.set_title(f"{cfg['title']}\n(Mean: {mean_val:.4f} {unit_str})")
            plt.colorbar(sc, ax=ax, label=f'{cfg["title"]} ({unit_str})', extend=cfg.get("extend", "neither"))

        ax.set_ylabel('Latitude')
        ax.set_xlabel('Longitude')

    # 转换为统计 DataFrame
    summary_df = pd.DataFrame(stats_list)
    
    # 终端打印展示
    print("\n" + "="*85)
    print(f" ERROR DISTRIBUTION & VARIABLE SUMMARY TABLE ({time_format_str})")
    print("="*85)
    print(summary_df.to_string(index=False))
    print("="*85 + "\n")
    
    # 保存误差统计结果
    stats_csv_filename = f"Teff_stats_summary_{time_format_str}.csv"
    summary_df.to_csv(os.path.join(output_dir, stats_csv_filename), index=False)
    print(f"Statistics summary table successfully saved to CSV: {stats_csv_filename}")

    plt.tight_layout()
    plot_filename = f"Teff_diff_map_{time_format_str}.png"
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300)
    plt.close()
    print(f"Spatial discrepancy profile plot successfully generated: {plot_filename}")

if __name__ == "__main__":
    
    output_dir = '/home/liusy/research_lists/2026-06-01_research_list/compare_diff_teff/results_try_1'
    t0=time.time()
    for month in range(1, 13):
        main(2016, month, 1, 0, output_dir=output_dir)
        print(f"Month {month} completed in {time.time()-t0:.2f} seconds")
        # # 获取 2016 年该月的天数
        # _, days_in_month = calendar.monthrange(2016, month)
        # for day in range(1, days_in_month + 1):
        #     for hour in range(24):
        #         main(2016, month, day, hour, output_dir=output_dir)
