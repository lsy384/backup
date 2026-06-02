import os
import glob
import torch
import datetime
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt

# Import the RTM model containing the added Teff schemes
from rtm import DifferentiableRTM

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
            
            # Extract dynamic and depth-dependent variables
            t_soisno = dataset.variables['t_soisno'][time_idx, valid_mask, :]
            wliq_soisno = dataset.variables['wliq_soisno'][time_idx, valid_mask, :]
            wf_clay = dataset.variables['wf_clay'][valid_mask, :]
            
            dataset.close()

            t_soi = torch.tensor(t_soisno[:, 5:], dtype=torch.float32, device=device)
            wliq_soi = torch.tensor(wliq_soisno[:, 5:], dtype=torch.float32, device=device)
            clay_all = torch.tensor(wf_clay, dtype=torch.float32, device=device)

            dz_soi = rtm_model.dz_soi.unsqueeze(0).expand(t_soi.shape[0], -1)
            wc_all = wliq_soi / (dz_soi * 100.0)
            
            # Calculate surface and deep parameters
            wtot = 0.0175 + 0.0276
            t_surf = ((t_soi[:, 0]*0.0175 + t_soi[:, 1]*0.0276) / wtot)
            wc_surf = (wliq_soi[:, 0] + wliq_soi[:, 1]) / (wtot * 1000.0)
            t_deep = ((t_soi[:, 6]*(0.8289-0.5) + t_soi[:, 7]*(1.0-0.8289)) / 0.5)

            # Compute complex dielectric constant using stable M09
            eps = rtm_model.diel_soil_M09(wc_all, t_soi - rtm_model.tfrz, clay_all, f)

            # Compute the four Teff schemes
            t_lam = torch.full((t_soi.shape[0],), lam, dtype=torch.float32, device=device)
            t_theta = torch.full((t_soi.shape[0],), sat_theta, dtype=torch.float32, device=device)

            # Unpack all 4 variables from coherent Wilheit model
            r_h, r_v, teff_wilheit_h, teff_wilheit_v = rtm_model.eff_soil_temp_Wilheit(dz_soi, t_soi, eps, t_theta, lamcm)
            
            teff_lv_multi = rtm_model.eff_soil_temp_Lv_multi(dz_soi, t_soi, eps, t_lam)
            teff_lv_two = rtm_model.eff_soil_temp_Lv_two(dz_soi, t_soi, eps, t_lam)
            teff_wigneron = rtm_model.eff_soil_temp_Wigneron2001(wc_surf, t_surf, t_deep)

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
                'T_eff_wigneron': teff_wigneron.cpu().numpy()
            })
            all_results.append(df_batch)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    final_df = pd.concat(all_results, ignore_index=True)
    
    # Save CSV using the specified format YYYY-MM-DD-HH
    csv_filename = f"Teff_compare_{time_format_str}.csv"
    final_df.to_csv(os.path.join(output_dir, csv_filename), index=False)
    print(f"Calculation finished successfully. Saved data table to: {csv_filename}\n")

    # Expanded to 4 rows and 2 columns subplot layout
    fig, axs = plt.subplots(4, 2, figsize=(16, 22))
    
    plot_configs = [
        {"col": "T_eff_lv_multi", "ref": "T_eff_wilheit_H", "title": "Lv Multi-layer - Wilheit (H)", "row_idx": 0, "col_idx": 0, "is_temp": True, "is_unified": True},
        {"col": "T_eff_lv_multi", "ref": "T_eff_wilheit_V", "title": "Lv Multi-layer - Wilheit (V)", "row_idx": 0, "col_idx": 1, "is_temp": True, "is_unified": True},
        {"col": "T_eff_lv_two", "ref": "T_eff_wilheit_H", "title": "Lv Two-layer - Wilheit (H)", "row_idx": 1, "col_idx": 0, "is_temp": True, "is_unified": True},
        {"col": "T_eff_lv_two", "ref": "T_eff_wilheit_V", "title": "Lv Two-layer - Wilheit (V)", "row_idx": 1, "col_idx": 1, "is_temp": True, "is_unified": True},
        {"col": "T_eff_wigneron", "ref": "T_eff_wilheit_H", "title": "Wigneron 2001 - Wilheit (H)", "row_idx": 2, "col_idx": 0, "is_temp": True, "is_unified": True},
        {"col": "T_eff_wigneron", "ref": "T_eff_wilheit_V", "title": "Wigneron 2001 - Wilheit (V)", "row_idx": 2, "col_idx": 1, "is_temp": True, "is_unified": True},
        {"col": "T_eff_wilheit_H", "ref": "T_eff_wilheit_V", "title": "Wilheit (H) - Wilheit (V)", "row_idx": 3, "col_idx": 0, "is_temp": True, "is_unified": False},
        {"col": "r_H_wilheit", "ref": "r_V_wilheit", "title": "Wilheit r_H - Wilheit r_V", "row_idx": 3, "col_idx": 1, "is_temp": False, "is_unified": False}
    ]

    # Pre-calculate unified range for plots 1-6
    global_max_abs = 0.0
    for cfg in plot_configs:
        if cfg["is_unified"]:
            diff = final_df[cfg["col"]] - final_df[cfg["ref"]]
            max_abs = max(abs(diff.min()), abs(diff.max()))
            if max_abs > global_max_abs:
                global_max_abs = max_abs
    if global_max_abs == 0:
        global_max_abs = 1e-4

    # ==================================================
    # CONSTRUCT SUMMARY TABLE & PLOT
    # ==================================================
    stats_list = []

    for cfg in plot_configs:
        diff = final_df[cfg["col"]] - final_df[cfg["ref"]]
        
        mean_bias = diff.mean()
        std_dev = diff.std()
        rmse_val = np.sqrt((diff ** 2).mean())
        min_diff = diff.min()
        max_diff = diff.max()
        unit_str = "K" if cfg["is_temp"] else "-"
        
        # 将单位作为独立列，使 CSV 结构更为规整
        stats_list.append({
            "Scheme Profile": cfg["title"],
            "Total Patches": len(diff),
            "Unit": unit_str,
            "MBE": round(mean_bias, 4),
            "SD": round(std_dev, 4),
            "RMSE": round(rmse_val, 4),
            "Min Diff": round(min_diff, 4),
            "Max Diff": round(max_diff, 4)
        })
        
        # 绘图逻辑
        ax = axs[cfg["row_idx"], cfg["col_idx"]]
        vmin, vmax = (-global_max_abs, global_max_abs) if cfg["is_unified"] else (-max(abs(min_diff), abs(max_diff)) or -1e-4, max(abs(min_diff), abs(max_diff)) or 1e-4)
            
        sc = ax.scatter(final_df['patch_lon'], final_df['patch_lat'], 
                        c=diff, cmap='coolwarm', s=1, vmin=-20, vmax=20)
        
        ax.set_title(f"{cfg['title']}\n(RMSE: {rmse_val:.4f}{unit_str}, Bias: {mean_bias:.4f}{unit_str})")
        ax.set_ylabel('Latitude')
        ax.set_xlabel('Longitude')
        cb_label = 'T_eff Difference (K)' if cfg["is_temp"] else 'Reflectivity Difference (-)'
        plt.colorbar(sc, ax=ax, label=cb_label)

    # 转换为统计 DataFrame
    summary_df = pd.DataFrame(stats_list)
    
    # 终端打印展示
    print("\n" + "="*80)
    print(f" ERROR DISTRIBUTION SUMMARY TABLE ({time_format_str})")
    print("="*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")
    
    # 修改处：将误差统计结果保存为标准的 CSV 文件而不再是 XLSX
    stats_csv_filename = f"Teff_stats_summary_{time_format_str}.csv"
    summary_df.to_csv(os.path.join(output_dir, stats_csv_filename), index=False)
    print(f"Statistics summary table successfully saved to CSV: {stats_csv_filename}")

    plt.tight_layout()
    plot_filename = f"Teff_diff_map_{time_format_str}.png"
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300)
    plt.close()
    print(f"Spatial discrepancy profile plot successfully generated: {plot_filename}")

if __name__ == "__main__":
    
    for m in range(1, 13):
        main(2016, m, 1, 0, 
            output_dir='/home/liusy/research_lists/2026-06-01_research_list/compare_diff_teff/results')
