import os
import glob
import pandas as pd
import numpy as np
import netCDF4 as nc



def process_single_grid(df_nc_index, df_obs, patch_map_file, nc_dir):

    
    print('len(df_obs)有效时间步数量 =', len(df_obs))
    grid_lon = df_obs['lon'].iloc[0]
    grid_lat = df_obs['lat'].iloc[0]
    
    tb_h = df_obs['tb_h'].values
    tb_v = df_obs['tb_v'].values
    date = df_obs['date'].values
    time_index = df_obs['time_index'].values.astype(int)
    len_eff_obs = len(time_index)

    # 2. 在 patch_map 中找到对应的 patch 经纬度
    df_map = pd.read_csv(patch_map_file)
    mask_map = np.isclose(df_map['ease_lon'], grid_lon, atol=1e-5) & \
               np.isclose(df_map['ease_lat'], grid_lat, atol=1e-5)
    target_patches = df_map[mask_map]
    
    if target_patches.empty:
        # print("未找到对应的 patch。")
        return
        
    patch_lons_target = target_patches['patch_lon'].values
    patch_lats_target = target_patches['patch_lat'].values
    # print(f"该网格包含 {len(patch_lons_target)} 个 patch。")

    # 3. 🎯 核心优化：查表定位，不再盲搜
    file_to_indices = {}
    
    for t_lon, t_lat in zip(patch_lons_target, patch_lats_target):
        match = df_nc_index[np.isclose(df_nc_index['patch_lon'], t_lon, atol=1e-5) & 
                            np.isclose(df_nc_index['patch_lat'], t_lat, atol=1e-5)]
        
        if not match.empty:
            fname = match.iloc[0]['nc_filename']
            idx = int(match.iloc[0]['patch_idx'])
            if fname not in file_to_indices:
                file_to_indices[fname] = []
            file_to_indices[fname].append(idx)

    # 初始化存储容器
    extracted_data = {
        'patchtype': [], 'patchclass': [], 'forc_topo': [], 'htop': [],
        'wf_clay': [], 'wf_sand': [], 'wf_silt': [], 'BD_all': [], 'porsl': [],
        'tref': [], 'tleaf': [], 'snowdp': [], 'lai': [], 'sai': [],
        'dz_sno': [], 't_soisno': [], 'wliq_soisno': [], 'wice_soisno': [],
        'h2osoi': [], 't_brt_smap_h': [], 't_brt_smap_v': []
    }

    # 4. 精准读取数据
    for fname, indices in file_to_indices.items():
        nc_file_path = os.path.join(nc_dir, fname)
        # print(f" 🎯 命中目标文件: {fname}, 需要提取 {len(indices)} 个 patch...")
        try:
            ds = nc.Dataset(nc_file_path, 'r')
            indices = np.sort(np.array(indices))
            
            # === 静态变量 ===
            for var in ['patchtype', 'patchclass', 'forc_topo', 'htop']:
                extracted_data[var].append(ds.variables[var][indices])
            for var in ['wf_clay', 'wf_sand', 'wf_silt', 'BD_all', 'porsl']:
                extracted_data[var].append(ds.variables[var][indices, :])
            
            # === 动态变量 ===
            t_idx = time_index.tolist()
            
            for var in ['tref', 'tleaf', 'snowdp', 'lai', 'sai']:
                data = ds.variables[var][t_idx, indices]
                extracted_data[var].append(data)
                
            for var in ['dz_sno', 't_soisno', 'wliq_soisno', 'wice_soisno', 'h2osoi']:
                data = ds.variables[var][t_idx, indices, :]
                extracted_data[var].append(data)
                
            t_brt = ds.variables['t_brt_smap'][t_idx, indices, :]
            extracted_data['t_brt_smap_h'].append(t_brt[:, :, 0])
            extracted_data['t_brt_smap_v'].append(t_brt[:, :, 1])
            
            ds.close()
            # print(f"   ✅ {fname} 提取完成")
                
        except Exception as e:
            # print(f"读取文件 {fname} 时发生错误: {e}")
            continue

    # ==========================================
    # 5. 跨文件数据合并 (Concatenate)
    # ==========================================
    # print("\n" + "="*50)
    # print("开始进行跨文件数据合并...")
    
    # 静态变量在 patch 维度 (axis=0) 合并
    static_vars = ['patchtype', 'patchclass', 'forc_topo', 'htop', 
                   'wf_clay', 'wf_sand', 'wf_silt', 'BD_all', 'porsl']
    for var in static_vars:
        if extracted_data[var]:
            extracted_data[var] = np.concatenate(extracted_data[var], axis=0)

    # 动态变量在 patch 维度 (axis=1) 合并，因为第一维是 time
    dynamic_vars = ['tref', 'tleaf', 'snowdp', 'lai', 'sai', 't_brt_smap_h', 't_brt_smap_v',
                    'dz_sno', 't_soisno', 'wliq_soisno', 'wice_soisno', 'h2osoi']
    for var in dynamic_vars:
        if extracted_data[var]:
            extracted_data[var] = np.concatenate(extracted_data[var], axis=1)


    # ==========================================
    # 6. 处理 Snowdp，进行数据筛选
    # ==========================================
    # 检查是否存在数据
    if not isinstance(extracted_data['snowdp'], np.ndarray):
        # print("未提取到有效数据，退出。")
        return None

    # 条件：如果某一个 time 维度中任意一个 patch 的 snowdp > 0.01，则标记该 time 需要被丢弃
    has_snow_mask = np.any(extracted_data['snowdp'] > 0.01, axis=1)
    
    # 保留没有雪的时间步
    valid_time_mask = ~has_snow_mask
    len_eff_obs_without_snow = np.sum(valid_time_mask)
    # print(f"初始有效观测时间: {len_eff_obs} -> 去雪后有效观测时间: {len_eff_obs_without_snow}")

    # 如果所有数据均被过滤
    if len_eff_obs_without_snow == 0:
        # print("所有时间步均受冰雪影响，跳过此格点。")
        return None

    # 8. 将过滤掩码(Mask)应用到所有时间维度（包含观测和模拟数据）
    # 更新观测数据
    tb_h_final = tb_h[valid_time_mask]
    tb_v_final = tb_v[valid_time_mask]
    date_final = date[valid_time_mask]
    time_index_final = time_index[valid_time_mask]
    
    # 更新提取的模型输出数据 (只有 dynamic_vars 有时间维度)
    for var in dynamic_vars:
        extracted_data[var] = extracted_data[var][valid_time_mask]

    # print(f"成功处理并过滤完毕。最终有效时间长度为: {len_eff_obs_without_snow}。")
    # print("--- 最终合并后数据形状明细 ---")
    # for var_name, data_array in extracted_data.items():
    #     if isinstance(data_array, np.ndarray):
    #         print(f"变量: {var_name:15} | 最终形状: {data_array.shape}")
    # print("="*50 + "\n")
    
    result = {
        'obs': {
            'tb_h': tb_h_final, 'tb_v': tb_v_final, 
            'date': date_final, 'time_index': time_index_final
        },
        'model_inputs': extracted_data
    }
    
    return result

# if __name__ == "__main__":
#     result = process_single_grid()
