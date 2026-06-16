"""
我已经为你重写了网络架构和特征处理逻辑。核心改动包括：

双网络架构：定义了 DielectricPredictor 和 RoughnessPredictor。

特征解耦：将 19 维特征拆解为物理意义明确的两个小子集，分别进行归一化。

联合优化器：使用 itertools.chain 或列表拼接，将两个网络的参数同时交给 Adam 优化。

新增修改：
- 指定单个经纬度处理逻辑。
- 提取并保存各类有效温度及土壤物理参数到CSV。
- 绘制包含时间序列和剖面的多子图并保存。
"""
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt 
import pickle 
import glob
from step_5_1 import process_single_grid  # 确保这个函数返回了我们需要的 result 字典
from rtm import DifferentiableRTM
import sys
from scipy.optimize import curve_fit


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"随机种子已固定为: {seed}")

# =======================================================================
# 提取 M09 经验公式参数计算 (完全基于 rtm.py 的 diel_soil_M09 公式)
# 支持 Batch 和多层同时计算
# =======================================================================


def compute_M09_params(t_celsius, wf_clay_percent):
    """
    计算 Mironov 2009 土壤介电模型的光谱参数
    """
    # --- 统一转为 Tensor，避免标量类型错误 ---
    if not torch.is_tensor(t_celsius):
        t_celsius = torch.tensor(t_celsius, dtype=torch.float32)
    if not torch.is_tensor(wf_clay_percent):
        wf_clay_percent = torch.tensor(wf_clay_percent, dtype=torch.float32)

    ts = torch.tensor(20.0, dtype=torch.float32)
    tk = t_celsius + torch.tensor(273.15, dtype=torch.float32)

    # 1. 干土折射率 (实部/虚部)
    znd = 1.634 - 0.539e-2 * wf_clay_percent + 0.2748e-4 * (wf_clay_percent ** 2)
    zkd = 0.03952 - 0.04038e-2 * wf_clay_percent

    # 2. 最大束缚水体积含量
    zxmvt = 0.02863 + 0.30673e-2 * wf_clay_percent

    # 3. 束缚水静介电常数 (zep0b)
    e0b = 79.8 - 85.4e-2 * wf_clay_percent + 32.7e-4 * (wf_clay_percent ** 2)
    Bb = (8.67e-19 - 0.00126e-2 * wf_clay_percent
          + 0.00184e-4 * (wf_clay_percent ** 2)
          - 9.77e-10 * (wf_clay_percent ** 3)
          - 1.39e-15 * (wf_clay_percent ** 4))
    Fb = torch.log(torch.clamp((e0b - 1.0) / (e0b + 2.0), min=1e-8))
    exp_term_b = torch.exp(Fb - Bb * (t_celsius - ts))
    zep0b = (1.0 + 2.0 * exp_term_b) / (1.0 - exp_term_b)

    # 4. 束缚水弛豫时间 (ztaub)
    dHbR = (1467.0 + 2697e-2 * wf_clay_percent
            - 980e-4 * (wf_clay_percent ** 2)
            + 1.368e-10 * (wf_clay_percent ** 3)
            - 8.61e-13 * (wf_clay_percent ** 4))
    dSbR = (0.888 + 9.7e-2 * wf_clay_percent
            - 4.262e-4 * (wf_clay_percent ** 2)
            + 6.79e-21 * (wf_clay_percent ** 3)
            + 4.263e-22 * (wf_clay_percent ** 4))
    ztaub = 48e-12 * torch.exp(dHbR / tk - dSbR) / tk

    # 5. 束缚水有效电导率 (zsigmab)
    Bsgb = (0.0028 + 0.02094e-2 * wf_clay_percent
            - 0.01229e-4 * (wf_clay_percent ** 2)
            - 5.03e-22 * (wf_clay_percent ** 3)
            + 4.163e-24 * (wf_clay_percent ** 4))
    sigmabt = 0.3112 + 0.467e-2 * wf_clay_percent
    zsigmab = sigmabt + Bsgb * (t_celsius - ts)

    # 6. 自由水静介电常数 (zep0u)
    e0u = torch.tensor(100.0, dtype=torch.float32)
    Bu = (1.11e-4 - 1.603e-7 * wf_clay_percent
          + 1.239e-9 * (wf_clay_percent ** 2)
          + 8.33e-13 * (wf_clay_percent ** 3)
          - 1.007e-14 * (wf_clay_percent ** 4))
    Fu = torch.log(torch.clamp((e0u - 1.0) / (e0u + 2.0), min=1e-8))
    exp_term_u = torch.exp(Fu - Bu * (t_celsius - ts))
    zep0u = (1.0 + 2.0 * exp_term_u) / (1.0 - exp_term_u)

    # 7. 自由水弛豫时间 (ztauu)
    dHuR = (2231.0 - 143.1e-2 * wf_clay_percent
            + 223.2e-4 * (wf_clay_percent ** 2)
            - 142.1e-6 * (wf_clay_percent ** 3)
            + 27.14e-8 * (wf_clay_percent ** 4))
    dSuR = (3.649 - 0.4894e-2 * wf_clay_percent
            + 0.763e-4 * (wf_clay_percent ** 2)
            - 0.4859e-6 * (wf_clay_percent ** 3)
            + 0.0928e-8 * (wf_clay_percent ** 4))
    ztauu = 48e-12 * torch.exp(dHuR / tk - dSuR) / tk

    # 8. 自由水有效电导率 (zsigmau)
    Bsgu = (0.00108 + 0.1413e-2 * wf_clay_percent
            - 0.2555e-4 * (wf_clay_percent ** 2)
            + 0.2147e-6 * (wf_clay_percent ** 3)
            - 0.0711e-8 * (wf_clay_percent ** 4))
    sigmaut = 0.05 + 1.4 * (1.0 - (1.0 - wf_clay_percent * 1e-2) ** 4.664)
    zsigmau = sigmaut + Bsgu * (t_celsius - ts)

    return znd, zkd, zxmvt, zep0b, ztaub, zsigmab, zep0u, ztauu, zsigmau

class DielectricPredictor(nn.Module):
    def __init__(self, num_classes=20, embed_dim=8):
        super(DielectricPredictor, self).__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(6 + embed_dim, 64), 
            nn.LayerNorm(64),
            nn.GELU(), 
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 3) 
        )
        self._init_weights()

    def _init_weights(self):
        last_linear = [layer for layer in self.net if isinstance(layer, nn.Linear)][-1]
        nn.init.xavier_uniform_(last_linear.weight, gain=0.1) 
        nn.init.zeros_(last_linear.bias)

    def forward(self, x, pclass):
        emb = self.embed(pclass)
        x_in = torch.cat([x, emb], dim=-1)
        logits = self.net(x_in)             
        ratios = 0.5 + 1 * torch.sigmoid(logits)
        return ratios, logits

class RoughnessPredictor(nn.Module):
    def __init__(self, num_classes=20, embed_dim=8):
        super(RoughnessPredictor, self).__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(5 + embed_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(), 
            nn.Linear(32, 1) 
        )
        self._init_weights()

    def _init_weights(self):
        last_linear = [layer for layer in self.net if isinstance(layer, nn.Linear)][-1]
        nn.init.xavier_uniform_(last_linear.weight, gain=0.1)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x, pclass):
        emb = self.embed(pclass)
        x_in = torch.cat([x, emb], dim=-1)
        logits = self.net(x_in)
        hr = torch.nn.functional.softplus(logits)
        return hr.squeeze(-1), logits.squeeze(-1)

def fun_holmes(X, b):
    # X[0] 为 eps_r, X[1] 为 eps_i
    eps0_param = 0.08
    eps_ratio = X[1] / X[0]
    C = (eps_ratio / eps0_param) ** b
    return np.clip(C, 0.001, np.inf)

def run_step_5_2_calibration(result, target_patch_idx, patch_lat, patch_lon, output_dir):
    if result is None:
        print("未接收到有效数据，跳过 Step 5.2。")
        return None

    # === 0. 基础配置 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"========= 🚀 开始执行单点测试 (设备: {device}) =========")
    os.makedirs(output_dir, exist_ok=True)
    
    obs = result['obs']
    model_inputs = result['model_inputs']
    time_dim = len(obs['tb_h'])
    patch_dim = len(model_inputs['htop'])
    num_samples = time_dim * patch_dim

    # === 1. 维度扩展 ===
    def expand_static_1d(arr): return np.broadcast_to(arr, (time_dim, patch_dim)).reshape(-1)
    def expand_static_2d(arr, dim3): return np.broadcast_to(arr, (time_dim, patch_dim, dim3)).reshape(-1, dim3)
    def flatten_dynamic_1d(arr): return arr.reshape(-1)
    def flatten_dynamic_2d(arr, dim3): return arr.reshape(-1, dim3)

    patchtype = expand_static_1d(model_inputs['patchtype'])
    patchclass = expand_static_1d(model_inputs['patchclass'])
    forc_topo = expand_static_1d(model_inputs['forc_topo'])
    htop = expand_static_1d(model_inputs['htop'])
    
    soil_dim = model_inputs['wf_clay'].shape[1] if model_inputs['wf_clay'].ndim > 1 else 10
    wf_clay = expand_static_2d(model_inputs['wf_clay'], soil_dim)
    wf_sand = expand_static_2d(model_inputs['wf_sand'], soil_dim)
    wf_silt = expand_static_2d(model_inputs['wf_silt'], soil_dim)
    BD_all = expand_static_2d(model_inputs['BD_all'], soil_dim)
    porsl = expand_static_2d(model_inputs['porsl'], soil_dim)

    tref = flatten_dynamic_1d(model_inputs['tref'])
    tleaf = flatten_dynamic_1d(model_inputs['tleaf'])
    snowdp = flatten_dynamic_1d(model_inputs['snowdp'])
    lai = flatten_dynamic_1d(model_inputs['lai'])
    sai = flatten_dynamic_1d(model_inputs['sai'])
    
    snl_dim = model_inputs['dz_sno'].shape[2]
    dz_sno = flatten_dynamic_2d(model_inputs['dz_sno'], snl_dim)
    
    soisno_dim = model_inputs['t_soisno'].shape[2]
    t_soisno = flatten_dynamic_2d(model_inputs['t_soisno'], soisno_dim)
    wliq_soisno = flatten_dynamic_2d(model_inputs['wliq_soisno'], soisno_dim)
    wice_soisno = flatten_dynamic_2d(model_inputs['wice_soisno'], soisno_dim)
    h2osoi = flatten_dynamic_2d(model_inputs['h2osoi'], soil_dim)
    
    t_brt_smap_h = model_inputs['t_brt_smap_h'].mean(axis=1).reshape(-1)
    t_brt_smap_v = model_inputs['t_brt_smap_v'].mean(axis=1).reshape(-1)

    # === 2. Tensor 转换 ===
    t_patchtype = torch.tensor(patchtype, dtype=torch.long, device=device)
    t_patchclass = torch.tensor(patchclass, dtype=torch.long, device=device)
    t_dz_sno = torch.tensor(dz_sno, dtype=torch.float32, device=device)
    t_forc_topo = torch.tensor(forc_topo, dtype=torch.float32, device=device)
    t_htop = torch.tensor(htop, dtype=torch.float32, device=device)
    t_tref = torch.tensor(tref, dtype=torch.float32, device=device)
    t_t_soisno = torch.tensor(t_soisno, dtype=torch.float32, device=device)
    t_tleaf = torch.tensor(tleaf, dtype=torch.float32, device=device)
    t_wliq_soisno = torch.tensor(wliq_soisno, dtype=torch.float32, device=device)
    t_wice_soisno = torch.tensor(wice_soisno, dtype=torch.float32, device=device)
    t_h2osoi = torch.tensor(h2osoi, dtype=torch.float32, device=device)
    t_snowdp = torch.tensor(snowdp, dtype=torch.float32, device=device)
    t_lai = torch.tensor(lai, dtype=torch.float32, device=device)
    t_sai = torch.tensor(sai, dtype=torch.float32, device=device)
    t_wf_clay = torch.tensor(wf_clay, dtype=torch.float32, device=device)
    t_wf_sand = torch.tensor(wf_sand, dtype=torch.float32, device=device)
    t_wf_silt = torch.tensor(wf_silt, dtype=torch.float32, device=device)
    t_BD_all = torch.tensor(BD_all, dtype=torch.float32, device=device)
    t_porsl = torch.tensor(porsl, dtype=torch.float32, device=device)

    # 1.4 GHz
    t_sat_fghz = torch.full((num_samples,), 1.4, dtype=torch.float32, device=device)
    t_sat_theta = torch.full((num_samples,), 40.0 * np.pi / 180.0, dtype=torch.float32, device=device)

    # === 3. 构建分离与多层特征输入 ===
    maxsnl = 5  
    t_wf_total = t_wf_clay + t_wf_sand + t_wf_silt
    w1, w2 = 0.0175, 0.0276        
    wtot = w1 + w2
    
    # —— 表层参数 ——
    surface_sm =  (t_wliq_soisno[:, maxsnl] + t_wliq_soisno[:, maxsnl+1]) / (wtot * 1000.0)
    surface_t =  ((t_t_soisno[:, maxsnl]*w1 + t_t_soisno[:, maxsnl+1]*w2) / wtot) - 273.15 
    surface_clay = (t_wf_clay[:, 0]/t_wf_total[:, 0]*w1 + t_wf_clay[:, 1]/t_wf_total[:, 1]*w2) / wtot * 100
    surface_sand = (t_wf_sand[:, 0]/t_wf_total[:, 0]*w1 + t_wf_sand[:, 1]/t_wf_total[:, 1]*w2) / wtot * 100
    surface_porsl = (t_porsl[:, 0]/t_wf_total[:, 0]*w1 + t_porsl[:, 1]/t_wf_total[:, 1]*w2) / wtot
    surface_BD_all = (t_BD_all[:, 0]/t_wf_total[:, 0]*w1 + t_BD_all[:, 1]/t_wf_total[:, 1]*w2) / wtot / 1000

    # —— 为深层介电常数准备的 10 层土壤参数 ——
    dz_soi = torch.tensor([0.0175, 0.0276, 0.0455, 0.0750, 0.1236, 0.2038, 0.3360, 0.5539, 0.9133, 1.5058], device=device)
    t_wliq_soi = t_wliq_soisno[:, maxsnl:]
    t_t_soi = t_t_soisno[:, maxsnl:]
    
    sm_all = t_wliq_soi / (dz_soi.unsqueeze(0) * 1000.0)
    t_all = t_t_soi - 273.15
    clay_all = (t_wf_clay / t_wf_total) * 100.0
    
    # === 4. 初始化模型 ===
    rtm_model = DifferentiableRTM(def_da_rtm_diel=4, def_da_rtm_rough=0, def_da_rtm_veg=0, 
                                  num_grids=num_samples, maxsnl=maxsnl).to(device)
    
    # === 计算 M09 先验知识 ===
    znd_M09_surf, zkd_M09_surf, zxmvt_M09_surf, zep0b_M09_surf, ztaub_M09_surf, zsigmab_M09_surf, zep0u_M09_surf, ztauu_M09_surf, zsigmau_M09_surf = compute_M09_params(surface_t, surface_clay)
    znd_M09, zkd_M09, zxmvt_M09, zep0b_M09, ztaub_M09, zsigmab_M09, zep0u_M09, ztauu_M09, zsigmau_M09 = compute_M09_params(t_all, clay_all)
    
    # 跑一次前向模型以获取 T_eff 
    _ = rtm_model(
            t_patchtype, t_patchclass, t_dz_sno, t_forc_topo, t_htop, 
            t_tref, t_t_soisno, t_tleaf, t_wliq_soisno, t_wice_soisno, t_h2osoi, 
            t_snowdp, t_lai, t_sai, 
            t_wf_clay, t_wf_sand, t_wf_silt, t_BD_all, t_porsl, 
            t_sat_theta, t_sat_fghz,
            znd_M09_surf, zkd_M09_surf, zxmvt_M09_surf, zep0b_M09_surf, ztaub_M09_surf, zsigmab_M09_surf, zep0u_M09_surf, ztauu_M09_surf, zsigmau_M09_surf,
            znd_M09, zkd_M09, zxmvt_M09, zep0b_M09, ztaub_M09, zsigmab_M09, zep0u_M09, ztauu_M09, zsigmau_M09,
            torch.full_like(znd_M09_surf, 1.667)
            )

    # === 5. 提取需要的变量 ===
    t_wig = rtm_model.t_eff_wigneron.detach().cpu().numpy()
    t_hol = rtm_model.t_eff_holmes.detach().cpu().numpy()
    
    # 获取 Wilheit
    t_wil_v = rtm_model.t_eff_wilheit.detach().cpu().numpy()
    t_wil_h = t_wil_v 
    
    # 提取其它变量
    t_surf_np = rtm_model.t_surf.detach().cpu().numpy() + 273.15 # 转回K
    t_deep_np = rtm_model.t_deep.detach().cpu().numpy() + 273.15 # 转回K
    
    wc_surf_np = surface_sm.detach().cpu().numpy()
    clay_surf_np = surface_clay.detach().cpu().numpy()
    mvt_surf_np = zxmvt_M09_surf.detach().cpu().numpy()
    sand_surf_np = surface_sand.detach().cpu().numpy()
    bd_surf_np = surface_BD_all.detach().cpu().numpy()
    
    t_soi_np = t_t_soi.detach().cpu().numpy() # 形状为 [N, 10]
    t_soi_max = t_soi_np.max(axis=1)
    t_soi_min = t_soi_np.min(axis=1)
    
    eps_surf_nd = rtm_model.eps_soil_nd.detach().cpu().numpy()
    eps_surf_real = eps_surf_nd.real
    eps_surf_imag = eps_surf_nd.imag
    
    # === 计算 C_holmes2006, C_inverted_wilheit, b_param_inverted ===
    eps0_param = 0.08
    b_param_def = 0.87
    eps_ratio = eps_surf_imag / eps_surf_real
    C_holmes2006 = (eps_ratio / eps0_param) ** b_param_def
    
    # 取 H 极化和 V 极化的平均值代表整体有效温度进行反演
    t_wil_mean = (t_wil_h + t_wil_v) / 2.0
    denom = t_surf_np - t_deep_np
    
    # 处理分母极小的情况，避免除以0或产生极端值
    valid_idx = np.abs(denom) > 0.1
    C_inverted_wilheit = np.zeros_like(t_wil_mean)
    C_inverted_wilheit[valid_idx] = (t_wil_mean[valid_idx] - t_deep_np[valid_idx]) / denom[valid_idx]
    C_inverted_wilheit = np.clip(C_inverted_wilheit, 0.001, 1.0)
    
    # 尝试拟合新的 b 值
    try:
        popt, _ = curve_fit(fun_holmes, (eps_surf_real, eps_surf_imag), C_inverted_wilheit, p0=[0.87])
        b_param_inverted = np.full_like(C_inverted_wilheit, popt[0])
    except:
        b_param_inverted = np.full_like(C_inverted_wilheit, np.nan)

    # === 组装 DataFrame ===
    # 生成一个循环的 patch_id 序列，例如有3个patch，就是 [0, 1, 2, 0, 1, 2, ...]
    patch_indices = np.tile(np.arange(patch_dim), time_dim)
    
    df_out = pd.DataFrame({
        "date": np.repeat(obs['date'], patch_dim),
        "patch_id": patch_indices, 
        "patch_lon": patch_lon,
        "patch_lat": patch_lat,
        "patchclass": patchclass,
        "T_eff_wilheit_H": t_wil_h,
        "T_eff_wilheit_V": t_wil_v,
        "T_eff_wigneron": t_wig,
        "T_eff_holmes2006": t_hol,
        "t_surf": t_surf_np,
        "wc_surf": wc_surf_np,
        "clay_surf": clay_surf_np,
        "mvt_surf": mvt_surf_np,
        "sand_surf": sand_surf_np,
        "bd_surf": bd_surf_np,
        "t_deep": t_deep_np,
        "t_soi_max": t_soi_max,
        "t_soi_min": t_soi_min,
        "eps_surf_real": eps_surf_real,
        "eps_surf_imag": eps_surf_imag,
        "C_holmes2006": C_holmes2006,
        "C_inverted_wilheit": C_inverted_wilheit,
        "b_param_inverted": b_param_inverted
    })
    
    # 补充 10 层土壤温度
    for i in range(10):
        df_out[f"t_soi_layer_{i+1}"] = t_soi_np[:, i]
        
    # ================= 核心修改：只过滤出我们需要的目标 Patch =================
    df_target = df_out[df_out['patch_id'] == target_patch_idx].copy()
    
    # 按时间排序，确保画图时间轴是对的
    df_target['date'] = pd.to_datetime(df_target['date'])
    df_target = df_target.sort_values('date')
    
    # 按照你的要求命名文件
    out_csv_path = os.path.join(output_dir, f"patch_lon_{patch_lon:.6f}_patch_lat_{patch_lat:.6f}_Teff_infos.csv")
    df_target.to_csv(out_csv_path, index=False, float_format='%.6f')
    print(f"💾 Patch 数据已成功保存至: {out_csv_path}")

    # === 计算指标 (RMSE 和 Bias) ===
    # 我们以 Wilheit 的 V极化的值 作为参考基准
    t_wil_ref = df_target['T_eff_wilheit_V']
    
    rmse_hol = np.sqrt(((df_target['T_eff_holmes2006'] - t_wil_ref) ** 2).mean())
    bias_hol = (df_target['T_eff_holmes2006'] - t_wil_ref).mean()
    
    rmse_wig = np.sqrt(((df_target['T_eff_wigneron'] - t_wil_ref) ** 2).mean())
    bias_wig = (df_target['T_eff_wigneron'] - t_wil_ref).mean()

    # === 绘制图片 ===
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # 图 1: 有效温度比较
    axs[0].plot(df_target['date'], df_target['T_eff_wilheit_H'], label='Wilheit', color='blue', alpha=0.7)
    axs[0].plot(df_target['date'], df_target['T_eff_holmes2006'], label='Holmes 2006', color='red', linestyle='--', alpha=0.7)
    axs[0].plot(df_target['date'], df_target['T_eff_wigneron'], label='Wigneron 2001', color='green', linestyle=':', alpha=0.7)
    
    # 在图 1 中加入 RMSE 和 Bias 文本框
    metrics_text = (f"Holmes vs Wilheit: RMSE={rmse_hol:.2f} K, Bias={bias_hol:.2f} K\n"
                    f"Wigneron vs Wilheit: RMSE={rmse_wig:.2f} K, Bias={bias_wig:.2f} K")
    axs[0].text(0.02, 0.95, metrics_text, transform=axs[0].transAxes, 
                fontsize=11, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                
    axs[0].set_ylabel('Effective Temperature (K)')
    axs[0].set_title(f'Effective Temperature Comparison (Patch Lon: {patch_lon:.4f}, Lat: {patch_lat:.4f})')
    axs[0].legend(loc='lower right')
    axs[0].grid(True, linestyle=':', alpha=0.6)

    # 图 2: delta_t (t_surf - t_deep)
    delta_t = df_target['t_surf'] - df_target['t_deep']
    axs[1].plot(df_target['date'], delta_t, color='purple')
    axs[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axs[1].set_ylabel('Delta T (K)')
    axs[1].set_title('T_surf - T_deep')
    axs[1].grid(True, linestyle=':', alpha=0.6)

    # 图 3: 表层湿度
    axs[2].plot(df_target['date'], df_target['wc_surf'], color='teal')
    axs[2].set_ylabel('Soil Moisture (m³/m³)')
    axs[2].set_title('Surface Soil Moisture')
    axs[2].grid(True, linestyle=':', alpha=0.6)

    # 图 4: 10 层土壤温度
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    for i in range(10):
        axs[3].plot(df_target['date'], df_target[f't_soi_layer_{i+1}'], color=colors[i], label=f'Layer {i+1}')
    axs[3].set_ylabel('Soil Temperature (K)')
    axs[3].set_title('10-Layer Soil Temperature')
    axs[3].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    axs[3].grid(True, linestyle=':', alpha=0.6)

    axs[3].set_xlabel('Date')
    fig.autofmt_xdate()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"patch_lon_{patch_lon:.6f}_patch_lat_{patch_lat:.6f}_profiles.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 图片已保存至: {plot_path}")
    
    return None

if __name__ == "__main__":
    INDEX_FILE = '/home/liusy/storage_global_veg_wigneron/nc_patch_location_index.csv'
    csv_dir = '/home/liusy/store_global_forward/tb_for_EASE_open_lands'
    patch_map_file = '/home/liusy/storage_global_veg_wigneron/patch_map_EASE_open_lands.csv' 
    nc_dir = '/home/liusy/CoLM/outputs/global_veg_wigneron/forward_inputs_folder'
    
    output_dir="/home/liusy/research_lists/2026-06-01_research_list/compare_diff_teff_for_open_lands/results_try_for_a_patch"
    
    # ======= 设置目标网格和想要提取的 Patch 索引 =======
    TARGET_LAT, TARGET_LON = [31.62478,2.053942]
    TARGET_PATCH_INDEX = 0  # 0表示该网格下的第1个patch，1表示第2个，以此类推
    # ===================================================

    print("正在加载全局 NC 索引表和 Patch 映射表...")
    df_nc_index = pd.read_csv(INDEX_FILE)
    df_patch_map = pd.read_csv(patch_map_file)
    
    set_seed(seed=42)  
    
    # 1. 从 patch_map 里面筛选出属于这个网格的所有 patches
    grid_patches = df_patch_map[(np.isclose(df_patch_map['ease_lat'], TARGET_LAT)) & 
                                (np.isclose(df_patch_map['ease_lon'], TARGET_LON))]
    
    if grid_patches.empty:
        print(f"在 patch_map 中找不到网格 (经度: {TARGET_LON}, 纬度: {TARGET_LAT}) 的斑块信息！")
        sys.exit()
        
    if TARGET_PATCH_INDEX >= len(grid_patches):
        print(f"设定的 TARGET_PATCH_INDEX ({TARGET_PATCH_INDEX}) 超出范围，该网格共有 {len(grid_patches)} 个 patches。")
        sys.exit()

    # 2. 获取目标 patch 的真实经纬度
    patch_lat = grid_patches.iloc[TARGET_PATCH_INDEX]['patch_lat']
    patch_lon = grid_patches.iloc[TARGET_PATCH_INDEX]['patch_lon']
    
    target_csv = os.path.join(csv_dir, f"lat_{TARGET_LAT}_lon_{TARGET_LON}_SMAP_TB.csv")
    
    if not os.path.exists(target_csv):
        print(f"找不到网格 (经度: {TARGET_LON}, 纬度: {TARGET_LAT}) 的观测数据文件。")
        sys.exit()
        
    print(f"\n🚀 正在处理网格: 经度={TARGET_LON}, 纬度={TARGET_LAT}")
    print(f"🎯 锁定目标 Patch (Index={TARGET_PATCH_INDEX}): patch_lon={patch_lon}, patch_lat={patch_lat}")
    
    try:
        df_obs = pd.read_csv(target_csv)
        result = process_single_grid(df_nc_index, df_obs, patch_map_file, nc_dir) 
        
        # 执行计算并输出结果，把 patch 的信息也传进去
        run_step_5_2_calibration(result, TARGET_PATCH_INDEX, patch_lat, patch_lon, output_dir)
            
    except Exception as e:
        print(f"⚠️ 处理时抛出异常: {e}")

    print(f"\n✅ 执行完成！")
