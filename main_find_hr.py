"""
我已经为你重写了网络架构和特征处理逻辑。核心改动包括：

双网络架构：定义了 DielectricPredictor 和 RoughnessPredictor。

特征解耦：将 19 维特征拆解为物理意义明确的两个小子集，分别进行归一化。

联合优化器：使用 itertools.chain 或列表拼接，将两个网络的参数同时交给 Adam 优化。
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
    Args:
        t_celsius       : 土壤温度 (摄氏度) – 标量或张量
        wf_clay_percent : 粘土重量百分比 (例如 30 表示 30%) – 标量或张量
    Returns:
        znd, zkd, zxmvt, zep0b, ztaub, zsigmab, zep0u, ztauu, zsigmau
    """
    # --- 统一转为 Tensor，避免标量类型错误 ---
    if not torch.is_tensor(t_celsius):
        t_celsius = torch.tensor(t_celsius, dtype=torch.float32)
    if not torch.is_tensor(wf_clay_percent):
        wf_clay_percent = torch.tensor(wf_clay_percent, dtype=torch.float32)

    # 常量也转为 Tensor，保证所有运算都是张量操作
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
    e0u = torch.tensor(100.0, dtype=torch.float32)          # ★ 关键修复：必须为 Tensor
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

#=======================================================================
# 网络 1：介电常数与结合水参数预测器 (Mironov 机制导向)
# 输入：仅与土壤物理性质相关的 6 个变量 (sm, t, clay, sand, porosity, BD)
#=======================================================================
#=======================================================================
# 网络 1：修改版 （添加 Patchclass Embedding）
#=======================================================================
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
            nn.Linear(256, 3)  # <--- 核心修改：将 7 改为 3
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
        
        # 核心修改：网络不再直接输出物理绝对值，而是输出 0.5 ~ 1.5 的比例 (Ratio)
        # Sigmoid 范围是 [0, 1]，加 0.5 之后变成 [0.5, 1.5]
        # 乘以 0.6 变成 [0, 0.6]，再加上 0.7 变成 [0.7, 1.3]
        ratios = 0.5 + 1 * torch.sigmoid(logits)
        
        return ratios, logits

#=======================================================================
# 网络 2：有效粗糙度预测器 (考虑地形、植被衰减与水分平滑效应)
# 输入：topo, sm, lai, sai, htop (5 个变量) + Patchclass
#=======================================================================
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
        
        # 核心修改：使用 softplus 保证粗糙度非负，且没有上限限制
        hr = torch.nn.functional.softplus(logits)
        return hr.squeeze(-1), logits.squeeze(-1)

def run_step_5_2_calibration(result, ease_lat, ease_lon, output_dir):
    if result is None:
        print("未接收到有效数据，跳过 Step 5.2。")
        return

    # === 0. 基础配置 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========= 🚀 开始执行双网络解耦优化 (设备: {device}) =========")
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

    obs_tb_h = torch.tensor(obs['tb_h'], dtype=torch.float32, device=device)
    obs_tb_v = torch.tensor(obs['tb_v'], dtype=torch.float32, device=device)

    # === 3. 构建分离与多层特征输入 ===
    maxsnl = 5  
    t_wf_total = t_wf_clay + t_wf_sand + t_wf_silt
    w1, w2 = 0.0175, 0.0276        
    wtot = w1 + w2
    
    # —— 专供表层介电常数网络与输出使用的打包表层参数 ——
    surface_sm =  (t_wliq_soisno[:, maxsnl] + t_wliq_soisno[:, maxsnl+1]) / (wtot * 1000.0)
    surface_t =  ((t_t_soisno[:, maxsnl]*w1 + t_t_soisno[:, maxsnl+1]*w2) / wtot) - 273.15 
    surface_clay = (t_wf_clay[:, 0]/t_wf_total[:, 0]*w1 + t_wf_clay[:, 1]/t_wf_total[:, 1]*w2) / wtot * 100
    surface_sand = (t_wf_sand[:, 0]/t_wf_total[:, 0]*w1 + t_wf_sand[:, 1]/t_wf_total[:, 1]*w2) / wtot * 100
    surface_porsl = (t_porsl[:, 0]/t_wf_total[:, 0]*w1 + t_porsl[:, 1]/t_wf_total[:, 1]*w2) / wtot
    surface_BD_all = (t_BD_all[:, 0]/t_wf_total[:, 0]*w1 + t_BD_all[:, 1]/t_wf_total[:, 1]*w2) / wtot / 1000

    x_diel_raw_surf = torch.stack([surface_sm, surface_t, surface_clay, surface_sand, surface_porsl, surface_BD_all], dim=1)
    x_diel_norm_surf = (x_diel_raw_surf - x_diel_raw_surf.mean(dim=0)) / (x_diel_raw_surf.std(dim=0) + 1e-8)

    # —— 为深层介电常数准备的 10 层土壤参数 ——
    dz_soi = torch.tensor([0.0175, 0.0276, 0.0455, 0.0750, 0.1236, 0.2038, 0.3360, 0.5539, 0.9133, 1.5058], device=device)
    t_wliq_soi = t_wliq_soisno[:, maxsnl:]
    t_t_soi = t_t_soisno[:, maxsnl:]
    
    sm_all = t_wliq_soi / (dz_soi.unsqueeze(0) * 1000.0)
    t_all = t_t_soi - 273.15
    clay_all = (t_wf_clay / t_wf_total) * 100.0
    sand_all = (t_wf_sand / t_wf_total) * 100.0
    porsl_all = t_porsl / t_wf_total 
    BD_all_all = (t_BD_all / t_wf_total) / 1000.0
    
    # 组合为 [N, 10, 6] 的全层输入
    x_diel_raw_all = torch.stack([sm_all, t_all, clay_all, sand_all, porsl_all, BD_all_all], dim=-1)
    
    # 展平以便统一做 Normalize 和传入网络: [N*10, 6]
    x_diel_raw_flat = x_diel_raw_all.view(-1, 6)
    x_diel_norm_flat = (x_diel_raw_flat - x_diel_raw_flat.mean(dim=0)) / (x_diel_raw_flat.std(dim=0) + 1e-8)
    
    # 对应的 10 层 Patchclass 平铺: [N*10]
    patchclass_all_flat = t_patchclass.unsqueeze(1).expand(-1, 10).reshape(-1)

    # 特征集 2: 粗糙度网络 (5维) - 仅需表层参数
    x_rough_raw = torch.stack([t_forc_topo, surface_sm, surface_t, surface_porsl, surface_BD_all], dim=1)
    x_rough_norm = (x_rough_raw - x_rough_raw.mean(dim=0)) / (x_rough_raw.std(dim=0) + 1e-8)


    # === 4. 初始化模型与双网络 ===
    rtm_model = DifferentiableRTM(def_da_rtm_diel=4, def_da_rtm_rough=0, def_da_rtm_veg=0, 
                                  num_grids=num_samples, maxsnl=maxsnl).to(device)
    
    net_diel = DielectricPredictor().to(device)
    net_rough = RoughnessPredictor().to(device)
    
    # 将两个网络的参数合并给优化器
    params_to_optimize = list(net_diel.parameters()) + list(net_rough.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=0.01, weight_decay=1e-5) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1000)
    criterion = nn.MSELoss()


    t0 = time.time()
    net_diel.train()
    net_rough.train()
    
    print('t_all',torch.min(t_all), torch.max(t_all), torch.mean(t_all))
    print('surface_t', torch.min(surface_t), torch.max(surface_t), torch.mean(surface_t))
    
    # === 计算 M09 先验知识 (分别针对打包的表层和展开的 10 层) ===
    znd_M09_surf, zkd_M09_surf, zxmvt_M09_surf, zep0b_M09_surf, ztaub_M09_surf, zsigmab_M09_surf, zep0u_M09_surf, ztauu_M09_surf, zsigmau_M09_surf = compute_M09_params(surface_t, surface_clay)
    znd_M09, zkd_M09, zxmvt_M09, zep0b_M09, ztaub_M09, zsigmab_M09, zep0u_M09, ztauu_M09, zsigmau_M09 = compute_M09_params(t_all, clay_all)
    
    print('zep0b_M09:', zep0b_M09)
    print('zep0u_M09:', zep0u_M09, np.shape(zep0b_M09))
    print('zsigmau_M09', zsigmau_M09)
    
    t_brt_smap_h = torch.tensor(t_brt_smap_h, dtype=torch.float32, device=device)
    t_brt_smap_v = torch.tensor(t_brt_smap_v, dtype=torch.float32, device=device)
    # print('初始使用CoLM计算的,H亮温误差:', torch.sqrt(criterion(t_brt_smap_h, obs_tb_h)).item(), 'V亮温误差:', torch.sqrt(criterion(t_brt_smap_v, obs_tb_v)).item())

    tb_toa_h_patch, tb_toa_v_patch = rtm_model(
            t_patchtype, t_patchclass, t_dz_sno, t_forc_topo, t_htop, 
            t_tref, t_t_soisno, t_tleaf, t_wliq_soisno, t_wice_soisno, t_h2osoi, 
            t_snowdp, t_lai, t_sai, 
            t_wf_clay, t_wf_sand, t_wf_silt, t_BD_all, t_porsl, 
            t_sat_theta, t_sat_fghz,
            znd_M09_surf, zkd_M09_surf, zxmvt_M09_surf, zep0b_M09_surf, ztaub_M09_surf, zsigmab_M09_surf, zep0u_M09_surf, ztauu_M09_surf, zsigmau_M09_surf,
            znd_M09, zkd_M09, zxmvt_M09, zep0b_M09, ztaub_M09, zsigmab_M09, zep0u_M09, ztauu_M09, zsigmau_M09,
            torch.full_like(znd_M09_surf, 1.667)
            )
    tb_h_sim = tb_toa_h_patch.view(time_dim, patch_dim).mean(dim=1)
    tb_v_sim = tb_toa_v_patch.view(time_dim, patch_dim).mean(dim=1)
    print(f'RTM计算与CoLM,H亮温误差:', torch.sqrt(criterion(tb_h_sim, t_brt_smap_h)).item(), 'V亮温误差:', torch.sqrt(criterion(tb_v_sim, t_brt_smap_v)).item(),'平均误差:', (torch.sqrt(criterion(tb_h_sim, t_brt_smap_h)).item() + torch.sqrt(criterion(tb_v_sim, t_brt_smap_v)).item())/2.0)
    print(f'RTM计算与观测,H亮温误差:', torch.sqrt(criterion(tb_h_sim, obs_tb_h)).item(), 'V亮温误差:', torch.sqrt(criterion(tb_v_sim, obs_tb_v)).item(),'平均误差:', (torch.sqrt(criterion(tb_h_sim, obs_tb_h)).item() + torch.sqrt(criterion(tb_v_sim, obs_tb_v)).item())/2.0  )

    sys.exit()

    tb_h_sim = tb_toa_h_patch.view(time_dim, patch_dim).mean(dim=1)
    tb_v_sim = tb_toa_v_patch.view(time_dim, patch_dim).mean(dim=1)
        
    loss_mse = torch.sqrt(nn.MSELoss()(
            torch.cat((tb_h_sim.unsqueeze(1), tb_v_sim.unsqueeze(1)), dim=1),
            torch.cat((obs_tb_h.unsqueeze(1), obs_tb_v.unsqueeze(1)), dim=1)
        ))
    print(f"初始 M09 先验计算的亮温误差: {loss_mse.item():.4f} K")  
    
    best_hr, best_loss = float('inf'), float('inf')
    for delta_hr in np.arange(-1.6, 1.6, 0.1):
    # for delta_hr in np.arange(-0.95, -0.75, 0.01):
        tb_toa_h_patch, tb_toa_v_patch = rtm_model(
            t_patchtype, t_patchclass, t_dz_sno, t_forc_topo, t_htop, 
            t_tref, t_t_soisno, t_tleaf, t_wliq_soisno, t_wice_soisno, t_h2osoi, 
            t_snowdp, t_lai, t_sai, 
            t_wf_clay, t_wf_sand, t_wf_silt, t_BD_all, t_porsl, 
            t_sat_theta, t_sat_fghz,
            znd_M09_surf, zkd_M09_surf, zxmvt_M09_surf, zep0b_M09_surf, ztaub_M09_surf, zsigmab_M09_surf, zep0u_M09_surf, ztauu_M09_surf, zsigmau_M09_surf,
            znd_M09, zkd_M09, zxmvt_M09, zep0b_M09, ztaub_M09, zsigmab_M09, zep0u_M09, ztauu_M09, zsigmau_M09,
            torch.full_like(znd_M09_surf, 1.667)+delta_hr
            )
        tb_h_sim = tb_toa_h_patch.view(time_dim, patch_dim).mean(dim=1)
        tb_v_sim = tb_toa_v_patch.view(time_dim, patch_dim).mean(dim=1)
        rmse_h, rmse_v = torch.sqrt(criterion(tb_h_sim, obs_tb_h)).item(), torch.sqrt(criterion(tb_v_sim, obs_tb_v)).item()
        rmse_mean = (rmse_h + rmse_v)/2.0
        # print(f'delta_hr={delta_hr}, RTM计算与CoLM,H亮温误差:', torch.sqrt(criterion(tb_h_sim, t_brt_smap_h)).item(), 'V亮温误差:', torch.sqrt(criterion(tb_v_sim, t_brt_smap_v)).item(),'平均误差:', (torch.sqrt(criterion(tb_h_sim, t_brt_smap_h)).item() + torch.sqrt(criterion(tb_v_sim, t_brt_smap_v)).item())/2.0)
        print(f'delta_hr={delta_hr}, RTM计算与观测,H亮温误差:', rmse_h, 'V亮温误差:', rmse_v,'平均误差:', rmse_mean)
        if rmse_mean < best_loss:
            best_loss = rmse_mean
            best_hr = 1.667 + delta_hr
    
    print(f'best_loss: {best_loss} | best_hr: {best_hr}')
    
    sys.exit(0) # 根据你的测试需求，可以保留或去掉 sys.exit(0)
    
    sm_array = sm_all.detach().cpu().numpy()
    b_array = rtm_model.b_array.detach().cpu().numpy()
    weight_array = rtm_model.weight_array.detach().cpu().numpy()
    t_array = rtm_model.t_array.detach().cpu().numpy()
    eps_array = np.round(rtm_model.eps_array.detach().cpu().numpy(), decimals=4)  # 保留4位小数以减少文件大小
    eps_soil_array = np.round(rtm_model.eps_soil_nd.detach().cpu().numpy(), decimals=4)  # 保留4位小数以减少文件大小
    print(np.shape(b_array), np.shape(weight_array), np.shape(t_array), np.shape(eps_array))
    pd.DataFrame(b_array).to_csv(os.path.join(output_dir, f"lat_{ease_lat}_lon_{ease_lon}_b_array.csv"), index=False)
    pd.DataFrame(weight_array).to_csv(os.path.join(output_dir, f"lat_{ease_lat}_lon_{ease_lon}_weight_array.csv"), index=False)
    pd.DataFrame(t_array).to_csv(os.path.join(output_dir, f"lat_{ease_lat}_lon_{ease_lon}_t_array.csv"), index=False)
    pd.DataFrame(eps_array).to_csv(os.path.join(output_dir, f"lat_{ease_lat}_lon_{ease_lon}_eps_array.csv"), index=False)
    # pd.DataFrame(eps_soil_array).to_csv(os.path.join(output_dir, f"lat_{ease_lat}_lon_{ease_lon}_eps_soil_array.csv"), index=False)
    pd.DataFrame(np.round(np.sqrt(eps_array), 4)).to_csv(os.path.join(output_dir, f"lat_{ease_lat}_lon_{ease_lon}_n_complex_array.csv"), index=False)

    # === 5. 绘制剖面子图 ===
    # 计算 n_complex_array
    n_complex_array = np.sqrt(eps_array)
    # 纵轴: 计算 10 层土壤的累加深度
    depths = dz_soi.cpu().numpy().cumsum()
    # 准备 8 个子图的数据和标题配置
    plot_items = [
        (b_array, 'b_array'),
        (weight_array, 'weight_array'),
        (t_array, 't_array'),
        (eps_array.real, 'eps_array (Real)'),
        (eps_array.imag, 'eps_array (Imag)'),
        (n_complex_array.real, 'n_complex (Real)'),
        (n_complex_array.imag, 'n_complex (Imag)'),
        (sm_array, 'Soil Moisture')  # 这个需要根据实际数据替换为对应的数组
    ]
    # 创建 1 行 8 列的子图，共享 Y 轴 (土壤深度)
    fig, axs = plt.subplots(1, 8, figsize=(24, 6), sharey=True)
    for ax, (data, title) in zip(axs, plot_items):
        # 绘制平均值。如果想画出所有的线，可以将 data.mean(axis=0) 改为 data.T，并加上 alpha=0.05
        ax.plot( data[0,:], depths, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Value")
        ax.grid(True, linestyle='--', alpha=0.7)
    # 设置统一的纵轴标签，并翻转 Y 轴（让 0 深度在最上面）
    axs[0].set_ylabel("Soil Depth (m)", fontsize=12)
    axs[0].invert_yaxis()
    plt.tight_layout()
    # 保存并展示
    plot_path = os.path.join(output_dir, f"lat_{ease_lat}_lon_{ease_lon}_profiles.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 剖面图已保存至: {plot_path}")
    plt.show()
    
    
    obs_tb_h_exp = np.repeat(obs['tb_h'], patch_dim)
    obs_tb_v_exp = np.repeat(obs['tb_v'], patch_dim)
    date_exp = np.repeat(obs['date'], patch_dim)
    time_index_exp = np.repeat(obs['time_index'], patch_dim)


    # 导出记录表层数据
    df_out = pd.DataFrame({
        "date": date_exp,
        "time_index": time_index_exp,
        "obs_tb_h": obs_tb_h_exp,
        "obs_tb_v": obs_tb_v_exp,
        "sim_tb_h": tb_toa_h_patch.detach().cpu().numpy(),        
        "sim_tb_v": tb_toa_v_patch.detach().cpu().numpy(),        
        
        "znd_M09_surf": znd_M09_surf.detach().cpu().numpy(),
        "zkd_M09_surf": zkd_M09_surf.detach().cpu().numpy(),
        "zxmvt_M09_surf": zxmvt_M09_surf.detach().cpu().numpy(),
        
        "zep0b_M09_surf": zep0b_M09_surf.detach().cpu().numpy(),
        "ztaub_M09_surf": ztaub_M09_surf.detach().cpu().numpy(),
        "zsigmab_M09_surf": zsigmab_M09_surf.detach().cpu().numpy(),
        
        "zep0u_M09_surf": zep0u_M09_surf.detach().cpu().numpy(),
        "ztauu_M09_surf": ztauu_M09_surf.detach().cpu().numpy(),
        "zsigmau_M09_surf": zsigmau_M09_surf.detach().cpu().numpy(),
        
        "hr": torch.full_like(znd_M09_surf, 1.667).detach().cpu().numpy(),
        "surface_sm": surface_sm.detach().cpu().numpy(),
        "surface_t": surface_t.detach().cpu().numpy(),
        "surface_clay": surface_clay.detach().cpu().numpy(),
        "surface_sand": surface_sand.detach().cpu().numpy(),
        "lai": t_lai.detach().cpu().numpy(),
        "sai": t_sai.detach().cpu().numpy(),
        "htop": t_htop.detach().cpu().numpy(),
        "surface_porsl": surface_porsl.detach().cpu().numpy(),
        "surface_BD_all": surface_BD_all.detach().cpu().numpy(),
        "forc_topo": t_forc_topo.detach().cpu().numpy()
        
    })

    out_csv_path = os.path.join(output_dir, f"lat_{ease_lat}_lon_{ease_lon}_infos.csv")
    df_out.to_csv(out_csv_path, index=False)
    print(f"💾 Step 5.2 数据已成功保存至: {out_csv_path}\n")

if __name__ == "__main__":
    INDEX_FILE = '/home/liusy/storage_global_veg_wigneron/nc_patch_location_index.csv'
    csv_dir = '/home/liusy/store_global_forward/tb_for_EASE_open_lands'
    patch_map_file = '/home/liusy/storage_global_veg_wigneron/patch_map_EASE_open_lands.csv' 
    nc_dir = '/home/liusy/CoLM/outputs/global_veg_wigneron/forward_inputs_folder'
    output_dir="/home/liusy/research_lists/2026-05-20_research_list/tb_calibrate_try"
    
    print("正在加载全局 NC 索引表...")
    df_nc_index = pd.read_csv(INDEX_FILE)
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    
    positions = [
    "lat_25.529726_lon_-98.77593",
    "lat_20.024717_lon_-5.041494",
    "lat_20.626143_lon_40.518673",
    "lat_48.579163_lon_92.05394",
    'lat_-19.72485_lon_141.72198', 
    "lat_-19.72485_lon_141.34854",
    'lat_41.80753_lon_-115.58091',
    'lat_23.67014_lon_43.132782',
    'lat_-21.532677_lon_22.59336',
    'lat_64.98099_lon_134.25311',
    'lat_39.950367_lon_-114.46058',
    'lat_14.994413_lon_2.80083',
    'lat_46.09634_lon_109.979256',
    'lat_48.579163_lon_78.983406',
    'lat_-22.140102_lon_147.6971',
    'lat_67.76884_lon_113.71369',
    'lat_8.932143_lon_29.315353',
    'lat_17.344746_lon_12.883818',
    'lat_18.233519_lon_29.315353',
    'lat_-16.754568_lon_22.59336',
    'lat_4.9487443_lon_36.784233',
    'lat_29.33835_lon_-104.75104',
    'lat_17.93678_lon_-4.2946057',
    'lat_38.499725_lon_80.10374',
    'lat_62.456604_lon_174.58507',
    'lat_27.735691_lon_53.215767',
    'lat_31.62478_lon_81.59751',
    'lat_29.01589_lon_72.26141',
    'lat_-21.532677_lon_142.46887',
    'lat_-22.140102_lon_-66.28631',
    'lat_30.966091_lon_-101.016594',
    'lat_9.790613_lon_44.626556',
    'lat_-29.33835_lon_135.37344',
    'lat_21.532677_lon_22.966805',
    'lat_9.790613_lon_33.79668',
    'lat_29.01589_lon_4.2946057',
    'lat_33.629246_lon_88.3195',
    'lat_69.294495_lon_-159.64731',
    'lat_34.30753_lon_68.15353',
    'lat_-26.15579_lon_119.6888',
    'lat_-21.836075_lon_125.29046',
    'lat_29.661814_lon_44.626556'
    ]
    
    set_seed(seed=42)  
    
    for grid_id, position in enumerate(positions[0:]):  
        
        target_csv = os.path.join(csv_dir, f"{position}_SMAP_TB.csv") 
        ease_lat, ease_lon = map(float, os.path.basename(target_csv).split('_')[1:4:2])
        print(f"正在处理网格文件: {target_csv} (经度: {ease_lon}, 纬度: {ease_lat})")
        df_obs = pd.read_csv(target_csv)
        result = process_single_grid(df_nc_index, df_obs, patch_map_file, nc_dir) 
        run_step_5_2_calibration(result, ease_lat, ease_lon, output_dir)
        break
