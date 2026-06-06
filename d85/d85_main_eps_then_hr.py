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
import glob
from step_5_1 import process_single_grid
from rtm import DifferentiableRTM
import sys
import csv
import math

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed has been fixed to: {seed}")



def compute_M09_params(t_celsius, wf_clay_percent):
    """
    Calculate the spectral parameters of the Mironov 2009 soil dielectric model
    Args:
        t_celsius       : Soil temperature (Celsius) - scalar or tensor
        wf_clay_percent : Clay weight percentage (e.g., 30 means 30%) - scalar or tensor
    Returns:
        znd, zkd, zxmvt, zep0b, ztaub, zsigmab, zep0u, ztauu, zsigmau
    """
    # --- Uniformly convert to Tensor to avoid scalar type errors ---
    if not torch.is_tensor(t_celsius):
        t_celsius = torch.tensor(t_celsius, dtype=torch.float32)
    if not torch.is_tensor(wf_clay_percent):
        wf_clay_percent = torch.tensor(wf_clay_percent, dtype=torch.float32)

    # Convert constants to Tensors to ensure all operations are tensor operations
    ts = torch.tensor(20.0, dtype=torch.float32)
    tk = t_celsius + torch.tensor(273.15, dtype=torch.float32)

    # 1. Dry soil refractive index (real/imaginary part)
    znd = 1.634 - 0.539e-2 * wf_clay_percent + 0.2748e-4 * (wf_clay_percent ** 2)
    zkd = 0.03952 - 0.04038e-2 * wf_clay_percent

    # 2. Maximum bound water volume content
    zxmvt = 0.02863 + 0.30673e-2 * wf_clay_percent

    # 3. Bound water static dielectric constant (zep0b)
    e0b = 79.8 - 85.4e-2 * wf_clay_percent + 32.7e-4 * (wf_clay_percent ** 2)
    Bb = (8.67e-19 - 0.00126e-2 * wf_clay_percent
          + 0.00184e-4 * (wf_clay_percent ** 2)
          - 9.77e-10 * (wf_clay_percent ** 3)
          - 1.39e-15 * (wf_clay_percent ** 4))
    Fb = torch.log(torch.clamp((e0b - 1.0) / (e0b + 2.0), min=1e-8))
    exp_term_b = torch.exp(Fb - Bb * (t_celsius - ts))
    zep0b = (1.0 + 2.0 * exp_term_b) / (1.0 - exp_term_b)

    # 4. Bound water relaxation time (ztaub)
    dHbR = (1467.0 + 2697e-2 * wf_clay_percent
            - 980e-4 * (wf_clay_percent ** 2)
            + 1.368e-10 * (wf_clay_percent ** 3)
            - 8.61e-13 * (wf_clay_percent ** 4))
    dSbR = (0.888 + 9.7e-2 * wf_clay_percent
            - 4.262e-4 * (wf_clay_percent ** 2)
            + 6.79e-21 * (wf_clay_percent ** 3)
            + 4.263e-22 * (wf_clay_percent ** 4))
    ztaub = 48e-12 * torch.exp(dHbR / tk - dSbR) / tk

    # 5. Bound water effective conductivity (zsigmab)
    Bsgb = (0.0028 + 0.02094e-2 * wf_clay_percent
            - 0.01229e-4 * (wf_clay_percent ** 2)
            - 5.03e-22 * (wf_clay_percent ** 3)
            + 4.163e-24 * (wf_clay_percent ** 4))
    sigmabt = 0.3112 + 0.467e-2 * wf_clay_percent
    zsigmab = sigmabt + Bsgb * (t_celsius - ts)

    # 6. Free water static dielectric constant (zep0u)
    e0u = torch.tensor(100.0, dtype=torch.float32)          # ★ Critical fix: Must be a Tensor
    Bu = (1.11e-4 - 1.603e-7 * wf_clay_percent
          + 1.239e-9 * (wf_clay_percent ** 2)
          + 8.33e-13 * (wf_clay_percent ** 3)
          - 1.007e-14 * (wf_clay_percent ** 4))
    Fu = torch.log(torch.clamp((e0u - 1.0) / (e0u + 2.0), min=1e-8))
    exp_term_u = torch.exp(Fu - Bu * (t_celsius - ts))
    zep0u = (1.0 + 2.0 * exp_term_u) / (1.0 - exp_term_u)

    # 7. Free water relaxation time (ztauu)
    dHuR = (2231.0 - 143.1e-2 * wf_clay_percent
            + 223.2e-4 * (wf_clay_percent ** 2)
            - 142.1e-6 * (wf_clay_percent ** 3)
            + 27.14e-8 * (wf_clay_percent ** 4))
    dSuR = (3.649 - 0.4894e-2 * wf_clay_percent
            + 0.763e-4 * (wf_clay_percent ** 2)
            - 0.4859e-6 * (wf_clay_percent ** 3)
            + 0.0928e-8 * (wf_clay_percent ** 4))
    ztauu = 48e-12 * torch.exp(dHuR / tk - dSuR) / tk

    # 8. Free water effective conductivity (zsigmau)
    Bsgu = (0.00108 + 0.1413e-2 * wf_clay_percent
            - 0.2555e-4 * (wf_clay_percent ** 2)
            + 0.2147e-6 * (wf_clay_percent ** 3)
            - 0.0711e-8 * (wf_clay_percent ** 4))
    sigmaut = 0.05 + 1.4 * (1.0 - (1.0 - wf_clay_percent * 1e-2) ** 4.664)
    zsigmau = sigmaut + Bsgu * (t_celsius - ts)

    return znd, zkd, zxmvt, zep0b, ztaub, zsigmab, zep0u, ztauu, zsigmau


def get_M09_eps_complex(sm, t, clay):
    """
    Convert M09 parameters to absolute complex dielectric constants at 1.4 GHz.
    This provides a physically accurate baseline for initialization searches.
    """
    znd, zkd, zxmvt, zep0b, ztaub, zsigmab, zep0u, ztauu, zsigmau = compute_M09_params(t, clay)
    
    f = 1.4e9
    omega = 2.0 * np.pi * f
    epsv = 8.854e-12
    eps_inf = 4.9
    
    # Bound water complex dielectric
    eps_b_real = eps_inf + (zep0b - eps_inf) / (1.0 + (omega * ztaub)**2)
    eps_b_imag = (zep0b - eps_inf) * (omega * ztaub) / (1.0 + (omega * ztaub)**2) + zsigmab / (omega * epsv)
    
    mag_b = torch.sqrt(eps_b_real**2 + eps_b_imag**2)
    n_b = torch.sqrt((mag_b + eps_b_real) / 2.0)
    k_b = torch.sqrt((mag_b - eps_b_real) / 2.0)
    
    # Free water complex dielectric
    eps_u_real = eps_inf + (zep0u - eps_inf) / (1.0 + (omega * ztauu)**2)
    eps_u_imag = (zep0u - eps_inf) * (omega * ztauu) / (1.0 + (omega * ztauu)**2) + zsigmau / (omega * epsv)
    
    mag_u = torch.sqrt(eps_u_real**2 + eps_u_imag**2)
    n_u = torch.sqrt((mag_u + eps_u_real) / 2.0)
    k_u = torch.sqrt((mag_u - eps_u_real) / 2.0)
    
    # RIMM Mixing Phase
    wb = torch.clamp(sm, max=zxmvt)
    wu = torch.clamp(sm - zxmvt, min=0.0)
    
    n_s = znd + (n_b - 1.0) * wb + (n_u - 1.0) * wu
    k_s = zkd + k_b * wb + k_u * wu
    
    eps_real = n_s**2 - k_s**2
    eps_imag = 2.0 * n_s * k_s
    
    return torch.complex(eps_real, eps_imag)


#=======================================================================
# Network 1: Direct Dielectric Predictor (Outputs wet soil complex dielectric constant)
# Input: 6 variables related only to soil physical properties (sm, t, clay, sand, porosity, BD)
# Output: Complex dielectric constant (Real part range [1, 80], Imaginary part range [0, 80])
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
            nn.Linear(256, 2)  # <--- Output real and imaginary parts (2 channels)
        )
        self._init_weights()

    def _init_weights(self):
        last_linear = [layer for layer in self.net if isinstance(layer, nn.Linear)][-1]
        nn.init.xavier_uniform_(last_linear.weight, gain=0.1) 
        nn.init.zeros_(last_linear.bias)

    # === Replace the original forward method with the following code ===
    def forward(self, x, pclass):
        emb = self.embed(pclass)
        x_in = torch.cat([x, emb], dim=-1)
        logits = self.net(x_in)             
        
        # 输出 Dobson 模型的 beta' 和 电导率 sigma_soil
        # 利用 sigmoid 将预测值约束在合理的物理区间内 (beta' 在 0.1 到 5.0)
        # beta = 0.001 + 24.999 * torch.sigmoid(logits[..., 0])
        beta = 0.001 + torch.nn.functional.softplus(logits[..., 0])
        # 新增：将第二个输出约束为 sigma_soil (范围 0.05 到 25)
        sigma_soil = 0.001 + torch.nn.functional.softplus(logits[..., 1])
        # sigma_soil = 0.001 + 24.999 * torch.sigmoid(logits[..., 1])
        
        # 组合并返回 (覆盖原本返回复介电常数的位置)
        return torch.stack([beta, sigma_soil], dim=-1), logits

#=======================================================================
# Network 2: Effective Roughness Predictor (Original logic retained)
# Input: topo, sm, lai, sai, htop (5 variables) + Patchclass
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
        
        # hr = torch.nn.functional.softplus(logits)
        hr = torch.sigmoid(logits)
        return hr.squeeze(-1), logits.squeeze(-1)


# =======================================================================
# Physics-Informed Loss Function (Numerically Stabilized)
# =======================================================================
# def physics_constrained_loss(TB_obs_h, TB_obs_v, TB_sim_h, TB_sim_v, eps_complex_surf, eps_complex_all, PR_obs):

#     # 1. Base data fitting error (RMSE) with epsilon protection for zero-gradients
#     rmse_h = torch.sqrt(nn.MSELoss()(TB_sim_h, TB_obs_h) + 1e-8)
#     rmse_v = torch.sqrt(nn.MSELoss()(TB_sim_v, TB_obs_v) + 1e-8)
#     rmse_loss = (rmse_h + rmse_v) / 2.0
   

#     # Epsilon protection for PR calculation to prevent division by zero
#     PR_sim = (TB_sim_v - TB_sim_h) / (TB_sim_v + TB_sim_h + 1e-8)
#     PR_loss = torch.sqrt(nn.MSELoss()(PR_sim, PR_obs) + 1e-8)

#     EPS_REAL_MIN = 2.0

#     # Extract all predicted real and imaginary parts
#     eps_real = torch.cat([eps_complex_surf.real, eps_complex_all.real.flatten()])
#     eps_imag = torch.cat([eps_complex_surf.imag, eps_complex_all.imag.flatten()])

#     # 2. Physical boundary constraints
#     bound_loss = torch.mean(torch.relu(EPS_REAL_MIN - eps_real)) + torch.mean(torch.relu(-eps_imag))

#     # 3. Synergistic physical envelope constraint (L-band)
#     lower_bound = 0.05 * (eps_real - EPS_REAL_MIN)
#     upper_bound = 0.35 * (eps_real - EPS_REAL_MIN)
#     structure_loss = torch.mean(torch.relu(lower_bound - eps_imag)) + torch.mean(torch.relu(eps_imag - upper_bound))
    
#     # Total loss function
#     total_loss = rmse_loss + 10.0 * bound_loss + 5.0 * structure_loss + 1000.0 * PR_loss

#     return total_loss, rmse_loss, PR_loss

# =======================================================================
# Physics-Informed Loss Function (Numerically Stabilized)
# =======================================================================
def physics_constrained_loss(TB_obs_h, TB_obs_v, TB_sim_h, TB_sim_v, eps_complex_surf, eps_complex_all, PR_obs, eps_M09_surf_real,
                             lamda_bound_loss=10, lamda_structure_loss=5, lamda_PR_loss=1000):
    # 1. Base data fitting error (RMSE) with epsilon protection for zero-gradients
    rmse_h = torch.sqrt(nn.MSELoss()(TB_sim_h, TB_obs_h) + 1e-8)
    rmse_v = torch.sqrt(nn.MSELoss()(TB_sim_v, TB_obs_v) + 1e-8)
    rmse_loss = (rmse_h + rmse_v) / 2.0
    
    # Epsilon protection for PR calculation to prevent division by zero
    PR_sim = (TB_sim_v - TB_sim_h) / (TB_sim_v + TB_sim_h + 1e-8)
    PR_loss = torch.sqrt(nn.MSELoss()(PR_sim, PR_obs) + 1e-8)
    
    EPS_REAL_MIN = 2.0
    
    # Extract all predicted real and imaginary parts
    eps_real = torch.cat([eps_complex_surf.real])  # , eps_complex_all.real.flatten()
    eps_imag = torch.cat([eps_complex_surf.imag])  # , eps_complex_all.imag.flatten()
    
    # 2. Physical boundary constraints
    # --- 保留原有的基础物理边界约束（针对所有层） ---
    bound_loss_general = torch.mean(torch.relu(EPS_REAL_MIN - eps_real)) + torch.mean(torch.relu(-eps_imag))
    
    # === 新增：限制虚部/实部的比值 (即 eps_imag <= 1.05 * eps_real) ===
    ratio_loss = torch.mean(torch.relu(eps_imag - 1.05 * eps_real))
    
    # --- 新增：仅针对表层实部的 M09 ±50% 约束 ---
    lower_bound_surf = 0.5 * eps_M09_surf_real
    upper_bound_surf = 1.5 * eps_M09_surf_real
    bound_loss_surf = torch.mean(torch.relu(lower_bound_surf - eps_complex_surf.real)) + \
                      torch.mean(torch.relu(eps_complex_surf.real - upper_bound_surf))
                      
    # 合并边界约束 Loss (将 ratio_loss 加入总 bound_loss 中)
    bound_loss = bound_loss_general + bound_loss_surf + ratio_loss
    
    # 3. Synergistic physical envelope constraint (L-band)
    lower_bound = 0.05 * (eps_real - EPS_REAL_MIN)
    upper_bound = 0.35 * (eps_real - EPS_REAL_MIN)
    
    structure_loss = torch.mean(torch.relu(lower_bound - eps_imag)) + \
                     torch.mean(torch.relu(eps_imag - upper_bound))
                     
    # Total loss function
    total_loss = rmse_loss + lamda_bound_loss* bound_loss + lamda_structure_loss * structure_loss + lamda_PR_loss * PR_loss
    return total_loss, rmse_loss, PR_loss



def run_step_5_2_calibration(result, ease_lat, ease_lon, output_dir):
    if result is None:
        print("No valid data received, skipping Step 5.2.")
        return

    # === 0. Basic configuration ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"========= 🚀 Starting End-to-End Complex Dielectric Predictor Network (Device: {device}) =========")
    os.makedirs(output_dir, exist_ok=True)
    

    obs = result['obs']
    model_inputs = result['model_inputs']
    time_dim = len(obs['tb_h'])
    patch_dim = len(model_inputs['htop'])
    num_samples = time_dim * patch_dim

    # === 1. Dimension expansion ===
    def expand_static_1d(arr): return np.broadcast_to(arr, (time_dim, patch_dim)).reshape(-1)
    def expand_static_2d(arr, dim3): return np.broadcast_to(arr, (time_dim, patch_dim, dim3)).reshape(-1, dim3)
    def flatten_dynamic_1d(arr): return arr.reshape(-1)
    def flatten_dynamic_2d(arr, dim3): return arr.reshape(-1, dim3)

    # === 0.5 Calculate Inverse Distance Weighting (IDW) ===
    lat_grid = float(ease_lat)
    lon_grid = float(ease_lon)
    patch_lats_raw = model_inputs['lat']
    patch_lons_raw = model_inputs['lon']
    # Calculate Euclidean distance
    distances = np.sqrt((patch_lats_raw - lat_grid)**2 + (patch_lons_raw - lon_grid)**2)
    # Safe clipping of minimum values to prevent division by 0 if a patch is exactly at the grid center
    distances = np.where(distances == 0, 1e-8, distances)
    # Calculate the reciprocal of the distance and normalize to ensure the sum of weights is 1
    inv_distances = 1.0 / distances
    patch_weights = inv_distances / np.sum(inv_distances)
    # Convert to Tensor and move to the corresponding device for matrix broadcasting multiplication in the training loop
    t_weights = torch.tensor(patch_weights, dtype=torch.float32, device=device)
    # t_weights = torch.tensor(np.array([0.25, 0.25, 0.25, 0.25]), dtype=torch.float32, device=device)
    
    patch_lat = expand_static_1d(model_inputs['lat'])
    patch_lon = expand_static_1d(model_inputs['lon'])
    
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

    # === 2. Tensor conversion ===
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

    # === 3. Build separated and multi-layer feature inputs ===
    maxsnl = 5  
    t_wf_total = t_wf_clay + t_wf_sand + t_wf_silt
    w1, w2 = 0.0175, 0.0276        
    wtot = w1 + w2
    
    # —— Packaged surface parameters dedicated for surface dielectric network and output ——
    surface_sm =  (t_wliq_soisno[:, maxsnl] + t_wliq_soisno[:, maxsnl+1]) / (wtot * 1000.0)
    surface_t =  ((t_t_soisno[:, maxsnl]*w1 + t_t_soisno[:, maxsnl+1]*w2) / wtot) - 273.15 
    surface_clay = (t_wf_clay[:, 0]/t_wf_total[:, 0]*w1 + t_wf_clay[:, 1]/t_wf_total[:, 1]*w2) / wtot * 100
    surface_sand = (t_wf_sand[:, 0]/t_wf_total[:, 0]*w1 + t_wf_sand[:, 1]/t_wf_total[:, 1]*w2) / wtot * 100
    surface_porsl = (t_porsl[:, 0]/t_wf_total[:, 0]*w1 + t_porsl[:, 1]/t_wf_total[:, 1]*w2) / wtot
    surface_BD_all = (t_BD_all[:, 0]/t_wf_total[:, 0]*w1 + t_BD_all[:, 1]/t_wf_total[:, 1]*w2) / wtot / 1000

    x_diel_raw_surf = torch.stack([surface_sm, surface_t, surface_clay, surface_sand, surface_porsl, surface_BD_all], dim=1) 
    x_diel_norm_surf = (x_diel_raw_surf - x_diel_raw_surf.mean(dim=0)) / (x_diel_raw_surf.std(dim=0) + 1e-8)

    # —— 10-layer soil parameters prepared for deep dielectric constant ——
    dz_soi = torch.tensor([0.0175, 0.0276, 0.0455, 0.0750, 0.1236, 0.2038, 0.3360, 0.5539, 0.9133, 1.5058], device=device)
    t_wliq_soi = t_wliq_soisno[:, maxsnl:]
    t_t_soi = t_t_soisno[:, maxsnl:]
    
    sm_all = t_wliq_soi / (dz_soi.unsqueeze(0) * 1000.0)
    t_all = t_t_soi - 273.15
    clay_all = (t_wf_clay / t_wf_total) * 100.0
    sand_all = (t_wf_sand / t_wf_total) * 100.0
    porsl_all = t_porsl / t_wf_total 
    BD_all_all = (t_BD_all / t_wf_total) / 1000.0
    
    # Combine into [N, 10, 6] full-layer inputs
    x_diel_raw_all = torch.stack([sm_all, t_all, clay_all, sand_all, porsl_all, BD_all_all], dim=-1) 
    
    # Flatten to uniformly Normalize and pass into the network: [N*10, 6]
    x_diel_raw_flat = x_diel_raw_all.view(-1, 6)
    x_diel_norm_flat = (x_diel_raw_flat - x_diel_raw_flat.mean(dim=0)) / (x_diel_raw_flat.std(dim=0) + 1e-8)
    
    # Corresponding 10-layer Patchclass tiling: [N*10]
    patchclass_all_flat = t_patchclass.unsqueeze(1).expand(-1, 10).reshape(-1)

    # Feature set 2: Roughness network (5D) - only surface parameters required
    x_rough_raw = torch.stack([t_forc_topo, surface_sm, surface_t, surface_porsl, surface_BD_all], dim=1)
    x_rough_norm = (x_rough_raw - x_rough_raw.mean(dim=0)) / (x_rough_raw.std(dim=0) + 1e-8)


    # === 4. Initialize models and dual networks ===
    rtm_model = DifferentiableRTM(def_da_rtm_diel=4, def_da_rtm_rough=0, def_da_rtm_veg=0, 
                                  num_grids=num_samples, maxsnl=maxsnl).to(device)
    
    net_diel = DielectricPredictor().to(device)
    net_rough = RoughnessPredictor().to(device)
    criterion = nn.MSELoss()
    
    t_brt_smap_h = torch.tensor(t_brt_smap_h, dtype=torch.float32, device=device)
    t_brt_smap_v = torch.tensor(t_brt_smap_v, dtype=torch.float32, device=device)
    print('Initial H brightness temperature error calculated using CoLM:', torch.sqrt(criterion(t_brt_smap_h, obs_tb_h)).item(), 'V error:', torch.sqrt(criterion(t_brt_smap_v, obs_tb_v)).item())

    # Dynamically search for best_hr using the pure physical M09 baseline
    best_hr, best_loss = float('inf'), float('inf')
    with torch.no_grad():
        # 2. Generate pure M09 dielectric constants to isolate the hr search 
        eps_M09_surf = get_M09_eps_complex(surface_sm, surface_t, surface_clay)
        eps_M09_all = get_M09_eps_complex(sm_all, t_all, clay_all)
        print("Starting search for optimal roughness hr using physical M09 baseline...")
        for delta_hr in np.arange(-1.6, 1.6, 0.1):
            temp_hr = 1.667 + delta_hr
            hr_pred_temp = torch.full((num_samples,), temp_hr, device=device)
            # Pass the M09 priors into the RTM for an untainted physical baseline evaluation
            tb_toa_h_patch, tb_toa_v_patch = rtm_model(
                t_patchtype, t_patchclass, t_dz_sno, t_forc_topo, t_htop, 
                t_tref, t_t_soisno, t_tleaf, t_wliq_soisno, t_wice_soisno, t_h2osoi, 
                t_snowdp, t_lai, t_sai, 
                t_wf_clay, t_wf_sand, t_wf_silt, t_BD_all, t_porsl, 
                t_sat_theta, t_sat_fghz,
                eps_M09_surf, eps_M09_all, hr_pred_temp
            )            
            tb_h_sim = torch.sum(tb_toa_h_patch.view(time_dim, patch_dim) * t_weights, dim=1)
            tb_v_sim = torch.sum(tb_toa_v_patch.view(time_dim, patch_dim) * t_weights, dim=1)
            rmse_h = torch.sqrt(criterion(tb_h_sim, obs_tb_h)).item()
            rmse_v = torch.sqrt(criterion(tb_v_sim, obs_tb_v)).item()
            rmse_mean = (rmse_h + rmse_v) / 2.0
            
            # print(f'delta_hr={delta_hr:.2f}, RTM vs OBS, H Error: {rmse_h:.4f}, V Error: {rmse_v:.4f}, Mean Error: {rmse_mean:.4f}')
            if rmse_mean < best_loss:
                best_loss = rmse_mean
                best_hr = temp_hr
                
        print(f'best_loss: {best_loss:.4f} | best_hr: {best_hr:.4f}')
        # 3. Apply dynamically found bounds
        center_hr = best_hr
        left_hr = torch.tensor(center_hr, device=device) - 0.1
        right_hr = torch.tensor(center_hr, device=device) + 0.1
        hr_pred_init = torch.full((num_samples,), center_hr, device=device)
    # 4. Re-run the RTM once with the network's INITIAL RANDOM eps and the optimal best_hr 
    # to load the computational graphs and RTM internal memory states (like ffrz_array) for Stage 1
    tb_toa_h_patch, tb_toa_v_patch = rtm_model(
        t_patchtype, t_patchclass, t_dz_sno, t_forc_topo, t_htop, 
        t_tref, t_t_soisno, t_tleaf, t_wliq_soisno, t_wice_soisno, t_h2osoi, 
        t_snowdp, t_lai, t_sai, 
        t_wf_clay, t_wf_sand, t_wf_silt, t_BD_all, t_porsl, 
        t_sat_theta, t_sat_fghz,
         eps_M09_surf, eps_M09_all, hr_pred_init
    )
    tb_h_sim = torch.sum(tb_toa_h_patch.view(time_dim, patch_dim) * t_weights, dim=1)
    tb_v_sim = torch.sum(tb_toa_v_patch.view(time_dim, patch_dim) * t_weights, dim=1)
    print(f'pytorchRTM calculation with M09 eps + optimal hr bounds | H error: {torch.sqrt(criterion(tb_h_sim, obs_tb_h)).item():.4f} | V error: {torch.sqrt(criterion(tb_v_sim, obs_tb_v)).item():.4f}')
    
    ffrz_array = rtm_model.ffrz
    eps_surf_M09 = rtm_model.eps_soil_nd_M09
    eps_surf_D85 = rtm_model.eps_soil_nd_D85
    # Extract M09 real parts as reference bounds
    eps_M09_surf_real = eps_surf_M09.real.detach()
    PR_obs = (obs_tb_v - obs_tb_h) / (obs_tb_v + obs_tb_h)
    
    # === 在 STAGE 1 之前加入这个辅助函数 ===
    def get_eps_from_beta(beta_out, sm, t, sand, clay, bd_all):
        f_val = 1.4e9
        omega_val = 2.0 * np.pi * f_val
        # 1. 先生水介电常数 ew，将网络预测的 sigma_soil 传入
        ew = rtm_model.diel_water_soil(-1, sm, t, sand, clay, bd_all, 0.0, f_val, omega_val, sigma_pred=beta_out[:, 1])
        # 2. 调用覆盖了 beta 和 beta_i 参数的 Dobson 模型，将 beta_i_pred 固定为 1.05
        return rtm_model.diel_soil_D85(ew, sm, sand, clay, bd_all, beta_pred=beta_out[:, 0], beta_i_pred=1.05)

    loss_array_stage1 = []    
    loss_array_stage2 = []
    epochs = 50000
    patience = 5000 
    whether_patience = True
    t0 = time.time()
    # =======================================================================
    # STAGE 1: Optimize Dielectric Network (net_diel)
    # =======================================================================
    print("--- Stage 1: Optimizing Dielectric Network ---")
    optimizer_diel = optim.Adam(net_diel.parameters(), lr=0.01, weight_decay=1e-5) 
    scheduler_diel = optim.lr_scheduler.ReduceLROnPlateau(optimizer_diel, mode='min', factor=0.9, patience=1000)
    
    net_diel.train()
    net_rough.eval()
    
    best_loss_stage1 = float('inf')
    best_diel_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        optimizer_diel.zero_grad()
    
        # 1. Forward pass: calculate beta and then convert to complex dielectric using Dobson
        beta_pred_surf, _ = net_diel(x_diel_norm_surf, t_patchclass)
        beta_pred_flat, _ = net_diel(x_diel_norm_flat, patchclass_all_flat)
        eps_pred_surf = get_eps_from_beta(beta_pred_surf, surface_sm, surface_t, surface_sand, surface_clay, surface_BD_all)
        eps_pred_flat = get_eps_from_beta(beta_pred_flat, sm_all.flatten(), t_all.flatten(), sand_all.flatten(), clay_all.flatten(), BD_all_all.flatten())
        eps_pred_all = eps_pred_flat.view(-1, 10)
        
        # Calculate roughness (Fixed)
        # 将 hr_pred_init 直接设定为中心值 center_hr，形状与样本数一致
        hr_pred_init = torch.full((num_samples,), center_hr, device=device)
  
        # 2. Couple RTM (directly pass in predicted surface and deep dielectric constant tensors)
        tb_toa_h_patch, tb_toa_v_patch = rtm_model(
            t_patchtype, t_patchclass, t_dz_sno, t_forc_topo, t_htop, 
            t_tref, t_t_soisno, t_tleaf, t_wliq_soisno, t_wice_soisno, t_h2osoi, 
            t_snowdp, t_lai, t_sai, 
            t_wf_clay, t_wf_sand, t_wf_silt, t_BD_all, t_porsl, 
            t_sat_theta, t_sat_fghz,
            eps_pred_surf, eps_pred_all,
            hr_pred_init
        )
        
        tb_h_sim = torch.sum(tb_toa_h_patch.view(time_dim, patch_dim) * t_weights, dim=1)
        tb_v_sim = torch.sum(tb_toa_v_patch.view(time_dim, patch_dim) * t_weights, dim=1)
        
        loss, rmse_loss, PR_loss = physics_constrained_loss(
                obs_tb_h, obs_tb_v, 
                tb_h_sim, tb_v_sim, 
                eps_pred_surf, eps_pred_all,
                PR_obs, eps_M09_surf_real, 
                lamda_bound_loss=10, lamda_structure_loss=0,lamda_PR_loss=1000
            )
        
        loss.backward()
        # Gradient clipping (ultimate insurance)
        torch.nn.utils.clip_grad_norm_(net_diel.parameters(), max_norm=1.0)
        optimizer_diel.step()
        
        current_loss = loss.item()
        current_rmse = rmse_loss.item()
        scheduler_diel.step(current_loss)
        loss_array_stage1.append(current_rmse)
        
        # Determine the best model based on pure brightness temperature error
        if current_rmse < best_loss_stage1:
            patience_counter = 0 
            best_loss_stage1 = current_rmse
            best_diel_weights = copy.deepcopy(net_diel.state_dict())
        else:
            patience_counter += 1
            
        if epoch % 2000 == 0 or epoch == epochs - 1:
            current_lr = optimizer_diel.param_groups[0]['lr']
            print(f"Stage 1 Epoch {epoch:04d} | Loss: {current_loss:.4f} | RMSE Loss: {current_rmse:.4f} | PR_loss: {PR_loss:.4f} | Best RMSE: {best_loss_stage1:.4f} | LR: {current_lr:.6f} | H Err: {torch.sqrt(criterion(tb_h_sim, obs_tb_h)).item():.4f} | V Err: {torch.sqrt(criterion(tb_v_sim, obs_tb_v)).item():.4f}")

        if (patience_counter >= patience and whether_patience ) or best_loss_stage1 < 1 :
            print(f"⏹ Trigger early stopping! Training ends early at Epoch {epoch}.")
            break

    # =======================================================================
    # STAGE 2: Optimize Roughness Network (net_rough)
    # =======================================================================
    print("--- Stage 2: Optimizing Roughness Network ---")
    net_diel.load_state_dict(best_diel_weights)
    net_diel.eval()
    
    optimizer_rough = optim.Adam(net_rough.parameters(), lr=0.01, weight_decay=1e-5) 
    scheduler_rough = optim.lr_scheduler.ReduceLROnPlateau(optimizer_rough, mode='min', factor=0.9, patience=500)
    
    net_rough.train()
    
    best_loss_stage2 = float('inf')
    best_rough_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        optimizer_rough.zero_grad()
        
        # Fixed Dielectric
        with torch.no_grad():
            beta_pred_surf, _ = net_diel(x_diel_norm_surf, t_patchclass)
            beta_pred_flat, _ = net_diel(x_diel_norm_flat, patchclass_all_flat)
            eps_pred_surf = get_eps_from_beta(beta_pred_surf, surface_sm, surface_t, surface_sand, surface_clay, surface_BD_all)
            eps_pred_flat = get_eps_from_beta(beta_pred_flat, sm_all.flatten(), t_all.flatten(), sand_all.flatten(), clay_all.flatten(), BD_all_all.flatten())
            eps_pred_all = eps_pred_flat.view(-1, 10)
        
        # Calculate roughness
        hr_pred, _ = net_rough(x_rough_norm, t_patchclass)
        hr_pred = left_hr + (right_hr - left_hr) * hr_pred
        
        # Couple RTM
        tb_toa_h_patch, tb_toa_v_patch = rtm_model(
            t_patchtype, t_patchclass, t_dz_sno, t_forc_topo, t_htop, 
            t_tref, t_t_soisno, t_tleaf, t_wliq_soisno, t_wice_soisno, t_h2osoi, 
            t_snowdp, t_lai, t_sai, 
            t_wf_clay, t_wf_sand, t_wf_silt, t_BD_all, t_porsl, 
            t_sat_theta, t_sat_fghz,
            eps_pred_surf, eps_pred_all,
            hr_pred
        )
        
        tb_h_sim = torch.sum(tb_toa_h_patch.view(time_dim, patch_dim) * t_weights, dim=1)
        tb_v_sim = torch.sum(tb_toa_v_patch.view(time_dim, patch_dim) * t_weights, dim=1)
        
        loss, rmse_loss, PR_loss = physics_constrained_loss(
                obs_tb_h, obs_tb_v, 
                tb_h_sim, tb_v_sim, 
                eps_pred_surf, eps_pred_all,
                PR_obs, eps_M09_surf_real, 
                lamda_bound_loss=10, lamda_structure_loss=0, lamda_PR_loss=1000
            )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_rough.parameters(), max_norm=1.0)
        optimizer_rough.step()
        
        current_loss = loss.item()
        current_rmse = rmse_loss.item()
        scheduler_rough.step(current_loss)
        loss_array_stage2.append(current_rmse)
        
        if current_rmse < best_loss_stage2:
            patience_counter = 0 
            best_loss_stage2 = current_rmse
            best_rough_weights = copy.deepcopy(net_rough.state_dict())
        else:
            patience_counter += 1
            
        if epoch % 2000 == 0 or epoch == epochs - 1:
            current_lr = optimizer_rough.param_groups[0]['lr']
            print(f"Stage 2 Epoch {epoch:04d} | Loss: {current_loss:.4f} | RMSE Loss: {current_rmse:.4f} | PR_loss: {PR_loss:.4f} | Best RMSE: {best_loss_stage2:.4f} | LR: {current_lr:.6f} | H Err: {torch.sqrt(criterion(tb_h_sim, obs_tb_h)).item():.4f} | V Err: {torch.sqrt(criterion(tb_v_sim, obs_tb_v)).item():.4f}")

        if (patience_counter >= patience/2 and whether_patience) or best_loss_stage2 < 0.5 :
            print(f"⏹ Trigger early stopping! Training ends early at Epoch {epoch}.")
            break

    print(f"Optimization converged! Time taken: {time.time()-t0:.2f}s | Best Stage 2 Loss (RMSE): {best_loss_stage2:.4f}")

    current_time_str = time.strftime("%Y%m%d_%H%M%S")

    plt.plot(np.array(loss_array_stage1 + loss_array_stage2), label='Combined Loss')
    plt.axvline(x=len(loss_array_stage1), color='r', linestyle='--', label='Stage 2 Start')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    os.makedirs('tb_calibrate_loss_figure', exist_ok=True)
    plt.savefig(f'tb_calibrate_loss_figure/training_loss_curve_lat_{ease_lat}_lon_{ease_lon}.png')
    plt.close()
  
    # === 6. Output calibration results to CSV ===
    net_diel.load_state_dict(best_diel_weights)
    net_rough.load_state_dict(best_rough_weights)
    net_diel.eval()
    net_rough.eval()
    
    with torch.no_grad():
        final_beta_surf, _ = net_diel(x_diel_norm_surf, t_patchclass)
        final_beta_flat, _ = net_diel(x_diel_norm_flat, patchclass_all_flat)
        final_eps_surf = get_eps_from_beta(final_beta_surf, surface_sm, surface_t, surface_sand, surface_clay, surface_BD_all)
        final_eps_flat = get_eps_from_beta(final_beta_flat, sm_all.flatten(), t_all.flatten(), sand_all.flatten(), clay_all.flatten(), BD_all_all.flatten())
        final_eps_all = final_eps_flat.view(-1, 10)
        
        final_hr, _ = net_rough(x_rough_norm, t_patchclass)
        final_hr = left_hr + (right_hr - left_hr) * final_hr
        
        final_tb_h_patch, final_tb_v_patch = rtm_model(
            t_patchtype, t_patchclass, t_dz_sno, t_forc_topo, t_htop, 
            t_tref, t_t_soisno, t_tleaf, t_wliq_soisno, t_wice_soisno, t_h2osoi, 
            t_snowdp, t_lai, t_sai, 
            t_wf_clay, t_wf_sand, t_wf_silt, t_BD_all, t_porsl, 
            t_sat_theta, t_sat_fghz,
            final_eps_surf, final_eps_all, final_hr
        )
        t_eff_array = rtm_model.t_eff[0]
        r_s_h, r_s_v = rtm_model.r_s[0], rtm_model.r_s[1]
        r_r_h, r_r_v = rtm_model.r_r[0], rtm_model.r_r[1]
        # 新增：提取三个不同模型的温度计算结果
        t_eff_wilheit_v = rtm_model.t_eff_wilheit[1]
        t_eff_holmes = rtm_model.t_eff_holmes[0]
        t_eff_wigneron = rtm_model.t_eff_wigneron[0]

    obs_tb_h_exp = np.repeat(obs['tb_h'], patch_dim)
    obs_tb_v_exp = np.repeat(obs['tb_v'], patch_dim)
    date_exp = np.repeat(obs['date'], patch_dim)
    time_index_exp = np.repeat(obs['time_index'], patch_dim)

    # sim_tb_h_mean = final_tb_h_patch.view(time_dim, patch_dim).mean(dim=1).cpu().numpy()
    # sim_tb_v_mean = final_tb_v_patch.view(time_dim, patch_dim).mean(dim=1).cpu().numpy()
    sim_tb_h_mean = torch.sum(final_tb_h_patch.view(time_dim, patch_dim) * t_weights, dim=1).cpu().numpy()
    sim_tb_v_mean = torch.sum(final_tb_v_patch.view(time_dim, patch_dim) * t_weights, dim=1).cpu().numpy()
    sim_tb_h_exp = np.repeat(sim_tb_h_mean, patch_dim)
    sim_tb_v_exp = np.repeat(sim_tb_v_mean, patch_dim)

    # === 新增：利用原本的经验公式计算传统的 Dobson 表层 beta 和 beta_i ===
    beta_surf_orig = (127.48 - 0.519 * surface_sand - 0.152 * surface_clay) / 100.0
    beta_i_surf_orig = (133.797 - 0.603 * surface_sand - 0.166 * surface_clay) / 100.0
    sigma_eff_org =  -1.645 + 1.939 * surface_BD_all - 0.02256 * surface_sand + 0.01594 * surface_clay
    
    # Export records modified to directly output the directly predicted surface dielectric real and imaginary parts
    df_out = pd.DataFrame({
        "date": date_exp,
        "time_index": time_index_exp,
        "patchclass": t_patchclass.cpu().numpy(),
        "patch_lat": patch_lat,
        "patch_lon": patch_lon,
        "patch_weights": np.tile(t_weights.cpu().numpy(), time_dim),
        "obs_tb_h": obs_tb_h_exp,
        "obs_tb_v": obs_tb_v_exp,
        "sim_tb_h": sim_tb_h_exp,        
        "sim_tb_v": sim_tb_v_exp,        
        "sim_tb_h_patch": final_tb_h_patch.cpu().numpy(),
        "sim_tb_v_patch": final_tb_v_patch.cpu().numpy(),
        
        # === 新增：将表层的 beta' 和 beta'' 写入 CSV === 
        # === 新增：原先 Dobson 公式计算出的传统 beta 参数 ===
        "beta_surf_r": final_beta_surf[:, 0].cpu().numpy(),
        "beta_surf_r_D85": beta_surf_orig.cpu().numpy(),

        "sigma_surf": final_beta_surf[:, 1].cpu().numpy(),  # 列名改为 sigma_surf_pred
        "sigma_surf_D85": sigma_eff_org.cpu().numpy(),
        
        "eps_surf_r": final_eps_surf.real.cpu().numpy(),
        "eps_surf_r_M09": eps_surf_M09.real.cpu().numpy(),
        "eps_surf_r_D85": eps_surf_D85.real.cpu().numpy(),
        "eps_surf_i": final_eps_surf.imag.cpu().numpy(),
        "eps_surf_i_M09":eps_surf_M09.imag.cpu().numpy(),
        "eps_surf_i_D85":eps_surf_D85.imag.cpu().numpy(),
        
        "hr": final_hr.cpu().numpy(),
        "surface_sm": surface_sm.cpu().numpy(),
        "surface_t": surface_t.cpu().numpy(),
        "surface_clay": surface_clay.cpu().numpy(),
        "surface_sand": surface_sand.cpu().numpy(),
        "lai": t_lai.cpu().numpy(),
        "sai": t_sai.cpu().numpy(),
        "htop": t_htop.cpu().numpy(),
        "surface_porsl": surface_porsl.cpu().numpy(),
        "surface_BD_all": surface_BD_all.cpu().numpy(),
        "forc_topo": t_forc_topo.cpu().numpy(),
        "ffrz": ffrz_array.cpu().numpy(),
        # === 修改与新增有效温度列 ===
        "t_eff_array": t_eff_array.cpu().numpy(),        # 默认保存的也是 Wilheit H极化
        "t_eff_wilheit_v": t_eff_wilheit_v.cpu().numpy(),
        "t_eff_holmes": t_eff_holmes.cpu().numpy(),
        "t_eff_wigneron": t_eff_wigneron.cpu().numpy(),
        "r_s_h": r_s_h.cpu().numpy(),
        "r_s_v": r_s_v.cpu().numpy(),
        "r_r_h": r_r_h.cpu().numpy(),
        "r_r_v": r_r_v.cpu().numpy(),
    })

    out_csv_path = os.path.join(output_dir, f"lat_{ease_lat}_lon_{ease_lon}_calibrate.csv")
    df_out.to_csv(out_csv_path, index=False, float_format="%.6f")
    print(f"💾 Step 5.2 Data successfully saved to: {out_csv_path}")
    
    # 1. Split the real and imaginary parts of the dielectric constant
    eps_real_all, eps_imag_all = final_eps_all.real, final_eps_all.imag
    eps_real_surf, eps_imag_surf = final_eps_surf.real, final_eps_surf.imag

    # 2. Matrix concatenation: directly prepend surf to all to form a [708, 11] matrix
    # The order of each row automatically becomes: surf -> all[0] -> all[1] -> ... -> all[9]
    combined_real = torch.cat([eps_real_surf.unsqueeze(1)], dim=1) #, eps_real_all
    combined_imag = torch.cat([eps_imag_surf.unsqueeze(1)], dim=1) # , eps_imag_all
    combined_sm = torch.cat([surface_sm.unsqueeze(1)], dim=1) # , sm_all
    combined_t = torch.cat([surface_t.unsqueeze(1)], dim=1)  # , t_all
    combined_clay = torch.cat([surface_clay.unsqueeze(1)], dim=1) # , clay_all
    combined_sand = torch.cat([surface_sand.unsqueeze(1)], dim=1) # , sand_all
    combined_porsl = torch.cat([surface_porsl.unsqueeze(1)], dim=1) # , porsl_all
    combined_BD_all = torch.cat([surface_BD_all.unsqueeze(1)], dim=1) # , BD_all_all

    # 3. Flatten by row and export
    df = pd.DataFrame(
        {
            "eps_real": combined_real.flatten().cpu().numpy(),
            "eps_imag": combined_imag.flatten().cpu().numpy(),
            "soil_moisture": combined_sm.flatten().cpu().numpy(),
            "temperature": combined_t.flatten().cpu().numpy(),
            "clay_fraction": combined_clay.flatten().cpu().numpy(),
            "sand_fraction": combined_sand.flatten().cpu().numpy(),
            "porosity": combined_porsl.flatten().cpu().numpy(),
            "bulk_density": combined_BD_all.flatten().cpu().numpy(),
        }
    ).to_csv(os.path.join(output_dir, f"lat_{ease_lat}_lon_{ease_lon}_soil_dielectric_data_11layers.csv"),
              index=False, float_format="%.6f")
    print('Dielectric constant related data has been outputted.\n')
    arr_base = t_eff_array.cpu().numpy() if hasattr(t_eff_array, "cpu") else t_eff_array
    # 计算 RMSE 的闭包函数
    def calc_rmse(arr1, arr2):
        # 如果是 PyTorch Tensor，先脱离计算图、移到 CPU 并转为 numpy
        if hasattr(arr1, "detach"):
            arr1 = arr1.detach().cpu().numpy()
        if hasattr(arr2, "detach"):
            arr2 = arr2.detach().cpu().numpy()
        return np.sqrt(np.mean((arr1 - arr2) ** 2))
    # 提取用于构建 DataFrame 的一维数组并计算 RMSE
    arr_base = t_eff_array.cpu().numpy() if hasattr(t_eff_array, "cpu") else t_eff_array
    rmse_wilheit_v = calc_rmse(t_eff_wilheit_v, arr_base)
    rmse_holmes = calc_rmse(t_eff_holmes, arr_base)
    rmse_wigneron = calc_rmse(t_eff_wigneron, arr_base)
    print("================================================================")
    print("📊 Effective Temperature (T_eff) Comparison RMSE Results:")
    print(f"  -> RMSE(t_eff_wilheit_v vs t_eff_array): {rmse_wilheit_v:.4f} K")
    print(f"  -> RMSE(t_eff_holmes    vs t_eff_array): {rmse_holmes:.4f} K")
    print(f"  -> RMSE(t_eff_wigneron  vs t_eff_array): {rmse_wigneron:.4f} K")
    print("================================================================")
    
    return best_loss_stage2


if __name__ == "__main__":
    INDEX_FILE = 'nc_patch_location_index.csv'
    csv_dir = '/home/liusy/store_global_forward/tb_for_EASE_open_lands'
    patch_map_file = '/home/liusy/storage_global_veg_wigneron/patch_map_EASE_open_lands.csv' 
    nc_dir = '/home/liusy/CoLM/outputs/global_veg_wigneron/forward_inputs_folder'
    output_dir = "/home/liusy/storage_global_veg_wigneron/tb_calibrate_try_1"
    csv_files = open('csv_files_list_for_training.csv', encoding='utf-8').read().splitlines()[:10]
    # 定义输出结果文件
    output_result_csv = 'grid_loss_results.csv'
    
    print("Loading global NC index table...")
    df_nc_index = pd.read_csv(INDEX_FILE)
    
    set_seed(seed=42)
    
    file_exists = os.path.exists(output_result_csv)
    with open(output_result_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['target_csv', 'best_loss'])

    for grid_id in range(len(csv_files)):
        # try:
            target_csv = csv_files[grid_id]
            ease_lat, ease_lon = map(float, os.path.basename(target_csv).split('_')[1:4:2])
            print(f"Processing grid file: {target_csv} (Longitude: {ease_lon}, Latitude: {ease_lat})")
            df_obs = pd.read_csv(target_csv)
            result = process_single_grid(df_nc_index, df_obs, patch_map_file, nc_dir)
            best_loss = run_step_5_2_calibration(result, ease_lat, ease_lon, output_dir)
            # 实时追加当前网格的结果
            with open(output_result_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([target_csv, best_loss])
            
            # break
        # except:
        #     print(f'Error in {target_csv}\n')
        #     continue
           
        
        
