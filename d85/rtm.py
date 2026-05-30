#-----------------------------------------------------------------------
# DESCRIPTION:
#    Forward modeling of brightness temperature observations
#    Adapted from MOD_DA_RTM.F90
#    (Differentiable PyTorch version for Data Assimilation / Parameter Inversion)
#
# AUTHOR/DATE/EMAIL:
#    Shuyue LIU, 04/2026, sliu347@connect.hkust-gz.edu.cn
#-----------------------------------------------------------------------
import torch
import torch.nn as nn
import math

class DifferentiableRTM(nn.Module):
    def __init__(self, def_da_rtm_diel=0, def_da_rtm_rough=0, def_da_rtm_veg=0, num_grids=1, maxsnl=5):
        super(DifferentiableRTM, self).__init__()
        
        self.maxsnl = maxsnl 
        
        # 物理常数 (Physical Constants)
        self.tfrz = 273.15           # freezing temperature [K]
        self.denh2o = 1000.0         # density of water [kg/m3]
        self.denice = 917.0          # density of ice [kg/m3]
        self.eps_w_inf = 4.9         # high frequency dielectric constant of water
        self.eps_0 = 8.8541878128e-12# permittivity of free space [F/m]
        self.mu0 = 4 * math.pi * 1e-7# permeability of free space [H/m]
        self.z0 = torch.sqrt(torch.tensor(self.mu0 / self.eps_0)) # impedance of free space [ohms]
        self.pi = math.pi
        self.C = 299792458.0         # speed of light in vacuum [m/s]
        self.rho_soil = 2.66         # density of solid soil material [g/cm3]
        self.jj = 1j  # 虚数单位
        
        # 选项配置
        self.def_da_rtm_diel = def_da_rtm_diel    # option for dielectric model
        self.def_da_rtm_rough = def_da_rtm_rough  # option for rough surface reflectivity
        self.def_da_rtm_veg = def_da_rtm_veg      # option for vegetation model (0: Wigneron, 1: Jackson, 2: Kirdyashev)
        self.rgh_surf = 2.2 # 来自 MOD_DA_Const.F90 的默认表面粗糙度
        self.tau_nadir = None
        self.gamma_p_h = None
        self.gamma_p_v = None
        self.tb_veg_h = None
        self.tb_veg_v = None
        self.t_surf = None
        self.t_deep = None
        self.t_eff =None
        self.r_s =None
        self.r_r =None 
        self.tb_soil = None
        self.tb_soil_d = None
        self.tb_soil_nd = None
        self.ew = None
        self.eps_x = None
        self.eps = None 
        self.ffrz = None
        self.eps_soil_nd_M09 = None
        self.eps_soil_nd_M09_all = None
        self.eps_soil_nd_D85 = None
     
        
        # 定义默认土壤各层厚度 (m)，通过 register_buffer 自动跟随模型挂载到 GPU
        self.register_buffer('dz_soi', torch.tensor([0.0175, 0.0276, 0.0455, 0.0750, 0.1236, 
                                                     0.2038, 0.3360, 0.5539, 0.9133, 1.5058]))
        
        # ====================================================================
        # IGBP 经验参数表 (大小为 18，索引 1-17 对应 IGBP)
        # ====================================================================
        self.register_buffer('tth', torch.tensor([0.0, 
            0.80, 1.00, 0.80, 0.49, 0.49, 1.00, 1.00, 1.00, 1.00, 1.00, 
            1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 2.00]))
        self.register_buffer('ttv', torch.tensor([0.0, 
            0.80, 1.00, 0.80, 0.46, 0.46, 1.00, 1.00, 1.00, 1.00, 1.00, 
            1.00, 2.00, 1.00, 2.00, 1.00, 1.00, 1.00]))

        self.register_buffer('b1', torch.tensor([0.0, 
            0.2600, 0.2260, 0.2600, 0.2260, 0.2260, 0.0375, 0.0375, 0.0375, 0.0375, 0.0375, 
            0.0000, 0.0500, 0.0000, 0.0500, 0.0000, 0.0000, 0.0500]))
        self.register_buffer('b2', torch.tensor([0.0, 
            0.0060, 0.0010, 0.0060, 0.0010, 0.0010, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]))
        self.register_buffer('b3', torch.tensor([0.0, 
            0.6900, 0.7000, 0.6900, 0.7000, 0.7000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]))

        self.register_buffer('w_CMEM', torch.tensor([0.0, 
            0.080, 0.095, 0.080, 0.070, 0.070, 0.050, 0.050, 0.050, 0.050, 0.050, 
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]))
            
        self.register_buffer('hr_SMAP', torch.tensor([0.0, 
            0.160, 0.160, 0.160, 0.160, 0.160, 0.110, 0.110, 0.125, 0.156, 0.156, 
            0.100, 0.108, 0.000, 0.130, 0.000, 0.150, 0.000]))
        self.register_buffer('hr_SMOS', torch.tensor([0.0, 
            0.300, 0.300, 0.300, 0.300, 0.300, 0.100, 0.100, 0.100, 0.100, 0.100, 
            0.100, 0.100, 0.100, 0.100, 0.000, 0.100, 0.000]))
        self.register_buffer('hr_P16', torch.tensor([0.0, 
            0.350, 0.460, 0.430, 0.450, 0.410, 0.260, 0.170, 0.350, 0.230, 0.130, 
            0.020, 0.170, 0.190, 0.220, 0.000, 0.020, 0.000]))


    # === 修改后 ===
    def forward(self, patchtype, patchclass, dz_sno,
                forc_topo, htop, 
                tref, t_soisno, tleaf, 
                wliq_soisno, wice_soisno, h2osoi, 
                snowdp, lai, sai, 
                wf_clay, wf_sand, wf_silt, BD_all, porsl, 
                sat_theta, sat_fghz,
                eps_pred_surf, eps_pred_all,
                hr_pred
                ):
        """
        前向传播函数 - 全 GPU 向量化执行
        """
        
        t_soi = t_soisno[:, self.maxsnl:]       
        
        wliq_sno = wliq_soisno[:, :self.maxsnl]
        wliq_soi = wliq_soisno[:, self.maxsnl:]
        
        wice_sno = wice_soisno[:, :self.maxsnl]
        wice_soi = wice_soisno[:, self.maxsnl:]
        
        # 将一维的层厚张量扩展至与 Batch 对齐
        dz_soi_batch = self.dz_soi.unsqueeze(0).expand(t_soi.shape[0], -1)

        # ==========================================
        # 1. 计算卫星相关参数
        # ==========================================
        theta = sat_theta
        fghz = sat_fghz
        f = fghz * 1e9                 
        omega = 2.0 * self.pi * f      
        lam = self.C / f               
        k = 2 * self.pi / lam          
        kcm = k / 100.0                
        kr = k * (0.5 * 1e-3)          
        
        # ==========================================
        # 2. 状态变量预处理
        # ==========================================
        wf_total = wf_clay + wf_sand + wf_silt
        w1, w2 = 0.0175, 0.0276        
        wtot = w1 + w2
        
        wf_clay_surf = (wf_clay[:, 0]/wf_total[:, 0]*w1 + wf_clay[:, 1]/wf_total[:, 1]*w2) / wtot * 100
        wf_sand_surf = (wf_sand[:, 0]/wf_total[:, 0]*w1 + wf_sand[:, 1]/wf_total[:, 1]*w2) / wtot * 100
        BD_all_surf = (BD_all[:, 0]*w1 + BD_all[:, 1]*w2) / wtot / 1000
        porsl_surf = (porsl[:, 0]*w1 + porsl[:, 1]*w2) / wtot
        
        t_surf = ((t_soi[:, 0]*w1 + t_soi[:, 1]*w2) / wtot) - self.tfrz
        t_deep = ((t_soi[:, 6]*(0.8289-0.5) + t_soi[:, 7]*(1.0-0.8289)) / 0.5) - self.tfrz
        self.t_deep, self.t_surf = t_deep, t_surf
        
        liq_surf = (wliq_soi[:, 0] + wliq_soi[:, 1]) / (wtot * self.denh2o)
        ice_surf = (wice_soi[:, 0] + wice_soi[:, 1]) / (wtot * self.denice)

        mask_sno_layer = wliq_sno >= 0.0
        lb = torch.sum(mask_sno_layer, dim=1) - 1 
        lb = torch.clamp(lb, min=0)
        
        lb_idx = lb.unsqueeze(1)
        wliq_sno_lb = torch.gather(wliq_sno, 1, lb_idx).squeeze(1)
        wice_sno_lb = torch.gather(wice_sno, 1, lb_idx).squeeze(1)
        dz_sno_lb = torch.gather(dz_sno, 1, lb_idx).squeeze(1)
        
        #############################################################################
        # 3. 大气模块 (Atmosphere module)
        #############################################################################
        tau_atm, tb_au, tb_ad = self.atm(forc_topo, tref, theta)
        
        #############################################################################
        # 4. 土壤模块 (Soil module)
        #############################################################################
        
        r_r,tb_soil = self.soil(patchclass, t_surf, t_deep, liq_surf, ice_surf, 
             wf_sand_surf, wf_clay_surf, BD_all_surf, porsl_surf, 
             theta, fghz, f, omega, kcm, kr, lam, self.rgh_surf,
             eps_pred_surf, eps_pred_all, 
             hr_pred, dz_soi_batch, t_soi, h2osoi, wf_clay)
        
        #############################################################################
        # 5. 植被与积雪模块 (Vegetation and snow module)
        #############################################################################
        has_veg = (lai + sai > 1e-6)  
        has_sno = (snowdp > 0.01)     
        
        dz_sno_lb_safe = torch.clamp(dz_sno_lb, min=1e-3)
        tmp_rho_snow = (wliq_sno_lb + wice_sno_lb) / (dz_sno_lb_safe*1e3)
        tmp_liq_snow = wliq_sno_lb * tmp_rho_snow / (dz_sno_lb_safe*1e3)
        
        condition = (tmp_liq_snow > 1.0) | (tmp_rho_snow > 1.0)
        rho_snow = torch.where(condition, torch.tensor(1.0, device=t_soi.device), tmp_rho_snow)
        liq_snow = torch.where(condition, wliq_sno_lb * rho_snow / (dz_sno_lb_safe*1e3), tmp_liq_snow)

        t_soi_0 = t_soi[:, 0]
        
        # A) 计算植被特性 (brightness temperature of vegetation) 分支处理
        if self.def_da_rtm_veg == 0:
            tb_veg, gamma_p = self.veg_wigneron(patchclass, lai, htop, snowdp, tleaf, theta)
        elif self.def_da_rtm_veg == 1:
            tb_veg, gamma_p = self.veg_jackson(patchclass, lai, htop, snowdp, tleaf, theta)
        else:
            tb_veg, gamma_p = self.veg_kirdyashev(patchclass, lai, htop, snowdp, tleaf, theta, f, omega, k)
        
        # B) 计算雪特性（情景 1：雪下是裸土）
        r_snow_A, tb_tos_A = self.snow(t_soi_0, t_soi_0, snowdp, rho_snow, liq_snow, r_r, theta, fghz, omega, k)
        
        # C) 计算雪特性（情景 2：雪下有植被）
        tb_tov_noad = tb_veg + tb_veg * gamma_p * r_r + tb_soil * gamma_p
        r_sn = 1.0 - tb_tov_noad / t_soi_0
        r_snow_B, tb_tos_B = self.snow(t_soi_0, t_soi_0, snowdp, rho_snow, liq_snow, r_sn, theta, fghz, omega, k)

        tb_tov_1 = tb_soil + tb_ad * r_r
        tb_tov_2 = tb_tos_A + tb_ad * r_snow_A
        tb_tov_3 = tb_veg + tb_veg * gamma_p * r_r + tb_soil * gamma_p + tb_ad * r_r * (gamma_p**2)
        
        tb_tov_4a = tb_tos_B + tb_ad * r_snow_B
        tb_tov_4b = tb_veg + tb_veg * gamma_p * r_snow_A + tb_tos_A * gamma_p + tb_ad * r_snow_A * (gamma_p**2)
        
        mask_htop = (htop < snowdp).unsqueeze(0).expand(2, -1) 
        tb_tov_4 = torch.where(mask_htop, tb_tov_4a, tb_tov_4b)

        mask_veg_2d = has_veg.unsqueeze(0).expand(2, -1)
        mask_sno_2d = has_sno.unsqueeze(0).expand(2, -1)
        
        tb_tov_no_veg = torch.where(mask_sno_2d, tb_tov_2, tb_tov_1)
        tb_tov_has_veg = torch.where(mask_sno_2d, tb_tov_4, tb_tov_3)
        tb_tov = torch.where(mask_veg_2d, tb_tov_has_veg, tb_tov_no_veg)
                
        #############################################################################
        # 6. 计算 TOA (Top of Atmosphere) 亮度温度
        #############################################################################
        tb_toa = tb_tov * torch.exp(-tau_atm) + tb_au
        
        mask_water = (patchtype >= 3).unsqueeze(0).expand(2, -1)
        tb_toa = torch.where(mask_water, torch.tensor(float('nan'), device=tb_toa.device), tb_toa)

        tb_toa_h, tb_toa_v = tb_toa[0], tb_toa[1]
        return tb_toa_h, tb_toa_v

    def atm(self, z, tref, theta):
        tau_atm = torch.exp(-3.9262 - 0.2211*z/1000. - 0.00369*tref) / torch.cos(theta)
        gossat = torch.exp(-tau_atm)
        t_eq = torch.exp(4.9274 + 0.002195*tref)
        t_sky = 2.7 
        tb_au = t_eq * (1. - gossat)
        tb_ad = t_eq * (1. - gossat) + t_sky * gossat
        return tau_atm, torch.stack([tb_au, tb_au]), torch.stack([tb_ad, tb_ad])

    def soil(self, patchclass, t_surf, t_deep, liq_surf, ice_surf, 
             wf_sand_surf, wf_clay_surf, BD_all_surf, porsl_surf, 
             theta, fghz, f, omega, kcm, kr, lam, rgh_surf,
             eps_pred_surf, eps_pred_all, hr_pred,
             dz_soi, t_soi, wliq_soi, wf_clay):
        
        is_desert = (liq_surf < 0.02) & (wf_sand_surf > 90)
        ffrz = torch.where((liq_surf + ice_surf) <= 0.0, 
                           torch.zeros_like(liq_surf), 
                           ice_surf / torch.clamp(liq_surf + ice_surf, min=1e-6))
        self.ffrz = ffrz

        eps_f = torch.complex(torch.full_like(liq_surf, 5.0), torch.full_like(liq_surf, 0.5))
        ew = torch.where(ffrz > 0.95, 
                         self.diel_ice(t_surf, fghz), 
                         self.diel_water_soil(-1, liq_surf, t_surf, wf_sand_surf, wf_clay_surf, BD_all_surf, 0.0, f, omega))
            
        # if self.def_da_rtm_diel == 0:
        #     eps_soil_nd = self.diel_soil_W80(ew, liq_surf, wf_sand_surf, wf_clay_surf, porsl_surf)
        # elif self.def_da_rtm_diel == 1:
        #     eps_soil_nd = self.diel_soil_D85(ew, liq_surf, wf_sand_surf, wf_clay_surf, BD_all_surf)
        # elif self.def_da_rtm_diel == 2:
        #     eps_soil_nd = self.diel_soil_M04(liq_surf, wf_clay_surf, f)
        # elif self.def_da_rtm_diel == 3:
        #     eps_soil_nd = self.diel_soil_M09(liq_surf, t_surf, wf_clay_surf, f)
        # else:
        #     # 引入网络直接预测的各项中间变量计算表层介电常数 (直接使用外部传进来的 _surf)
        #     eps_soil_nd = self.diel_soil_Debye_framework(liq_surf, f, 
        #                                      znd_surf, zkd_surf, zxmvt_surf, zep0b_surf, ztaub_surf, zsigmab_surf, zep0u_surf, ztauu_surf, zsigmau_surf,)
        self.eps_soil_nd_D85 = self.diel_soil_D85(ew, liq_surf, wf_sand_surf, wf_clay_surf, BD_all_surf)
        self.eps_soil_nd_M09 = self.diel_soil_M09(liq_surf, t_surf, wf_clay_surf, f)
        # 网络直接预测的表层介电常数
        eps_soil_nd = eps_pred_surf
        eps_soil_nd = eps_soil_nd*(1.-ffrz) + eps_f*ffrz
        
        eps_soil_d_real = torch.full_like(liq_surf, 2.53) + (2.79 - 2.53)/(1 + (fghz/0.27)**2)
        eps_soil_d_imag = (2.79 - 2.53)*(fghz/0.27)/(1 + (fghz/0.27)**2) + 0.002
        eps_soil_d = torch.complex(eps_soil_d_real, eps_soil_d_imag)
        
        
        eps_soil = torch.where(is_desert, eps_soil_d, eps_soil_nd)
        
        # 利用 Lv 模型计算多层有效温度，此处将 10 层的 pred 矩阵完整传入
        # 
        # 利用 Lv 模型计算多层有效温度，传入 10 层介电常数预测结果
        t_eff = self.eff_soil_temp_Lv(dz_soi, t_soi, wliq_soi, f, lam, wf_clay,
                                      eps_pred_all)
                                      
        self.t_eff = t_eff
        
        g = torch.sqrt(eps_soil - torch.sin(theta)**2)
        r_s_h = torch.abs((torch.cos(theta) - g)/(torch.cos(theta) + g))**2
        r_s_v = torch.abs((torch.cos(theta)*eps_soil - g)/(torch.cos(theta)*eps_soil + g))**2
        r_s = torch.stack([r_s_h, r_s_v])
        self.r_s = r_s
        
        r_r = self.rough_reflectivity(is_desert, patchclass, r_s, theta, fghz, kcm, rgh_surf, hr_pred) 
        self.r_r = r_r

        tb_soil_d = self.desert(t_eff, r_r, eps_soil, kr)
        tb_soil_nd = t_eff * (1 - r_r)
        self.tb_soil_d = tb_soil_d
        self.tb_soil_nd = tb_soil_nd
        
        tb_soil = torch.where(is_desert.unsqueeze(0).expand(2, -1), tb_soil_d, tb_soil_nd)
        self.tb_soil = tb_soil
            
        return r_r, tb_soil
    
    def eff_soil_temp_Lv(self, dz_soi, t_soi, wc_soi, f, lam, wf_clay,
                         eps_pred_all):
        f_2d = f.unsqueeze(1)
        lam_2d = lam.unsqueeze(1)

        # 1. 直接使用网络给定的 10 层介电常数
        eps = eps_pred_all
        eps_r = eps.real
        eps_i = torch.abs(eps.imag)

        # 2. 计算每层的 B_i 吸收系数 (使用扩展后的 lam_2d)
        B_i = dz_soi * (4.0 * self.pi / lam_2d) * (eps_i / (2.0 * torch.sqrt(eps_r)))

        exp_B = torch.exp(-B_i)

        # 3. 计算到达每层顶部的衰减乘积 prod_term
        # 利用 cumsum 替代 for 循环中的累乘: prod_term_i = exp( - sum_{j=0}^{i-1} B_j )
        tau_top = torch.cat([torch.zeros_like(B_i[:, :1]), torch.cumsum(B_i[:, :-1], dim=1)], dim=1)
        prod_term = torch.exp(-tau_top)

        # 4. 计算每层的发射权重
        weights = (1.0 - exp_B) * prod_term
        # 最后一层积分至无穷大，权重直接等于剩余的透射率
        weights[:, -1] = prod_term[:, -1]

        # 5. 求有效温度 (相加)
        t_eff_val = torch.sum(t_soi * weights, dim=1)

        # 返回堆叠的极化维度
        return torch.stack([t_eff_val, t_eff_val])
    
    
    def diel_soil_M04(self, wc, wf_clay, f):
        """
        Mironov 2004 介电常数模型
        输入:
            wc: 体积含水量 (m3/m3)
            wf_clay: 黏土质量百分比 (%)
            f: 频率 (Hz)
        """
        wf_clay_frac = wf_clay / 100.0
        
        # 1. 干土的折射率 (Refractive Index, RI) 与归一化吸收系数 (NAC)
        znd = 1.634 - 0.539 * wf_clay_frac + 0.2748 * (wf_clay_frac ** 2)
        zkd = 0.03952 - 0.04038 * wf_clay_frac

        # 2. 最大束缚水体积分数
        zxmvt = 0.02863 + 0.30673 * wf_clay_frac

        # 3. 束缚水 (Bound water) 介电参数
        zep0b = 79.8 - 85.4 * wf_clay_frac + 32.7 * (wf_clay_frac ** 2)
        ztaub = 1.062e-11 + 3.450e-12 * wf_clay_frac
        zsigmab = 0.3112 + 0.467 * wf_clay_frac
        
        # 4. 自由水 (Unbound/Free water) 介电参数
        zep0u = 100.0
        ztauu = 8.5e-12
        zsigmau = 0.3631 + 1.217 * wf_clay_frac

        # 5. 计算水的复介电常数 (束缚水 & 自由水)
        zcxb = (zep0b - self.eps_w_inf) / (1.0 + (2.0 * self.pi * f * ztaub)**2)
        zepwbx = self.eps_w_inf + zcxb
        zepwby = zcxb * (2.0 * self.pi * f * ztaub) + zsigmab / (2.0 * self.pi * self.eps_0 * f)

        zcxu = (zep0u - self.eps_w_inf) / (1.0 + (2.0 * self.pi * f * ztauu)**2)
        zepwux = self.eps_w_inf + zcxu
        zepwuy = zcxu * (2.0 * self.pi * f * ztauu) + zsigmau / (2.0 * self.pi * self.eps_0 * f)

        # 6. 计算水的折射率 (束缚水 & 自由水)
        sqrt_2 = math.sqrt(2.0)
        znb = torch.sqrt(torch.sqrt(zepwbx**2 + zepwby**2) + zepwbx) / sqrt_2
        zkb = torch.sqrt(torch.sqrt(zepwbx**2 + zepwby**2) - zepwbx) / sqrt_2
        znu = torch.sqrt(torch.sqrt(zepwux**2 + zepwuy**2) + zepwux) / sqrt_2
        zku = torch.sqrt(torch.sqrt(zepwux**2 + zepwuy**2) - zepwux) / sqrt_2

        # 7. 计算土壤综合折射率
        zxmvt2 = torch.minimum(wc, zxmvt)
        zflag = (wc >= zxmvt).to(wc.dtype)  # 当含水量大于最大束缚水时，存在自由水

        znm = znd + (znb - 1.0) * zxmvt2 + (znu - 1.0) * (wc - zxmvt) * zflag
        zkm = zkd + zkb * zxmvt2 + zku * (wc - zxmvt) * zflag

        # 8. 转化为介电常数实部和虚部
        zepmx = znm**2 - zkm**2
        zepmy = 2.0 * znm * zkm

        return torch.complex(zepmx, zepmy)

    def diel_ice(self, t, fghz):
        tk = t + self.tfrz
        betam = (0.0207/tk)*(torch.exp(335./tk)/((torch.exp(335./tk) - 1.)**2.)) + 1.16e-11*(fghz**2.)
        dbeta = torch.exp(-10.02 + 0.0364*t)
        beta = betam + dbeta
        t_inv = 300./tk - 1
        alpha = (0.00504 + 0.0062*t_inv)*torch.exp(-22.1*t_inv)
        eps_i_r = 3.1884 + 9.1e-4*t
        eps_i_i = alpha/fghz + beta*fghz
        return torch.complex(eps_i_r, -eps_i_i)


    # 注意函数签名这里新增了 sigma_pred=None
    def diel_water_soil(self, water_type, swc, t, wf_sand, wf_clay, BD_all, sal, f, omega, sigma_pred=None):
        eps_w_s = 87.134 - 1.949e-1 * t - 1.276e-2 * t**2 + 2.491e-4 * t**3
        tau_w = 1.768e-11 - 6.068e-13 * t + 1.104e-14 * t**2 - 8.111e-17 * t**3

        if water_type == 0:
            return self.eps_w_inf + (eps_w_s - self.eps_w_inf) / (1.0 - self.jj * omega * tau_w)
        elif water_type == 1:
            a_sal = 1.000 + 1.613e-5 * sal * t - 3.656e-3 * sal + 3.210e-5 * sal**2 - 4.232e-7 * sal**3
            b_sal = 1.000 + 2.282e-5 * sal * t - 7.638e-4 * sal - 7.760e-6 * sal**2 + 1.105e-8 * sal**3
            eps_w_s_sal = eps_w_s * a_sal
            tau_w_sal = tau_w * b_sal
            sigma = 0.1825 * sal - 0.1461 * sal**2 + 0.0209 * sal**3 
            term_debye = (eps_w_s_sal - self.eps_w_inf) / (1.0 - self.jj * omega * tau_w_sal)
            term_cond = self.jj * sigma / (omega * self.eps_0)
            return self.eps_w_inf + term_debye + term_cond
        else:
            # ⬇️ 这里是主要的修改点 ⬇️
            if sigma_pred is not None:
                sigma_soil = sigma_pred
            else:
                sigma_soil = -1.645 + 1.939 * BD_all - 0.02256 * wf_sand + 0.01594 * wf_clay
                sigma_soil = torch.clamp(sigma_soil, min=0.0) 
            # ⬆️ 修改结束 ⬆️
            wc = torch.clamp(swc, min=0.001)
            term_debye = (eps_w_s - self.eps_w_inf) / (1.0 - self.jj * omega * tau_w)
            term_cond_soil = self.jj * sigma_soil / (omega * self.eps_0) * (self.rho_soil - BD_all) / (self.rho_soil * wc)
            return self.eps_w_inf + term_debye + term_cond_soil
    
    
    def diel_soil_W80(self, ew, wc, wf_sand, wf_clay, porsl):
        wp = 0.06774 - 0.00064*wf_sand + 0.00478*wf_clay
        gamma = -0.57*wp + 0.481
        wt = 0.49*wp + 0.165
        
        # 修复点：使用 full_like / zeros_like 确保常数张量继承传入变量的 GPU device
        eps_a = torch.complex(torch.full_like(wc, 1.0), torch.zeros_like(wc)) 
        eps_r = torch.complex(torch.full_like(wc, 5.5), torch.full_like(wc, 0.2)) 
        eps_i = torch.complex(torch.full_like(wc, 3.2), torch.full_like(wc, 0.1)) 
        
        is_le = wc <= wt
        eps_x_le = eps_i + (ew - eps_i)*(wc/wt)*gamma               
        eps_le = wc*eps_x_le + (porsl - wc)*eps_a + (1.-porsl)*eps_r 
        
        eps_x_gt = eps_i + (ew - eps_i)*gamma                       
        eps_gt = wt*eps_x_gt + (wc - wt)*ew + (porsl - wc)*eps_a + (1.-porsl)*eps_r 
        
        eps = torch.where(is_le, eps_le, eps_gt)
        alpha = torch.clamp(100.*wp, max=26.0)
        ecl = alpha * wc**2                                         
        eps = eps + self.jj * ecl                                   
        return eps
    
    def diel_soil_D85(self, ew, swc, wf_sand, wf_clay, BD_all, beta_pred=None, beta_i_pred=None):
        wc = torch.clamp(swc, min=0.001)
        alphas = 0.65
        eps_s = (1.01 + 0.44 * self.rho_soil)**2.0 - 0.062                      
        # 如果传入了神经网络预测的 beta'，则使用预测值；否则使用原经验公式
        if beta_pred is not None:
            beta = beta_pred
        else:
            beta = (127.48 - 0.519 * wf_sand - 0.152 * wf_clay) / 100.0             
        eaa = 1.0 + (BD_all / self.rho_soil) * (eps_s**alphas - 1.0) + (wc**beta) * (ew.real**alphas) - wc 
        epsr = eaa ** (1.0/alphas)                                              
        # 如果传入了神经网络预测的 beta''，则使用预测值；否则使用原经验公式
        if beta_i_pred is not None:
            beta_i = beta_i_pred
        else:
            beta_i = (133.797 - 0.603 * wf_sand - 0.166 * wf_clay) / 100.0          
        eaa_i = (wc**beta_i) * (torch.abs(ew.imag)**alphas)                     
        epsi = eaa_i ** (1.0/alphas)                                            
        return torch.complex(epsr, epsi)
    
    
        
    def diel_soil_Debye_framework(self, wc, f, 
                      znd, zkd, zxmvt, zep0b, ztaub, zsigmab, zep0u, ztauu, zsigmau):
        # 移除了所有依赖粘土含量拟合的内部常数，完全依赖网络外部输入的参数
        
        # zep0u = 100.0                                                           
        # ztauu = 8.5e-12                                                         

        zcxb = (zep0b - self.eps_w_inf) / (1.0 + (2.0 * self.pi * f * ztaub)**2)
        zepwbx = self.eps_w_inf + zcxb                                          
        zepwby = zcxb * (2.0 * self.pi * f * ztaub) + zsigmab / (2.0 * self.pi * self.eps_0 * f) 

        zcxu = (zep0u - self.eps_w_inf) / (1.0 + (2.0 * self.pi * f * ztauu)**2)
        zepwux = self.eps_w_inf + zcxu                                          
        zepwuy = zcxu * (2.0 * self.pi * f * ztauu) + zsigmau / (2.0 * self.pi * self.eps_0 * f)

        sqrt_2 = math.sqrt(2.0)
        
        inner_nb = torch.clamp(torch.sqrt(zepwbx**2 + zepwby**2) + zepwbx, min=1e-12)
        inner_kb = torch.clamp(torch.sqrt(zepwbx**2 + zepwby**2) - zepwbx, min=1e-12)
        znb = torch.sqrt(inner_nb) / sqrt_2                                     
        zkb = torch.sqrt(inner_kb) / sqrt_2                                     

        inner_nu = torch.clamp(torch.sqrt(zepwux**2 + zepwuy**2) + zepwux, min=1e-12)
        inner_ku = torch.clamp(torch.sqrt(zepwux**2 + zepwuy**2) - zepwux, min=1e-12)
        znu = torch.sqrt(inner_nu) / sqrt_2                                     
        zku = torch.sqrt(inner_ku) / sqrt_2                                     

        zxmvt2 = torch.minimum(wc, zxmvt)
        zflag = (wc >= zxmvt).to(wc.dtype)

        znm = znd + (znb - 1.0) * zxmvt2 + (znu - 1.0) * (wc - zxmvt) * zflag   
        zkm = zkd + zkb * zxmvt2 + zku * (wc - zxmvt) * zflag                   

        zepmx = znm**2 - zkm**2                                                 
        zepmy = 2.0 * znm * zkm                                                 

        return torch.complex(zepmx, zepmy)

    def diel_soil_M09(self, wc, t, wf_clay, f):
        """
        Calculate the dielectric constant of wet soil based on Mironov 2009 model.
        Input:
            wc      : soil moisture (m3/m3) - Tensor
            t       : temperature (Celsius) - Tensor
            wf_clay : clay content (%) - Tensor
            f       : frequency (Hz) - Tensor or Float
        Output:
            complex dielectric constant (relative permittivity) - Tensor
        """
        # 确保输入至少是张量，并获取当前设备(CPU/GPU)和数据类型，防止跨设备报错
        if not isinstance(wf_clay, torch.Tensor):
            wf_clay = torch.tensor(wf_clay, dtype=torch.float32)
        device = wf_clay.device
        dtype = wf_clay.dtype

        # 将标量转换为与输入同设备、同类型的 Tensor
        e0u_val = torch.tensor(100.0, device=device, dtype=dtype)
        ts_val = torch.tensor(20.0, device=device, dtype=dtype)

        # temperature in Kelvin
        tk = t + self.tfrz

        # --- dry soil refractive index & NAC ---
        nd = 1.634 - 0.539e-2 * wf_clay + 0.2748e-4 * (wf_clay ** 2)
        kd = 0.03952 - 0.04038e-2 * wf_clay

        # maximum bound water fraction
        mvt = 0.02863 + 0.30673e-2 * wf_clay

        # --- bound water parameters ---
        e0b = 79.8 - 85.4e-2 * wf_clay + 32.7e-4 * (wf_clay ** 2)
        Bb = (8.67e-19 - 0.00126e-2 * wf_clay + 0.00184e-4 * (wf_clay ** 2)
            - 9.77e-10 * (wf_clay ** 3) - 1.39e-15 * (wf_clay ** 4))
        Bsgb = (0.0028 + 0.02094e-2 * wf_clay - 0.01229e-4 * (wf_clay ** 2)
                - 5.03e-22 * (wf_clay ** 3) + 4.163e-24 * (wf_clay ** 4))

        # static dielectric constant of bound water (eb0)
        # e0b 是由 wf_clay (Tensor) 计算出来的，本身就是 Tensor，可以直接 clamp
        Fb = torch.log(torch.clamp((e0b - 1.0) / (e0b + 2.0), min=1e-8))
        exp_term_b = torch.exp(Fb - Bb * (t - ts_val))
        eb0 = (1.0 + 2.0 * exp_term_b) / (1.0 - exp_term_b)

        # relaxation time of bound water
        dHbR = (1467.0 + 2697e-2 * wf_clay - 980e-4 * (wf_clay ** 2)
                + 1.368e-10 * (wf_clay ** 3) - 8.61e-13 * (wf_clay ** 4))
        dSbR = (0.888 + 9.7e-2 * wf_clay - 4.262e-4 * (wf_clay ** 2)
                + 6.79e-21 * (wf_clay ** 3) + 4.263e-22 * (wf_clay ** 4))
        taub = 48e-12 * torch.exp(dHbR / tk - dSbR) / tk

        # conductivity of bound water
        sigmabt = 0.3112 + 0.467e-2 * wf_clay
        sigmab = sigmabt + Bsgb * (t - ts_val)

        # --- unbound (free) water parameters ---
        Bu = (1.11e-4 - 1.603e-7 * wf_clay + 1.239e-9 * (wf_clay ** 2)
            + 8.33e-13 * (wf_clay ** 3) - 1.007e-14 * (wf_clay ** 4))
        Bsgu = (0.00108 + 0.1413e-2 * wf_clay - 0.2555e-4 * (wf_clay ** 2)
                + 0.2147e-6 * (wf_clay ** 3) - 0.0711e-8 * (wf_clay ** 4))

        # static dielectric constant of free water (eu0)
        # 【修改重点】这里将原先的 e0u 替换为 Tensor 类型的 e0u_val
        Fu = torch.log(torch.clamp((e0u_val - 1.0) / (e0u_val + 2.0), min=1e-8))
        exp_term_u = torch.exp(Fu - Bu * (t - ts_val))
        eu0 = (1.0 + 2.0 * exp_term_u) / (1.0 - exp_term_u)

        # relaxation time of free water
        dHuR = (2231.0 - 143.1e-2 * wf_clay + 223.2e-4 * (wf_clay ** 2)
                - 142.1e-6 * (wf_clay ** 3) + 27.14e-8 * (wf_clay ** 4))
        dSuR = (3.649 - 0.4894e-2 * wf_clay + 0.763e-4 * (wf_clay ** 2)
                - 0.4859e-6 * (wf_clay ** 3) + 0.0928e-8 * (wf_clay ** 4))
        tauu = 48e-12 * torch.exp(dHuR / tk - dSuR) / tk

        # conductivity of free water
        sigmaut = 0.05 + 1.4 * (1.0 - (1.0 - wf_clay * 1e-2) ** 4.664)
        sigmau = sigmaut + Bsgu * (t - ts_val)

        # --- dielectric constant of bound and free water (Debye relaxation) ---
        cxb = (eb0 - self.eps_w_inf) / (1.0 + (2.0 * self.pi * f * taub) ** 2)
        eb_r = self.eps_w_inf + cxb
        eb_i = cxb * (2.0 * self.pi * f * taub) + sigmab / (2.0 * self.pi * self.eps_0 * f)

        cxu = (eu0 - self.eps_w_inf) / (1.0 + (2.0 * self.pi * f * tauu) ** 2)
        eu_r = self.eps_w_inf + cxu
        eu_i = cxu * (2.0 * self.pi * f * tauu) + sigmau / (2.0 * self.pi * self.eps_0 * f)

        # --- refractive index and NAC of water components ---
        # 【修改重点】将原先的 math.sqrt(2.0) 改为了 torch.sqrt()，保持纯 Tensor 环境
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
        nb = torch.sqrt(torch.sqrt(eb_r ** 2 + eb_i ** 2) + eb_r) / sqrt_2
        kb = torch.sqrt(torch.sqrt(eb_r ** 2 + eb_i ** 2) - eb_r) / sqrt_2
        nu = torch.sqrt(torch.sqrt(eu_r ** 2 + eu_i ** 2) + eu_r) / sqrt_2
        ku = torch.sqrt(torch.sqrt(eu_r ** 2 + eu_i ** 2) - eu_r) / sqrt_2

        # --- soil refractive index (bound water regime vs free water regime) ---
        is_le_mvt = wc <= mvt
        nm_le = nd + (nb - 1.0) * wc
        km_le = kd + kb * wc
        nm_gt = nd + (nb - 1.0) * mvt + (nu - 1.0) * (wc - mvt)
        km_gt = kd + kb * mvt + ku * (wc - mvt)

        nm = torch.where(is_le_mvt, nm_le, nm_gt)
        km = torch.where(is_le_mvt, km_le, km_gt)

        # --- complex dielectric constant ---
        eps_r = nm ** 2 - km ** 2
        eps_i = 2.0 * nm * km

        return torch.complex(eps_r, eps_i)

    def rough_reflectivity(self, is_desert, patchclass, r_s, theta, fghz, kcm, rgh_surf, hr_pred):
        p_class = patchclass.long()
        hr = hr_pred  
        
        Q = torch.where(fghz < 2.0, 
                        torch.zeros_like(fghz),                   
                        0.35 * (1.0 - torch.exp(-0.6 * (rgh_surf**2) * fghz)))  

        if self.def_da_rtm_rough == 0:
            # hr = (2.0 * kcm * rgh_surf)**2.0
            nrh = torch.zeros_like(r_s[0])
            nrv = torch.zeros_like(r_s[0])
        elif self.def_da_rtm_rough == 1:
            # hr = self.hr_SMOS[p_class]  
            nrh = torch.full_like(r_s[0], 2.0)
            nrv = torch.zeros_like(r_s[0])
        elif self.def_da_rtm_rough == 2:
            # hr = self.hr_SMAP[p_class]
            nrh = torch.full_like(r_s[0], 2.0)
            nrv = torch.full_like(r_s[0], 2.0)
        elif self.def_da_rtm_rough == 3:
            # hr = self.hr_P16[p_class]
            nrh = torch.full_like(r_s[0], -1.0)
            nrv = torch.full_like(r_s[0], -1.0)
        else:
            # hr = (2.0 * kcm * rgh_surf)**2.0
            nrh = torch.zeros_like(r_s[0])
            nrv = torch.zeros_like(r_s[0])
            
        r_r_h = (Q * r_s[1] + (1.0 - Q) * r_s[0]) * torch.exp(-hr * (torch.cos(theta)**nrh))
        r_r_v = (Q * r_s[0] + (1.0 - Q) * r_s[1]) * torch.exp(-hr * (torch.cos(theta)**nrv))
        
        r_r_rough = torch.stack([r_r_h, r_r_v])
        
        is_desert_2d = is_desert.unsqueeze(0).expand(2, -1)
        r_r_final = torch.where(is_desert_2d, r_s, r_r_rough)

        return r_r_final

    def desert(self, t_soil, r_r, eps, kr):
        f0 = 0.7
        y_r = (eps.real - 1) / (eps.real + 2)                                   
        y_i = 3 * eps.imag / (eps.real + 2)**2                                  
        w = ((1 - f0)**4 * kr**3 * y_r**2) / ((1 - f0)**4 * kr**3 * y_r**2 + 1.5*(1 + 2*f0)**2 * y_i) 
        g = 0.23 * kr**2                                                        
        a = torch.sqrt((1 - w) / (1 - w*g))                                     
        em = (1 - r_r) * (2*a / ((1 + a) - (1 - a)*r_r))                        
        return t_soil * em

    def veg_wigneron(self, patchclass, lai, htop, snowdp, tleaf, theta):
        p_class = patchclass.long()
        b1_val = self.b1[p_class]
        b2_val = self.b2[p_class]
        b3_val = self.b3[p_class]
        tth_val = self.tth[p_class]
        ttv_val = self.ttv[p_class]
        w_cmem = self.w_CMEM[p_class]
        
        tau_nadir = torch.where(htop < snowdp, b1_val * lai + b2_val, b3_val)
        tau_veg_h = tau_nadir * (torch.cos(theta)**2 + tth_val * torch.sin(theta)**2) 
        tau_veg_v = tau_nadir * (torch.cos(theta)**2 + ttv_val * torch.sin(theta)**2) 
        
        gamma_p_h = torch.exp(-tau_veg_h / torch.cos(theta))                    
        gamma_p_v = torch.exp(-tau_veg_v / torch.cos(theta))                    
        
        tb_veg_h = (1.0 - w_cmem) * (1.0 - gamma_p_h) * tleaf
        tb_veg_v = (1.0 - w_cmem) * (1.0 - gamma_p_v) * tleaf
        
        self.tau_nadir = tau_nadir
        self.gamma_p_h = gamma_p_h
        self.gamma_p_v = gamma_p_v
        self.tb_veg_h = tb_veg_h
        self.tb_veg_v = tb_veg_v
        
        return torch.stack([tb_veg_h, tb_veg_v]), torch.stack([gamma_p_h, gamma_p_v])

    def veg_jackson(self, patchclass, lai, htop, snowdp, tleaf, theta):
        p_class = patchclass.long()
        b1_val = self.b1[p_class]
        b2_val = self.b2[p_class]
        b3_val = self.b3[p_class]
        w_cmem = self.w_CMEM[p_class]
        
        tau_nadir = torch.where(htop < snowdp, b1_val * lai + b2_val, b3_val)
        
        gamma_p = torch.exp(-tau_nadir / torch.cos(theta))
        tb_veg = (1.0 - w_cmem) * (1.0 - gamma_p) * tleaf
        
        return torch.stack([tb_veg, tb_veg]), torch.stack([gamma_p, gamma_p])

    def veg_kirdyashev(self, patchclass, lai, htop, snowdp, tleaf, theta, f, omega, k):
        p_class = patchclass.long()
        w_cmem = self.w_CMEM[p_class]
        
        vwc = lai * 0.5 
        a_geo_h = 2.0 / 3.0
        a_geo_v = 2.0 / 3.0

        eps_vw = self.diel_water_soil(0, torch.zeros_like(vwc), tleaf - self.tfrz, torch.zeros_like(vwc), torch.zeros_like(vwc), torch.zeros_like(vwc), 0.0, f, omega)
        eps_vw_i = torch.abs(eps_vw.imag)
        
        tau_veg_h = a_geo_h * k * (vwc / self.denh2o) * eps_vw_i
        tau_veg_v = a_geo_v * k * (vwc / self.denh2o) * eps_vw_i
        
        gamma_p_h = torch.exp(-tau_veg_h / torch.cos(theta))
        gamma_p_v = torch.exp(-tau_veg_v / torch.cos(theta))
        
        tb_veg_h = (1.0 - w_cmem) * (1.0 - gamma_p_h) * tleaf
        tb_veg_v = (1.0 - w_cmem) * (1.0 - gamma_p_v) * tleaf
        
        return torch.stack([tb_veg_h, tb_veg_v]), torch.stack([gamma_p_h, gamma_p_v])

    def snow(self, t_snow, t, snowdp, rho_snow, liq_snow, r_sn, theta, fghz, omega, k):
        eps_i = self.diel_ice(t_snow - self.tfrz, fghz)
        eps_i_r = eps_i.real
        eps_i_i = -eps_i.imag  
        
        sal_snow = torch.zeros_like(t_snow) 
        eps_i_is = 0.0026 / fghz + 0.00023 * (fghz**0.87)                       
        eps_i_ip = 6e-4 / fghz + 6.5e-5 * (fghz**1.07)                          
        eps_i_i = eps_i_i + (eps_i_is - eps_i_ip) * sal_snow / 13.0             

        rho_i = 0.916 
        rho_ds = (rho_snow - liq_snow) / torch.clamp(1.0 - liq_snow, min=1e-3)
        
        eps_ds_r = 1.0 + 1.58 * rho_ds / (1.0 - 0.365 * rho_ds)
        eps_ds_i = 3.0 * (rho_ds / rho_i) * eps_i_i * (eps_ds_r**2) * (2.0 * eps_ds_r + 1.0) / \
                   ((eps_i_r + 2.0 * eps_ds_r) * (eps_i_r + 2.0 * eps_ds_r**2)) 

        f0w = torch.full_like(t_snow, 9.0) 
        eps_w_s = torch.full_like(t_snow, 88.0) 
        aa, bb, cc = 0.005, 0.4975, 0.4975
        
        fa = f0w * (1.0 + (aa * (eps_w_s - self.eps_w_inf) / (eps_ds_r + (aa * (self.eps_w_inf - eps_ds_r)))))
        fb = f0w * (1.0 + (bb * (eps_w_s - self.eps_w_inf) / (eps_ds_r + (bb * (self.eps_w_inf - eps_ds_r)))))
        fc = f0w * (1.0 + (cc * (eps_w_s - self.eps_w_inf) / (eps_ds_r + (cc * (self.eps_w_inf - eps_ds_r)))))

        eps_a_inf = (liq_snow * (self.eps_w_inf - eps_ds_r) / 3.0) / (1.0 + aa * ((self.eps_w_inf / eps_ds_r) - 1.0))
        eps_b_inf = (liq_snow * (self.eps_w_inf - eps_ds_r) / 3.0) / (1.0 + bb * ((self.eps_w_inf / eps_ds_r) - 1.0))
        eps_c_inf = (liq_snow * (self.eps_w_inf - eps_ds_r) / 3.0) / (1.0 + cc * ((self.eps_w_inf / eps_ds_r) - 1.0))

        eps_a_s = (liq_snow / 3.0) * (eps_w_s - eps_ds_r) / (1.0 + aa * ((eps_w_s / eps_ds_r) - 1.0))
        eps_b_s = (liq_snow / 3.0) * (eps_w_s - eps_ds_r) / (1.0 + bb * ((eps_w_s / eps_ds_r) - 1.0))
        eps_c_s = (liq_snow / 3.0) * (eps_w_s - eps_ds_r) / (1.0 + cc * ((eps_w_s / eps_ds_r) - 1.0))

        eps_a = eps_a_inf + (eps_a_s - eps_a_inf) / (1.0 + self.jj * fghz / fa)
        eps_b = eps_b_inf + (eps_b_s - eps_b_inf) / (1.0 + self.jj * fghz / fb)
        eps_c = eps_c_inf + (eps_c_s - eps_c_inf) / (1.0 + self.jj * fghz / fc)

        eps_ws_complex = eps_a + eps_b + eps_c + torch.complex(eps_ds_r, -eps_ds_i)
        
        is_wet = liq_snow > 0.0
        eps_ws_r = torch.where(is_wet, eps_ws_complex.real, eps_ds_r)
        eps_ws_i = torch.where(is_wet, -eps_ws_complex.imag, eps_ds_i) 
        eps_ws = torch.complex(eps_ws_r, -eps_ws_i)

        alpha = k * torch.abs(torch.sqrt(eps_ws).imag)
        beta = k * torch.sqrt(eps_ws).real
        pp = 2.0 * alpha * beta
        qq = beta**2 - alpha**2 - (k**2) * (torch.sin(theta)**2)
        
        inner_s = torch.clamp(torch.sqrt(pp**2 + qq**2) + qq, min=1e-12)
        theta_s = torch.atan(k * torch.sin(theta) / ((1.0 / math.sqrt(2.0)) * torch.sqrt(inner_s)))
        
        z_s = self.z0 / torch.sqrt(eps_ws)

        r_sa_h = torch.abs((z_s * torch.cos(theta) - self.z0 * torch.cos(theta_s)) / 
                           (z_s * torch.cos(theta) + self.z0 * torch.cos(theta_s)))**2
        r_sa_v = torch.abs((self.z0 * torch.cos(theta) - z_s * torch.cos(theta_s)) / 
                           (self.z0 * torch.cos(theta) + z_s * torch.cos(theta_s)))**2
        r_sa = torch.stack([r_sa_h, r_sa_v])

        d = torch.clamp(1000.0 * (1.6e-4 + 1.1e-13 * ((rho_snow * 1000.0)**4)), max=3.0)

        ke_ds = 0.0018 * (fghz**2.8) * (d**2) / 4.3429                          
        
        b_ds = torch.clamp((eps_ds_i / eps_ds_r)**2, min=1e-12)
        ka_ds = 2.0 * omega * torch.sqrt(self.mu0 * self.eps_0 * eps_ds_r) * \
                torch.sqrt(b_ds / (2.0 * (torch.sqrt(1.0 + b_ds) + 1.0)))
        ke_ds = torch.where(ke_ds < ka_ds, ka_ds, ke_ds)

        b_ws = torch.clamp((eps_ws_i / eps_ws_r)**2, min=1e-12)
        ka_ws = 2.0 * omega * torch.sqrt(self.mu0 * self.eps_0 * eps_ws_r) * \
                torch.sqrt(b_ws / (2.0 * (torch.sqrt(1.0 + b_ws) + 1.0)))

        ke = (ke_ds - ka_ds) + ka_ws
        ks = ke_ds - ka_ds
        q_param = 0.96
        
        wk_h = snowdp / torch.clamp(rho_snow, min=1e-3)
        
        exponent = (ke - q_param * ks) * (1.0 / torch.cos(theta_s)) * wk_h
        exponent = torch.clamp(exponent, max=80.0) 
        l2_apu = torch.exp(exponent)

        denom_ke_ks = torch.clamp(ke - q_param * ks, min=1e-12)

        tb_2 = (1.0 + r_sn / l2_apu) * (1.0 - r_sa) * t_snow * \
               (ka_ws / denom_ke_ks) * (1.0 - 1.0 / l2_apu) / \
               (1.0 - r_sn * r_sa / (l2_apu**2))

        tb_3 = ((1.0 - r_sn) * (1.0 - r_sa) * t) / \
               (l2_apu * (1.0 - r_sn * r_sa / (l2_apu**2)))

        tb_tos = tb_2 + tb_3
        r_snow = 1.0 - (tb_2 / t_snow + tb_3 / t)

        return r_snow, tb_tos
