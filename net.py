import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 基础组件 (Basic Components)
# ==========================================

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
    def forward(self, x):
        return x + self.conv(x)
# ==========================================
# 2. 功能子网络 (Sub-Networks)
# ==========================================

class SoftThresholding(nn.Module):
    """
    可学习的软阈值收缩算子 (Proximal Operator for L1 norm)
    """
    def __init__(self, channels):
        super().__init__()
        # 初始化阈值参数，每个通道一个独立的阈值，必须大于0
        self.theta = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.05)

    def forward(self, x):
        # theta 限制在正数范围内
        theta = F.relu(self.theta)
        # Soft-thresholding 公式: sign(x) * max(|x| - theta, 0)
        return torch.sign(x) * F.relu(torch.abs(x) - theta)

class LCSC_Stage(nn.Module):
    """
    ISTA 算法的一个展开层 (Inner Unfolding Stage)
    """
    def __init__(self, in_channels=3, num_features=16):
        super().__init__()
        # 字典矩阵 (解码器)：将稀疏特征重建为图像
        self.W_d = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=False)
        # 编码矩阵：将图像残差提取为特征
        self.W_e = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=False)
        # 软阈值激活函数
        self.soft_thr = SoftThresholding(num_features)

    def forward(self, M_t, X):
        """
        M_t: 上一步的稀疏特征图 [B, num_features, H, W]
        X: 观测输入 U_tilde [B, C, H, W]
        """
        # 1. 重建图像并计算数据残差: (U_tilde - W_d * M_t)
        # 如果是第一步 (M_t is None)，残差直接是输入 X
        if M_t is None:
            residual = X
            M_t = 0
        else:
            X_recon = self.W_d(M_t)
            residual = X - X_recon
            
        # 2. 将残差映射到特征空间并加上过去的特征: M_t + W_e * residual
        feature_update = M_t + self.W_e(residual)
        
        # 3. 软阈值截断，产生极其稀疏且干净的 M_{t+1} (自动切除JPEG马赛克响应)
        M_next = self.soft_thr(feature_update)
        
        return M_next

class PriorNet(nn.Module):
    """
    基于学习卷积稀疏编码的双盲先验网络 (替换原有的 U-Net 黑盒)
    """
    def __init__(self, in_channels=3, num_features=32, num_unrolling=6):
        super().__init__()
        self.num_unrolling = num_unrolling
        
        # 构建 T 次展开的 LCSC 层
        # 为了减少参数，这里使用了权重非共享的设计；如果是权重共享，只需实例化一个 LCSC_Stage
        self.stages = nn.ModuleList([
            LCSC_Stage(in_channels, num_features) for _ in range(num_unrolling)
        ])
        
        # 最终输出的重建字典 (使用最后一次展开层的解码器)
        self.final_reconstruction = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, u_tilde):
        """
        u_tilde: 包含了物理更新但带有伪影的输入 (U + Lambda/rho)
        """
        M_t = None
        
        # 内层循环：ISTA 算法深度展开
        for i in range(self.num_unrolling):
            M_t = self.stages[i](M_t, u_tilde)
            
        # 最终重建出极其干净的去噪图像 Z
        Z_opt = self.final_reconstruction(M_t)
        
        # 传统做法会加上一个全局残差连接，保证整体能量不偏移
        return u_tilde + Z_opt
    
class SolverNet(nn.Module):
    """
    多尺度 SolverNet (U-Net 结构, 无额外辅助类版本)
    输入: [B, 6, H, W] (U, Fu)
    输出: [B, 3, H, W] (Delta U)
    """
    def __init__(self, in_channels=6, out_channels=3, base_dim=32):
        super().__init__()
        
        # ================== Encoder (编码器) ==================
        
        # Level 1: 原始分辨率
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            ResBlock(base_dim)
        )
        # Down 1: Level 1 -> Level 2 (使用 Stride=2 卷积下采样)
        self.down1 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim*2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Level 2: 1/2 分辨率
        self.enc2 = ResBlock(base_dim*2)
        
        # Down 2: Level 2 -> Level 3
        self.down2 = nn.Sequential(
            nn.Conv2d(base_dim*2, base_dim*4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # ================== Bottleneck (瓶颈层) ==================
        # Level 3: 1/4 分辨率 (处理全局信息)
        self.bottleneck = nn.Sequential(
            ResBlock(base_dim*4),
            ResBlock(base_dim*4),
            ResBlock(base_dim*4)
        )
        
        # ================== Decoder (解码器) ==================
        
        # Up 1 对应的卷积: Level 3 -> Level 2
        # 注意：先插值，再过这个卷积将通道数减半
        self.up1_conv = nn.Conv2d(base_dim*4, base_dim*2, 3, 1, 1)
        
        # Dec 1: 融合 Skip Connection 后的处理
        # 输入通道是 base_dim*2 (来自Up) + base_dim*2 (来自Enc2) = base_dim*4
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_dim*4, base_dim*2, 3, 1, 1), 
            nn.ReLU(inplace=True),
            ResBlock(base_dim*2)
        )
        
        # Up 2 对应的卷积: Level 2 -> Level 1
        self.up2_conv = nn.Conv2d(base_dim*2, base_dim, 3, 1, 1)
        
        # Dec 2: 融合 Skip Connection 后的处理
        # 输入通道是 base_dim (来自Up) + base_dim (来自Enc1) = base_dim*2
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_dim*2, base_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            ResBlock(base_dim)
        )
        
        # ================== Output Head (输出头) ==================
        self.tail = nn.Conv2d(base_dim, out_channels, 3, 1, 1)
        
        # 零初始化最后一层，保证初始输出 Delta U 接近 0
        nn.init.constant_(self.tail.weight, 0)
        nn.init.constant_(self.tail.bias, 0)

    def forward(self, x):
        # x: [B, 6, H, W]
        
        # --- Encoding ---
        # L1
        f1 = self.enc1(x)       # [B, 32, H, W]
        
        # Down -> L2
        f2 = self.down1(f1)     # [B, 64, H/2, W/2]
        f2 = self.enc2(f2)
        
        # Down -> L3
        f3 = self.down2(f2)     # [B, 128, H/4, W/4]
        
        # --- Bottleneck ---
        feat = self.bottleneck(f3) # [B, 128, H/4, W/4]
        
        # --- Decoding ---
        
        # 1. Up-sample (L3 -> L2)
        # 直接使用 functional API 进行双线性插值
        up_feat1 = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        up_feat1 = self.up1_conv(up_feat1) # [B, 64, H/2, W/2]
        
        # 2. Concat Skip Connection (f2)
        cat_feat1 = torch.cat([up_feat1, f2], dim=1) # [B, 128, H/2, W/2]
        dec_feat1 = self.dec1(cat_feat1)
        
        # 3. Up-sample (L2 -> L1)
        up_feat2 = F.interpolate(dec_feat1, scale_factor=2, mode='bilinear', align_corners=False)
        up_feat2 = self.up2_conv(up_feat2) # [B, 32, H, W]
        
        # 4. Concat Skip Connection (f1)
        cat_feat2 = torch.cat([up_feat2, f1], dim=1) # [B, 64, H, W]
        dec_feat2 = self.dec2(cat_feat2)
        
        # Output
        out = self.tail(dec_feat2) # [B, 3, H, W]
        
        return out

# ==========================================
# 3. 核心计算模块 (Function Block)
# ==========================================

class Gradient_Calculator(nn.Module):
    """
    计算 F(U) (3通道版本)
    """
    def __init__(self, channels=3, hidden_dim=16):
        super().__init__()
        self.h_fun_fwd = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1, bias=False, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect')
        )

        self.h_fun_bwd = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1, bias=False, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect')
        )

    def forward(self, curr_img, other_img, J_obs, Y_dual, 
                prior_net, rho, lam, mode='U'):
        
        # 1. 嵌入式先验梯度 (Z-Step)
        img_noisy = curr_img + Y_dual / rho
        Z_est = prior_net(img_noisy) 
        grad_prior = Y_dual + rho * (curr_img - Z_est)

        # 2. 数据保真梯度
        grad_data = curr_img - J_obs

        # 3. 耦合梯度 (Symmetric Dynamic Stream)
        if mode == 'U':
            # E = V - H(U)
            H_U = self.h_fun_fwd(curr_img)
            E_couple = other_img - H_U
            
            # 【核心改动】使用动态卷积模拟 H^T
            # 输入是误差 E，权重是专门预测出来的 w_bwd
            term_couple = self.h_fun_bwd(E_couple)
            
            grad_couple = -lam * term_couple
            
        else: # mode == 'V'
            # V 的梯度依然简单
            H_V = self.h_fun_fwd(other_img)
            E_couple = curr_img - H_V
            grad_couple = lam * E_couple

        # 4. 总梯度
        F_val = grad_data + grad_couple + grad_prior
        
        return F_val, Z_est

# ==========================================
# 4. 主网络架构 (DB-ADMM-Net RGB)
# ==========================================

class Solver_V(nn.Module):
    """
    Lightweight encoder-decoder for V-branch residual correction.

    There is no encoder-to-decoder skip connection. The analytical update
    remains the main path, while the network predicts a nonlinear correction
    from V and its total gradient.
    """
    def __init__(self, in_channels=3, base_dim=16):
        super().__init__()

        # Input: concat(V_prev, F_v), two in_channels tensors.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, base_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(base_dim),
        )

        self.down = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = ResBlock(base_dim * 2)

        self.up_conv = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = ResBlock(base_dim)
        self.proj_out = nn.Conv2d(base_dim, in_channels, kernel_size=3, padding=1)

        self.alpha = nn.Parameter(torch.tensor([0.2]))  # 可学习的融合权重

        # Start training from the analytical update and learn the correction
        # progressively, following Solver_U's stable output initialization.
        nn.init.constant_(self.proj_out.weight, 0.0)
        nn.init.constant_(self.proj_out.bias, 0.0)

    def forward(self, V_prev, F_v, lambda_val, rho_val):
        """
        显式传入变分参数 lambda_val 和 rho_val。
        解析步对应原二次部分，网络残差补偿动态耦合误差。
        """
        # ==========================================
        # Step 1: 严格保留数学解析逆计算
        # ==========================================
        analytical_step = 1.0 / (1.0 + lambda_val + rho_val)
        delta_V_math = - analytical_step * F_v
        
        # ==========================================
        # Step 2: 无 Encoder-Decoder skip connection 的轻量级残差修正
        x = torch.cat([V_prev, F_v], dim=1)

        feat = self.encoder(x)
        feat = self.down(feat)
        feat = self.bottleneck(feat)
        feat = F.interpolate(
            feat,
            size=V_prev.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        feat = self.up_conv(feat)
        feat = self.decoder(feat)
        residual_correction = self.proj_out(feat)
        
        # ==========================================
        # Step 3: 输出最终的有效更新量 delta_V
        # ==========================================
        # 有效更新量 = 纯数学更新量 + 网络的非线性抗噪补偿
        delta_V = self.alpha * delta_V_math +  residual_correction

        return delta_V
    
class DB_ADMM_Net_RGB(nn.Module):
    def __init__(self, num_stages=4, channels=3):
        super().__init__()
        self.num_stages = num_stages
        self.channels = channels
        
        # 参数
        self.rho = nn.Parameter(self._inverse_softplus_tensor(0.1))
        self.lam = nn.Parameter(self._inverse_softplus_tensor(0.5))
        
        # 模块列表
        self.prior_u = nn.ModuleList([PriorNet(channels) for _ in range(num_stages)])
        self.prior_v = nn.ModuleList([PriorNet(channels) for _ in range(num_stages)])
        
        self.grad_calc = nn.ModuleList([Gradient_Calculator(channels) for _ in range(num_stages)])
        
        # Solver_U 与 Solver_V 的输入均为6通道(3+3)
        self.solver_u = nn.ModuleList([SolverNet(in_channels=channels*2, out_channels=channels) for _ in range(num_stages)])
        self.solver_v = nn.ModuleList([Solver_V(in_channels=channels, base_dim=16) for _ in range(num_stages)])

    @staticmethod
    def _inverse_softplus_tensor(value):
        value = torch.tensor([value], dtype=torch.float32)
        return torch.log(torch.expm1(value))

    def forward(self, J_u, J_v):
        # [B, 3, H, W]
        U, V = J_u.clone(), J_v.clone()
        Y_u = torch.zeros_like(U)
        Y_v = torch.zeros_like(V)
        
        outputs = []
        rho = F.softplus(self.rho)
        lam = F.softplus(self.lam)
        
        for k in range(self.num_stages):
            # --- Step U ---
            F_u_val, Z_u = self.grad_calc[k](
                U, V, J_u, Y_u, 
                self.prior_u[k], 
                rho, lam, mode='U'
            )
            
            # Solver: Concat [U(3), Fu(3)]
            solver_in_u = torch.cat([U, F_u_val], dim=1)
            delta_U = self.solver_u[k](solver_in_u)
            U = U + delta_U
            
            # --- Step V ---
            F_v_val, Z_v = self.grad_calc[k](
                V, U, J_v, Y_v, 
                self.prior_v[k], 
                rho, lam, mode='V'
            )
            
            delta_V = self.solver_v[k](V, F_v_val, lam, rho)
            V = V + delta_V
            
            # --- Step Y ---
            Y_u = Y_u + rho * (U - Z_u)
            Y_v = Y_v + rho * (V - Z_v)
            
            outputs.append((U, V, F_u_val, F_v_val))

        return outputs

# ==========================================
# 5. 测试 Demo (RGB)
# ==========================================
if __name__ == "__main__":
    # 配置: Batch=1, Channels=3 (RGB), Size=256x256
    B, C, H, W = 1, 3, 256, 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    Ju = torch.randn(B, C, H, W).to(device)
    Jv = torch.randn(B, C, H, W).to(device)
    
    # 实例化 RGB 模型
    model = DB_ADMM_Net_RGB(num_stages=4, channels=3).to(device)
    
    # 运行
    outputs = model(Ju, Jv)
    U_final, V_final, _, _ = outputs[-1]
    
    print(f"Device: {device}")
    print(f"Input Shape: {Ju.shape}")      # [2, 3, 64, 64]
    print(f"Output U Shape: {U_final.shape}") # [2, 3, 64, 64]
    print(f"Output V Shape: {V_final.shape}") # [2, 3, 64, 64]
    
    # 验证梯度计算
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    
    from torchprofile import profile_macs
    inputs = (Ju, Jv)
    macs = profile_macs(model, args=inputs)
    print(f"Model MACs: {macs / 1e9:.4f} G")