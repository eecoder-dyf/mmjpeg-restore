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

class WindowAttention(nn.Module):
    """
    基于窗口的自注意力机制 (用于瓶颈层的全局交互)
    """
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Input: [B*num_windows, N, C]
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, nH, N, C/nH]

        q = q * self.scale
        attn = self.softmax(q @ k.transpose(-2, -1)) # [B_, nH, N, N]

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer Block with Window Attention"""
    def __init__(self, dim, window_size, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. 功能子网络 (Sub-Networks)
# ==========================================

class H_Predictor(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, kernel_size=3, 
                 base_dim=32, window_size=8, num_heads=4):
        super().__init__()
        self.window_size = window_size
        self.out_dim = out_channels * kernel_size**2
        
        # --- Encoder (CNN) ---
        # Level 1 (H, W)
        self.stem = nn.Conv2d(in_channels, base_dim, 3, 1, 1)
        self.enc1 = ConvBlock(base_dim, base_dim)
        
        # Down 1 -> Level 2 (H/2, W/2)
        self.down1 = nn.Conv2d(base_dim, base_dim*2, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(base_dim*2, base_dim*2)
        
        # Down 2 -> Level 3 (H/4, W/4)
        self.down2 = nn.Conv2d(base_dim*2, base_dim*4, 3, stride=2, padding=1)
        
        # --- Bottleneck (Transformer / Cross-Interaction) ---
        # 在 1/4 尺度上进行全局/大窗口交互
        self.bottleneck = nn.ModuleList([
            TransformerBlock(dim=base_dim*4, window_size=(window_size, window_size), num_heads=num_heads),
            TransformerBlock(dim=base_dim*4, window_size=(window_size, window_size), num_heads=num_heads)
        ])
        
        # --- Decoder (CNN) ---
        # Up 1 -> Level 2
        self.up1 = nn.Conv2d(base_dim*4, base_dim*2, 1) # Reduce channels
        self.dec1 = ConvBlock(base_dim*4, base_dim*2)   # Input = Cat(Up, Enc2)
        
        # Up 2 -> Level 1
        self.up2 = nn.Conv2d(base_dim*2, base_dim, 1)
        self.dec2 = ConvBlock(base_dim*2, base_dim)     # Input = Cat(Up, Enc1)
        
        # --- Output Head ---
        self.head = nn.Conv2d(base_dim, self.out_dim * 2, 1)

    def forward_transformer(self, x):
        """Helper to handle window partitioning for transformer"""
        B, C, H, W = x.shape
        # Pad to be divisible by window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        H_pad, W_pad = x.shape[2], x.shape[3]

        # Partition: [B, C, H, W] -> [B*num_windows, Wh*Ww, C]
        x_win = x.permute(0, 2, 3, 1).view(B, H_pad // self.window_size, self.window_size, W_pad // self.window_size, self.window_size, C)
        x_win = x_win.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        
        # Apply Transformer Blocks
        for blk in self.bottleneck:
            x_win = blk(x_win)
            
        # Reverse: [B*num_windows, Wh*Ww, C] -> [B, C, H, W]
        x = x_win.view(B, H_pad // self.window_size, W_pad // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H_pad, W_pad)
        
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        return x

    def forward(self, u, v):
        # 1. Early Fusion
        x = torch.cat([u, v], dim=1) # [B, 6, H, W]
        
        # 2. Encoder Path
        f1 = self.enc1(self.stem(x))        # Level 1
        f2 = self.enc2(self.down1(f1))      # Level 2
        f3 = self.down2(f2)                 # Level 3 (Bottleneck Input)
        
        # 3. Bottleneck (Transformer Interaction)
        # 这里进行 Cross-Modal Context Modeling
        feat_bot = self.forward_transformer(f3)
        
        # 4. Decoder Path
        # Up 1
        up1 = F.interpolate(feat_bot, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        up1 = self.up1(up1)
        cat1 = torch.cat([up1, f2], dim=1)
        dec1 = self.dec1(cat1)
        
        # Up 2
        up2 = F.interpolate(dec1, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        up2 = self.up2(up2)
        cat2 = torch.cat([up2, f1], dim=1)
        dec2 = self.dec2(cat2)
        
        # 5. Prediction
        kernels = self.head(dec2)
        w_fwd, w_bwd = torch.chunk(kernels, 2, dim=1)
        
        return w_fwd, w_bwd

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

class CrossGradientPrior(nn.Module):
    """
    Gray-domain cross-gradient prior.

    The hyper-Laplacian prior is applied to:
        C_g(U, V) = | grad(G_u(U)) x grad(G_v(V)) |

    where G_u and G_v are fixed gray/luminance projections.

    Output:
        Fu_C: [B, C, H, W], gradient wrt original U
        Fv_C: [B, C, H, W], gradient wrt original V
        C_map: [B, 1, H, W], gray-domain cross-gradient magnitude
    """
    def __init__(
        self,
        channels=3,
        q_max=10.0,
        u_gray_weights=(0.299, 0.587, 0.114),
        v_gray_weights=None,
    ):
        super().__init__()
        self.channels = channels
        self.q_max = q_max

        # U is visible RGB by default
        if channels == 3:
            u_w = torch.tensor(u_gray_weights, dtype=torch.float32).view(1, 3, 1, 1)
        elif channels == 1:
            u_w = torch.ones(1, 1, 1, 1, dtype=torch.float32)
        else:
            # fallback: average all channels
            u_w = torch.ones(1, channels, 1, 1, dtype=torch.float32) / channels

        # V is NIR. If V is stored as 3 channels, use average by default.
        # If your NIR is duplicated into 3 channels, average and luminance are almost equivalent.
        if v_gray_weights is None:
            if channels == 3:
                v_w = torch.ones(1, 3, 1, 1, dtype=torch.float32) / 3.0
            elif channels == 1:
                v_w = torch.ones(1, 1, 1, 1, dtype=torch.float32)
            else:
                v_w = torch.ones(1, channels, 1, 1, dtype=torch.float32) / channels
        else:
            v_w = torch.tensor(v_gray_weights, dtype=torch.float32).view(1, channels, 1, 1)

        self.register_buffer("u_gray_weight", u_w)
        self.register_buffer("v_gray_weight", v_w)

    @staticmethod
    def diff_x(x):
        out = torch.zeros_like(x)
        out[:, :, :, :-1] = x[:, :, :, 1:] - x[:, :, :, :-1]
        return out

    @staticmethod
    def diff_y(x):
        out = torch.zeros_like(x)
        out[:, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
        return out

    @staticmethod
    def diff_x_t(x):
        out = torch.zeros_like(x)
        g = x[:, :, :, :-1]
        out[:, :, :, :-1] -= g
        out[:, :, :, 1:] += g
        return out

    @staticmethod
    def diff_y_t(x):
        out = torch.zeros_like(x)
        g = x[:, :, :-1, :]
        out[:, :, :-1, :] -= g
        out[:, :, 1:, :] += g
        return out

    def to_gray_u(self, U):
        # U: [B, C, H, W] -> [B, 1, H, W]
        return (U * self.u_gray_weight).sum(dim=1, keepdim=True)

    def to_gray_v(self, V):
        # V: [B, C, H, W] -> [B, 1, H, W]
        return (V * self.v_gray_weight).sum(dim=1, keepdim=True)

    def gray_grad_to_u_channels(self, grad_gray):
        # chain rule: dPhi/dU_c = w_c * dPhi/dG_u(U)
        return grad_gray * self.u_gray_weight

    def gray_grad_to_v_channels(self, grad_gray):
        # chain rule: dPhi/dV_c = w_c * dPhi/dG_v(V)
        return grad_gray * self.v_gray_weight

    def forward(self, U, V, p=0.8, eps=1e-3):
        # 1. Project to gray/luminance domain
        U_g = self.to_gray_u(U)
        V_g = self.to_gray_v(V)

        # 2. Compute gray-domain gradients
        U_x = self.diff_x(U_g)
        U_y = self.diff_y(U_g)
        V_x = self.diff_x(V_g)
        V_y = self.diff_y(V_g)

        # 3. Cross-gradient determinant
        L = U_x * V_y - U_y * V_x
        C_map = torch.abs(L)   # [B, 1, H, W]

        # 4. Hyper-Laplacian generalized derivative
        q = p * torch.pow(C_map + eps, p - 1.0) * torch.sign(L)
        q = torch.clamp(q, min=-self.q_max, max=self.q_max)

        # 5. Gradient wrt gray U and gray V
        Fu_gray = self.diff_x_t(q * V_y) - self.diff_y_t(q * V_x)
        Fv_gray = self.diff_y_t(q * U_x) - self.diff_x_t(q * U_y)

        # 6. Chain rule: gray-domain gradient -> original channel-domain gradient
        Fu_C = self.gray_grad_to_u_channels(Fu_gray)
        Fv_C = self.gray_grad_to_v_channels(Fv_gray)

        return Fu_C, Fv_C, C_map

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
                prior_net, rho, lam, mode='U', cross_grad_term=None, mu=0.0):
        
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
        if cross_grad_term is not None:
            F_val = F_val + mu * cross_grad_term
        
        return F_val, Z_est

# ==========================================
# 4. 主网络架构 (DB-ADMM-Net RGB)
# ==========================================

class Solver_V(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=16):
        super().__init__()
        
        # 输入: V_prev (in_channels) + F_v (in_channels)
        self.proj_in = nn.Conv2d(in_channels * 2, hidden_dim, kernel_size=1)
        
        # 大核深度可分离卷积 + 归一化 (核心救命稻草！)
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim),
            # 使用 GroupNorm(1, ...) 等价于视觉任务中最稳的 LayerNorm
            nn.GroupNorm(1, hidden_dim), 
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )
        
        self.proj_out = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        
        nn.init.normal_(self.proj_out.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.proj_out.bias, 0.0)

        self.alpha = nn.Parameter(torch.tensor([0.2]))  # 可学习的融合权重

    def forward(self, V_prev, F_v, lambda_val, rho_val):
        """
        显式传入变分参数 lambda_val 和 rho_val。
        解析步对应原二次部分，网络残差补偿非光滑 C 项与动态耦合误差。
        """
        # ==========================================
        # Step 1: 严格保留数学解析逆计算
        # ==========================================
        analytical_step = 1.0 / (1.0 + lambda_val + rho_val)
        delta_V_math = - analytical_step * F_v
        
        # ==========================================
        # Step 2: 轻量级残差提纯 (精准制导)
        x = torch.cat([V_prev, F_v], dim=1)
        
        feat = self.proj_in(x)
        feat = self.spatial_refine(feat)
        residual_correction = self.proj_out(feat)
        
        # ==========================================
        # Step 3: 输出最终的有效更新量 delta_V
        # ==========================================
        # 有效更新量 = 纯数学更新量 + 网络的非线性抗噪补偿
        delta_V = self.alpha * delta_V_math +  residual_correction

        return delta_V
    
class DB_ADMM_Net_RGB(nn.Module):
    def __init__(self, num_stages=4, channels=3, mu_c=0.01, p_c=0.8, eps_c=1e-3, q_max_c=10.0):
        super().__init__()
        self.num_stages = num_stages
        self.channels = channels
        self.p_c = p_c
        self.eps_c = eps_c
        
        # 参数
        self.rho = nn.Parameter(self._inverse_softplus_tensor(0.1))
        self.lam = nn.Parameter(self._inverse_softplus_tensor(0.5))
        self.mu_c = nn.Parameter(self._inverse_softplus_tensor(mu_c))
        
        # 动态卷积算子 (RGB版)
        self.cross_grad_prior = CrossGradientPrior(channels=channels, q_max=q_max_c)
        
        # 模块列表
        self.prior_u = nn.ModuleList([PriorNet(channels) for _ in range(num_stages)])
        self.prior_v = nn.ModuleList([PriorNet(channels) for _ in range(num_stages)])
        
        # H_Predictor 输入6通道(3+3)，输出3通道动态核
        self.h_pred = nn.ModuleList([H_Predictor(in_channels=channels*2, out_channels=channels) for _ in range(num_stages)])
        
        self.grad_calc = nn.ModuleList([Gradient_Calculator(channels) for _ in range(num_stages)])
        
        # SolverNet 输入6通道(3+3)
        self.solver_u = nn.ModuleList([SolverNet(in_channels=channels*2, out_channels=channels) for _ in range(num_stages)])
        self.solver_v = nn.ModuleList([Solver_V(in_channels=channels) for _ in range(num_stages)]) 

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
        mu_c = F.softplus(self.mu_c)
        
        for k in range(self.num_stages):
            Fu_C, Fv_C, C_map = self.cross_grad_prior(U, V, p=self.p_c, eps=self.eps_c)

            # --- Step U ---
            F_u_val, Z_u = self.grad_calc[k](
                U, V, J_u, Y_u, 
                self.prior_u[k], 
                rho, lam, mode='U', cross_grad_term=Fu_C, mu=mu_c
            )
            
            # Solver: Concat [U(3), Fu(3)]
            solver_in_u = torch.cat([U, F_u_val], dim=1)
            delta_U = self.solver_u[k](solver_in_u)
            U = U + delta_U
            
            # --- Step V ---
            F_v_val, Z_v = self.grad_calc[k](
                V, U, J_v, Y_v, 
                self.prior_v[k], 
                rho, lam, mode='V', cross_grad_term=Fv_C, mu=mu_c
            )
            
            delta_V = self.solver_v[k](V, F_v_val, lam, rho)
            V = V + delta_V
            
            # --- Step Y ---
            Y_u = Y_u + rho * (U - Z_u)
            Y_v = Y_v + rho * (V - Z_v)
            
            outputs.append((U, V, F_u_val, F_v_val, C_map))

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
    U_final, V_final, _, _, C_map = outputs[-1]
    
    print(f"Device: {device}")
    print(f"Input Shape: {Ju.shape}")      # [2, 3, 64, 64]
    print(f"Output U Shape: {U_final.shape}") # [2, 3, 64, 64]
    print(f"Output V Shape: {V_final.shape}") # [2, 3, 64, 64]
    print(f"C Map Shape: {C_map.shape}")
    
    # 验证梯度计算
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    
    from torchprofile import profile_macs
    inputs = (Ju, Jv)
    macs = profile_macs(model, args=inputs)
    print(f"Model MACs: {macs / 1e9:.4f} G")
