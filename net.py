import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 基础组件 (Basic Components)
# ==========================================

class DynamicConv2d_RGB(nn.Module):
    """
    [升级版] 支持多通道的像素级动态卷积
    采用 Depthwise 策略：每个通道拥有独立的动态核
    """
    def __init__(self, channels=3, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.ks = kernel_size
        self.pad = kernel_size // 2

    def forward(self, x, weights):
        """
        x: [B, 3, H, W]
        weights: [B, 3 * K*K, H, W] -> 每个通道独立的 K*K 核
        """
        B, C, H, W = x.shape
        
        # 1. Unfold Input -> [B, C*K*K, HW]
        # x_unfold 包含了每个像素周围的 K*K 邻域
        x_unfold = F.unfold(x, self.ks, padding=self.pad)
        
        # 2. Reshape Input & Weights for Depthwise Operation
        # x_unfold: [B, C, K*K, HW]
        x_unfold = x_unfold.view(B, C, self.ks**2, H * W)
        
        # weights: [B, C, K*K, HW]
        w_reshape = weights.view(B, C, self.ks**2, H * W)
        
        # 3. Apply Weights (Pixel-wise Dot Product & Sum over kernel window)
        # 对应位置相乘，并在 K*K 维度上求和 -> 卷积操作
        out = (x_unfold * w_reshape).sum(dim=2) 
        
        # 4. Reshape back to Image
        out = out.view(B, C, H, W)
        return out

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

class H_Predictor(nn.Module):
    """
    根据 RGB 图像 U, V 预测动态卷积核
    输入: 6通道 (3+3)
    输出: 3 * K^2 通道 (每个颜色通道独立的核)
    """
    def __init__(self, in_channels=6, out_channels=3, kernel_size=3):
        super().__init__()
        mid = 64
        self.net_body = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        # 【改动】输出通道翻倍：一组给 H (Forward)，一组给 H^T (Backward)
        self.head = nn.Conv2d(mid, out_channels * kernel_size**2 * 2, 1)

    def forward(self, u, v):
        feat = self.net_body(torch.cat([u, v], dim=1))
        weights = self.head(feat)
        
        # 分割成两组权重
        w_fwd, w_bwd = torch.chunk(weights, 2, dim=1)
        return w_fwd, w_bwd

class PriorNet(nn.Module):
    """
    Z-Step: RGB 图像去噪/去伪影
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64), # RGB 信息更丰富，加深一层
            nn.Conv2d(64, in_channels, 3, 1, 1)
        )
    def forward(self, x):
        return x + self.net(x)

class Approx_Trans_Block(nn.Module):
    """
    模拟 H^T (反投影): 处理 RGB 误差
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, in_channels, 3, 1, 1)
        )
        # 初始化归零
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, x):
        return self.net(x)

class SolverNet(nn.Module):
    """
    牛顿求解器
    Input Channels = 9 (3 for Image + 3 for Grad + 3 for Hint)
    """
    def __init__(self, in_channels=9, out_channels=3):
        super().__init__()
        self.head = nn.Conv2d(in_channels, 64, 3, 1, 1)
        
        # 使用空洞卷积扩大感受野
        self.body = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            ResBlock(64),
            ResBlock(64)
        )
        self.tail = nn.Conv2d(64, out_channels, 3, 1, 1)

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat) + feat
        return self.tail(feat)

# ==========================================
# 3. 核心计算模块 (Function Block)
# ==========================================

class Gradient_Calculator(nn.Module):
    """
    计算 F(U) (3通道版本)
    """
    def __init__(self, channels=3):
        super().__init__()
        # self.approx_HT = Approx_Trans_Block(in_channels=channels) # 移除
        self.dyn_conv = DynamicConv2d_RGB(channels=channels)

    def forward(self, curr_img, other_img, J_obs, Y_dual, 
                h_func_fwd, w_bwd, # 输入变为: 正向函数 + 反向权重
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
            H_U = h_func_fwd(curr_img)
            E_couple = other_img - H_U
            
            # 【核心改动】使用动态卷积模拟 H^T
            # 输入是误差 E，权重是专门预测出来的 w_bwd
            term_couple = self.dyn_conv(E_couple, w_bwd)
            
            grad_couple = -lam * term_couple
            
        else: # mode == 'V'
            # V 的梯度依然简单
            H_U = h_func_fwd(other_img)
            E_couple = curr_img - H_U
            grad_couple = lam * E_couple

        # 4. 总梯度
        F_val = grad_data + grad_couple + grad_prior
        
        return F_val, Z_est

# ==========================================
# 4. 主网络架构 (DB-ADMM-Net RGB)
# ==========================================

class DB_ADMM_Net_RGB(nn.Module):
    def __init__(self, num_stages=4, channels=3):
        super().__init__()
        self.num_stages = num_stages
        self.channels = channels
        
        # 参数
        self.rho = nn.Parameter(torch.tensor([0.1]))
        self.lam = nn.Parameter(torch.tensor([0.5]))
        
        # 初始 H (3->3) 用于全局差分先验
        self.h_init = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        
        # 动态卷积算子 (RGB版)
        self.dyn_conv_op = DynamicConv2d_RGB(channels=channels, kernel_size=3)
        
        # 模块列表
        self.prior_u = nn.ModuleList([PriorNet(channels) for _ in range(num_stages)])
        self.prior_v = nn.ModuleList([PriorNet(channels) for _ in range(num_stages)])
        
        # H_Predictor 输入6通道(3+3)，输出3通道动态核
        self.h_pred = nn.ModuleList([H_Predictor(in_channels=channels*2, out_channels=channels) for _ in range(num_stages)])
        
        self.grad_calc_u = nn.ModuleList([Gradient_Calculator(channels) for _ in range(num_stages)])
        self.grad_calc_v = nn.ModuleList([Gradient_Calculator(channels) for _ in range(num_stages)])
        
        # SolverNet 输入9通道(3+3+3)
        self.solver_u = nn.ModuleList([SolverNet(in_channels=channels*3, out_channels=channels) for _ in range(num_stages)])
        self.solver_v = nn.ModuleList([SolverNet(in_channels=channels*3, out_channels=channels) for _ in range(num_stages)])

    def forward(self, J_u, J_v):
        # [B, 3, H, W]
        U, V = J_u.clone(), J_v.clone()
        Y_u = torch.zeros_like(U)
        Y_v = torch.zeros_like(V)
        
        # 差分先验 (RGB Difference Map)
        with torch.no_grad():
            init_proj = self.h_init(J_u)
            Map_hint = J_v - init_proj 

        outputs = []
        
        for k in range(self.num_stages):
            # 预测 H (RGB Dynamic Kernels for Forward and Backward)
            w_fwd, w_bwd = self.h_pred[k](U, V) # w_fwd是正向核，w_bwd是反向核
            h_func_fwd = lambda x: self.dyn_conv_op(x, w_fwd)
            
            # --- Step U ---
            F_u_val, Z_u = self.grad_calc_u[k](
                U, V, J_u, Y_u, 
                h_func_fwd, w_bwd, # Pass both forward func and backward weights
                self.prior_u[k], 
                self.rho, self.lam, mode='U'
            )
            
            # Solver: Concat [U(3), Fu(3), Hint(3)]
            solver_in_u = torch.cat([U, F_u_val, Map_hint], dim=1)
            delta_U = self.solver_u[k](solver_in_u)
            U = U + delta_U
            
            # --- Step V ---
            F_v_val, Z_v = self.grad_calc_v[k](
                V, U, J_v, Y_v, 
                h_func_fwd, w_bwd, # Pass both forward func and backward weights
                self.prior_v[k], 
                self.rho, self.lam, mode='V'
            )
            
            solver_in_v = torch.cat([V, F_v_val, Map_hint], dim=1)
            delta_V = self.solver_v[k](solver_in_v)
            V = V + delta_V
            
            # --- Step Y ---
            Y_u = Y_u + self.rho * (U - Z_u)
            Y_v = Y_v + self.rho * (V - Z_v)
            
            outputs.append((U, V, F_u_val, F_v_val))

        return outputs

# ==========================================
# 5. 测试 Demo (RGB)
# ==========================================
if __name__ == "__main__":
    # 配置: Batch=2, Channels=3 (RGB), Size=64x64
    B, C, H, W = 2, 3, 64, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    Ju = torch.randn(B, C, H, W).to(device)
    Jv = torch.randn(B, C, H, W).to(device)
    
    # 实例化 RGB 模型
    model = DB_ADMM_Net_RGB(num_stages=3, channels=3).to(device)
    
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