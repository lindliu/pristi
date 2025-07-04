import numpy as np
import torch
import torch.nn as nn
from diff_models import Guide_diff_pm25, Guide_diff_pm25_vae_error, Guide_diff_pm25_vae_latent_mean, Guide_diff_pm25_vae_latent_corr
from diff_models import Guide_diff_metrla, Guide_diff_metrla_vae_error, Guide_diff_metrla_vae_latent_mean, Guide_diff_metrla_vae_latent_corr
from diff_models import Guide_diff_pemsbay, Guide_diff_pemsbay_vae_error,Guide_diff_pemsbay_vae_latent_mean,Guide_diff_pemsbay_vae_latent_corr

class PriSTI(nn.Module):
    def __init__(self, target_dim, seq_len, config, device, name):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.seq_len = seq_len

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.use_guide = config["model"]["use_guide"]

        self.cde_output_channels = config["diffusion"]["channels"]
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        config_diff["device"] = device
        self.device = device
        self.name = name
        print(name)
        input_dim = 2
        if self.name == 'pm25':
            self.diffmodel = Guide_diff_pm25(config_diff, input_dim, target_dim, self.use_guide)
        if self.name == 'pm25_vae_error':
            self.diffmodel = Guide_diff_pm25_vae_error(config_diff, input_dim, target_dim, self.use_guide)
        if self.name == 'pm25_vae_latent_mean':
            self.diffmodel = Guide_diff_pm25_vae_latent_mean(config_diff, input_dim, target_dim, self.use_guide)
        if self.name == 'pm25_vae_latent_corr':
            self.diffmodel = Guide_diff_pm25_vae_latent_corr(config_diff, input_dim, target_dim, self.use_guide)
        
        
        if self.name == 'metrla':
            self.diffmodel = Guide_diff_metrla(config_diff, input_dim, target_dim, self.use_guide)
        if self.name == 'metrla_vae_error':
            self.diffmodel = Guide_diff_metrla_vae_error(config_diff, input_dim, target_dim, self.use_guide)
        if self.name == 'metrla_vae_latent_mean':
            self.diffmodel = Guide_diff_metrla_vae_latent_mean(config_diff, input_dim, target_dim, self.use_guide)
        if self.name == 'metrla_vae_latent_corr':
            self.diffmodel = Guide_diff_metrla_vae_latent_corr(config_diff, input_dim, target_dim, self.use_guide)
            
        
        if self.name == 'pemsbay':
            self.diffmodel = Guide_diff_pemsbay(config_diff, input_dim, target_dim, self.use_guide)
        if self.name == 'pemsbay_vae_error':
            self.diffmodel = Guide_diff_pemsbay_vae_error(config_diff, input_dim, target_dim, self.use_guide)
        if self.name == 'pemsbay_vae_latent_mean':
            self.diffmodel = Guide_diff_pemsbay_vae_latent_mean(config_diff, input_dim, target_dim, self.use_guide)
        if self.name == 'pemsbay_vae_latent_corr':
            self.diffmodel = Guide_diff_pemsbay_vae_latent_corr(config_diff, input_dim, target_dim, self.use_guide)




        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, itp_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, itp_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, itp_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        if not self.use_guide:
            itp_info = cond_mask * observed_data
        predicted = self.diffmodel(total_input, side_info, t, itp_info, cond_mask)

        # target_mask = observed_mask - cond_mask
        # # target_mask[:,[25, 32, 33, 34, 35],:] = 0
        # # target_mask[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32],:] = 0
        # residual = (noise - predicted) * target_mask
        # num_eval = target_mask.sum()
        # loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        
        
        if self.name=='pm25':
            
            ########## pms25 ############
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[0,  1, 22, 23, 24, 26, 27, 28, 29, 31, 25, 30, 32, 33, 34, 35],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss1 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[0,  1, 22, 23, 24, 26, 27, 28, 29, 31]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            #        19, 20, 21, 25, 30, 32, 33, 34, 35],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss2 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[25, 30, 32, 33, 34, 35]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                    # 19, 20, 21, 0,  1, 22, 23, 24, 26, 27, 28, 29, 31],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss3 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            loss = (loss1 + loss2 + loss3)/3
            ###############################
        
        
        
        if self.name=='pm25_vae_error':
            
            ########## pms25 ############
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[17, 21, 19, 16, 15, 14, 13, 12, 10, 11,  8,  7,  5,  4,  3,  9]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[0,  1, 22, 23, 24, 26, 27, 28, 29, 31, 25, 30, 32, 33, 34, 35],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss1 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[26, 30, 29, 28, 31, 27, 32, 25,  0, 23, 22,  6,  2,  1, 24]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            #        19, 20, 21, 25, 30, 32, 33, 34, 35],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss2 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[20, 18, 34, 33, 35]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                    # 19, 20, 21, 0,  1, 22, 23, 24, 26, 27, 28, 29, 31],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss3 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            loss = (loss1 + loss2 + loss3)/3
            ###############################
            
            
            
        if self.name=='pm25_vae_latent_mean':
            
            ########## pms25 ############
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[0, 22, 21, 20, 19, 15, 13, 12, 11, 14,  8,  7,  6,  5,  9,  3,  2]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[0,  1, 22, 23, 24, 26, 27, 28, 29, 31, 25, 30, 32, 33, 34, 35],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss1 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[32, 26, 25, 24, 23, 28, 29, 30, 31, 27,  1]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            #        19, 20, 21, 25, 30, 32, 33, 34, 35],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss2 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[33, 17, 34, 16, 10,  4, 18, 35]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                    # 19, 20, 21, 0,  1, 22, 23, 24, 26, 27, 28, 29, 31],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss3 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            loss = (loss1 + loss2 + loss3)/3
            ###############################
            
            
        
        if self.name=='pm25_vae_latent_corr':
            
            ########## pms25 ############
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[13, 15, 14, 21, 12, 11, 10,  9,  8,  7,  6,  4,  3, 20, 19]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[0,  1, 22, 23, 24, 26, 27, 28, 29, 31, 25, 30, 32, 33, 34, 35],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss1 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[22, 23, 27,  0, 29, 30, 25, 31, 32, 28, 26]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            #        19, 20, 21, 25, 30, 32, 33, 34, 35],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss2 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[33, 17, 18, 34, 16, 35]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                    # 19, 20, 21, 0,  1, 22, 23, 24, 26, 27, 28, 29, 31],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss3 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[5,  2,  1, 24]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                    # 19, 20, 21, 0,  1, 22, 23, 24, 26, 27, 28, 29, 31],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss4 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            loss = (loss1 + loss2 + loss3 + loss4)/4
            ###############################
        
        
        if self.name == 'metrla':
            ########## metrla ############
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[206,  91, 174,  87,  38, 170,  41, 165, 139, 163,  47,  48,  82,
                    81, 160,  52, 159,  56, 141, 155,  60,  76,  63, 150,  73, 148,
                   133,  95, 144, 191,  12, 127,  96,  15, 120, 193, 187, 100, 125,
                     4,  25, 204,  23]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[206,  91, 174,  87,  38, 170,  41, 165, 139, 163,  47,  48,  82,
            #         81, 160,  52, 159,  56, 141, 155,  60,  76,  63, 150,  73, 148,
            #        133,  95, 144, 191,  12, 127,  96,  15, 120, 193, 187, 100, 125,
            #          4,  25, 204,  23],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss1 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
           
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[75, 113, 111, 142,  74, 124, 140,  80, 119,
           115, 112,  94, 104, 205, 130, 138, 137, 101,  86, 131, 136,  72,
           134,  97,  93, 128,  89, 157,  69, 145,  26,  24, 186, 188,  19,
           190,  17, 181,  16, 195, 196, 197, 198,   6,   5,   3, 199, 192,
           180, 182,  33, 146, 149, 154,  61, 116,  57, 158,  54, 162, 161,
            43,  42, 166, 172, 175,  34, 164, 117]] = 1.
            target_mask = target_mask * mm
           #  target_mask[:,[75, 113, 111, 142,  74, 124, 140,  80, 119,
           # 115, 112,  94, 104, 205, 130, 138, 137, 101,  86, 131, 136,  72,
           # 134,  97,  93, 128,  89, 157,  69, 145,  26,  24, 186, 188,  19,
           # 190,  17, 181,  16, 195, 196, 197, 198,   6,   5,   3, 199, 192,
           # 180, 182,  33, 146, 149, 154,  61, 116,  57, 158,  54, 162, 161,
           #  43,  42, 166, 172, 175,  34, 164, 117],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss2 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[176, 122, 132, 185, 151,
            183, 103,  64,  84,   9,  88,   8,  14,  78,  98,  77, 102,   2,
             28,  59,  40,  70,  29,  68,  83,  79]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[176, 122, 132, 185, 151,
            # 183, 103,  64,  84,   9,  88,   8,  14,  78,  98,  77, 102,   2,
             # 28,  59,  40,  70,  29,  68,  83,  79],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss3 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[189, 153, 152, 114,  50,
             45,  44, 167,  31, 179, 178,  21, 156,  35, 105,  10,  90, 121,
             203,  99,  18, 106, 126, 200]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[189, 153, 152, 114,  50,
            #  45,  44, 167,  31, 179, 178,  21, 156,  35, 105,  10,  90, 121,
            #  203,  99,  18, 106, 126, 200],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss4 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[177,   1,  30, 202, 201,  13,  27,
              22,   7,  20, 194,  32, 184,  11,  49, 173, 110, 109, 123, 108,
             107, 129,  92, 135,  85, 143,  71, 147,  67,  66,  65,  62,  58,
              55,  53,  51, 118,  46, 168, 169,  39, 171,  37,  36,   0]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[177,   1,  30, 202, 201,  13,  27,
            #   22,   7,  20, 194,  32, 184,  11,  49, 173, 110, 109, 123, 108,
            #  107, 129,  92, 135,  85, 143,  71, 147,  67,  66,  65,  62,  58,
            #   55,  53,  51, 118,  46, 168, 169,  39, 171,  37,  36,   0],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss5 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        
            loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5
           #  #############################
            
            

        if self.name == 'metrla_vae_error':
            ########## metrla ############
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[103,  59, 135,  57, 136,  55,  54, 142, 145, 152, 153,  93,  44,
                   154, 156, 157, 123,  40, 122,  68,  89,  94,  98,  85,  84,  83,
                    99, 102,  80,  79,  78,  74, 110,  70, 111, 119, 161,  45,  19,
                   178,  20, 162, 182, 184,  10,   9, 201,   7,   6, 202, 203, 204,
                     2,   1, 177, 176,  91, 175,  27,  28, 173,  31, 169, 171, 167,
                   168,  24]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss1 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
           
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[107,  95, 166, 160, 196, 105, 106, 194, 109, 138, 139, 132, 129,
                   128, 125, 108, 140, 120, 181, 174, 143, 185, 187, 180, 163,   0,
                    46,  25,  26,  16,  66,  30,  56,  32,  51,  12,  77,  71,  11,
                     8,  36,  62]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss2 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[37, 127,  38,  22, 114,  39,  96]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss3 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[23,  42,  41,  29,  35, 172, 165, 170, 164,  33,  34, 158, 159,
                    90, 179,   3,   4,   5, 200, 199, 198, 197, 195, 193, 192, 191,
                   190, 189, 188,  13, 186,  14,  15, 183,  17, 155,  21,  18, 149,
                    47,  67, 118, 117, 116, 115, 113, 112,  69,  72,  73, 121,  75,
                   104, 205,  81, 101, 100,  82,  86,  97,  87,  88,  76,  65,  64,
                   124, 151, 150,  92, 148, 147, 146,  48, 144,  49,  50, 141,  52,
                    53, 137,  58, 134, 133,  60, 131, 130,  61, 126,  63,  43, 206]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss4 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        
            loss = (loss1 + loss2 + loss3 + loss4)/4
           #  #############################
           
           
        

        if self.name == 'metrla_vae_latent_mean':
            ########## metrla ############
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[67,  79,  78,  74, 161, 162,  68, 136,  66,  65,  64, 167, 168,
                   169, 171, 173, 176,  45,  80,  44,  85,  90, 132, 123, 122, 119,
                   118, 117, 116, 145, 110, 108, 107, 104, 205,  99,  97,  92, 153,
                    89, 177, 135,  40,  20, 188,  13,  24, 195, 198, 185,  28, 184,
                   183, 200, 201, 203,  31, 178, 182,  34,   2,  32,   1]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss1 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
           
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[3, 102, 101, 100,  98, 194, 150,  95, 133,  14, 131, 148, 139,
                   199, 109, 127, 140, 125, 124, 151, 112, 146,   5, 137,  18, 154,
                   164,  63,  41,  61,  60, 165,  57,  53, 170, 181, 172, 180,  47,
                   179,  39,  43, 163,  70, 206,  25,  84,  26, 156,  23, 158,  82,
                    76, 159,  73, 155,  77]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss2 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[192, 138, 175, 190, 152, 141, 186, 142, 204, 166, 202, 143, 147,
                   149, 157, 189,   0, 130,  58,  55,  54,  52, 134,  50,  49,  46,
                    42,  37,  36,  59,  35,  30,  27,  21,  19,  17,  11,  10,   9,
                     8,   7,   6,  33,  62, 103,  71, 129, 128, 126, 121, 120, 115,
                   114, 113,  69, 106, 105,  96,  94, 111,  88,  72,  83,  87,  86,
                    93]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss3 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[38,   4, 160,  75,  81, 197,  12,  15, 193, 191, 144, 187,  22,
                   174,  29,  48,  51]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss4 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        
        
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[196,  56,  91,  16]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss5 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5
           #  #############################
           


        if self.name == 'metrla_vae_latent_corr':
            ########## metrla ############
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[0, 108, 110, 111, 114, 115, 116, 107, 117, 119, 122, 123, 128,
                   129, 132, 118, 135, 106, 205,  79,  80,  83,  85,  86,  87, 104,
                    88,  90,  92,  93,  94,  97,  99,  89, 136, 138, 142, 183, 184,
                   185, 186, 188, 189, 182, 190, 198, 200, 201, 202, 203, 204, 195,
                   178, 177, 176, 145, 147, 149, 152, 153, 154, 157, 161, 162, 167,
                   168, 169, 171, 173, 175,  78,  74, 103,  49,  13,  55,  54,  19,
                    20,  46,  45,  44,  42,  24,  40,  39,  37,  36,  35,  27,  34,
                    28,  30,  33,  32,  11,  10,  31,   9,  66,  57,  65,  64,   7,
                    67,  68,  70,   6,  59,  71,   8,   2,   1,  58]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss1 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
           
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[151, 163,  25,  26, 155, 160, 159, 199,   5,   4, 156,   3, 158,
                    23, 165, 187,  12, 191,  14,  15, 181, 180, 179,  17, 150,  18,
                   192, 174, 193, 194, 172,  21, 170, 166, 164,  73, 139, 146, 148,
                   109,  51,  52,  53, 105, 102, 101, 100,  98,  50,  96,  60,  61,
                    62,  63,  84,  82,  81,  69,  77,  76,  95, 112, 206,  48, 143,
                   141, 140,  72, 137, 134, 133, 113, 131, 130,  38, 126, 125, 124,
                    41, 127, 121, 120,  43,  47]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss2 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[56,  75, 144,  29, 197,  22,  91, 196,  16]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss3 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            loss = (loss1 + loss2 + loss3)/3
           #  #############################
           
           
            
        if self.name=='pemsbay':
            ########## pemsbay ############
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[0,   4,   5,  14,  19,  21,  23,  25,  27,  28,  29,  30,  34,
                    35,  36,  40,  44,  45,  47,  52,  54,  57,  58,  59,  62,  64,
                    66,  67,  69,  70,  71,  77,  78,  86,  87,  88,  89,  90,  92,
                    93,  94,  98,  99, 100, 104, 105, 107, 110, 112, 118, 124, 125,
                   128, 129, 130, 131, 132, 137, 138, 139, 142, 143, 149, 150, 156,
                   157, 158, 159, 160, 161, 162, 165, 166, 168, 171, 172, 174, 181,
                   184, 186, 187, 188, 189, 190, 193, 194, 198, 202, 204, 211, 213,
                   214, 216, 218, 220, 222, 224, 228, 230, 237, 238, 239, 241, 243,
                   246, 247, 250, 253, 255, 257, 258, 259, 260, 261, 262, 263, 264,
                   268, 269, 270, 271, 272, 273, 280, 286, 287, 288, 289, 290, 291,
                   292, 293, 294, 295, 296, 297, 298, 310, 312, 314, 315, 316, 320,
                   321, 322, 323, 324]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[0,  1, 22, 23, 24, 26, 27, 28, 29, 31, 25, 30, 32, 33, 34, 35],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss1 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[2,   6,  13,  16,  20,  38,  39,  43,  53,  63,  68,  80, 111,
                   117, 140, 146, 148, 153, 154, 164, 170, 177, 180, 183, 185, 199,
                   206, 207, 209, 210, 212, 215, 242, 251, 265, 266, 267, 313, 319]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss2 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[1,   3,   7,   8,   9,  10,  11,  12,  15,  17,  18,  22,  24,
                    26,  31,  32,  33,  37,  41,  42,  46,  48,  49,  50,  51,  55,
                    56,  60,  61,  65,  72,  73,  74,  75,  76,  79,  81,  82,  83,
                    84,  85,  91,  95,  96,  97, 101, 102, 103, 106, 108, 109, 113,
                   114, 115, 116, 119, 120, 121, 122, 123, 126, 127, 133, 134, 135,
                   136, 141, 144, 145, 147, 151, 152, 155, 163, 167, 169, 173, 175,
                   176, 178, 179, 182, 191, 192, 195, 196, 197, 200, 201, 203, 205,
                   208, 217, 219, 221, 223, 225, 226, 227, 229, 231, 232, 233, 234,
                   235, 236, 240, 244, 245, 248, 249, 252, 254, 256, 274, 275, 276,
                   277, 278, 279, 281, 282, 283, 284, 285, 299, 300, 301, 302, 303,
                   304, 305, 306, 307, 308, 309, 311, 317, 318]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss3 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            loss = (loss1 + loss2 + loss3)/3
            ###############################
            
            
        if self.name=='pemsbay_vae_error':
            ########## pemsbay ############
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[0, 139, 142, 143, 149, 157, 159, 161, 171, 172, 183, 184, 186,
                   187, 188, 189, 137, 190, 132, 129,  87,  90,  92,  93,  94,  98,
                    99, 104, 105, 107, 110, 112, 118, 124, 128, 131,  78, 193, 214,
                   271, 272, 280, 286, 288, 289, 290, 294, 296, 297, 298, 312, 314,
                   315, 316, 270, 194, 269, 262, 216, 218, 220, 228, 230, 237, 238,
                   239, 241, 246, 253, 257, 258, 259, 260, 264,  77, 162, 324,  67,
                    66,  40,  28,  27,  44,  52,  29,  62,  45,  47,  59,  58,  57,
                    21,  54,   5,  36,  25,  69,  34,  35,  30,  14,  71,  70]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[0,  1, 22, 23, 24, 26, 27, 28, 29, 31, 25, 30, 32, 33, 34, 35],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss1 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[160, 212, 255, 158, 156, 213,  23, 202, 153, 152, 215, 150,  50,
                    39, 265, 267, 211, 154, 323, 165, 204, 185, 224,  26, 181, 242,
                   206, 247, 207, 209, 250, 222,  46, 251, 168,  24, 198, 163, 166,
                    19,  63,  64, 140,  95,  68,   4, 115,  89, 291,  10,  88, 310,
                    86, 121, 282,  13, 313, 100, 320,  53, 136, 273, 138, 322, 134,
                    82, 130,   2, 318, 321]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss2 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[236, 235, 311, 233,   3, 229, 223, 227, 226, 225, 317,   1, 232,
                    31,  15, 306,  17, 221,  20, 274, 276, 277, 278, 256, 279, 281,
                   285, 254,  11, 309,   9,   8,   7,   6, 249, 248, 299, 300, 244,
                   243, 301, 302, 240, 303, 252,  32, 200,  33, 114, 116,  60, 119,
                   120, 122, 123, 113, 126,  56,  55, 135, 219,  51, 144, 145, 127,
                   147,  61, 109,  73,  79,  80,  81,  83,  84,  85, 111,  72,  96,
                    97, 101, 102, 103, 106, 108,  91, 151, 141, 196, 179, 178,  43,
                    42,  41,  38, 191, 192, 176, 175,  37, 173, 195, 182, 180, 169,
                   167, 199,  75,  48, 201,  49, 203, 205, 208, 210, 155, 217, 197]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss3 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[234, 148, 292,  76, 146, 133, 174]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss4 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[74, 261, 263, 319,  18, 266, 268,  16, 164, 275, 231, 170, 293,
                   307, 305, 304, 125,  65, 177, 283, 284, 245,  12, 287, 295, 117,
                    22, 308]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss5 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5
            ###############################
            
            
            
            
        if self.name=='pemsbay_vae_latent_mean':
            ########## pemsbay ############
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[324,  86, 185, 186, 187, 188, 191,  77, 198,  75, 199, 206, 207,
                   209,  70, 211, 213, 137, 220, 222,  63, 234,  58,  56, 184,  89,
                   180, 178, 130, 129, 139, 141, 124, 123, 122, 118, 153, 155, 157,
                   239, 112, 110, 109, 108, 105, 103, 102, 166, 100, 168, 170,  95,
                   160, 240, 216,  52, 269, 270, 271,  25, 272, 280, 286, 287, 288,
                   291, 296, 311,  13,  12,  10, 314,   7, 316,   5, 317, 322, 265,
                   264, 136,  51,  34, 248, 246,  45,  36,  50, 255, 257,  46,  40]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[0,  1, 22, 23, 24, 26, 27, 28, 29, 31, 25, 30, 32, 33, 34, 35],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss1 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[190, 312, 156, 228, 154, 315, 150, 148, 143, 233, 245, 142, 237,
                   140, 319, 321, 230, 309, 164, 323, 189, 263, 193, 262, 266, 267,
                   181, 261, 260, 306, 179, 242, 175, 173, 172, 258, 212, 253, 165,
                   298, 259, 177,   0,  39,  92,  14,  33,  30, 107, 111,  27,  24,
                    44, 117,  62, 119,  78,  47,  65, 125,  76,  18,  22,  16]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss2 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[226, 247,  69,  49,  54, 215, 217, 238,  57,  11, 219,  59, 232,
                    61, 221, 223, 227, 224, 236, 313, 229, 252, 301, 300, 299, 297,
                   294,  15, 290, 289,  20, 284, 283, 282, 250, 281, 277, 273,  28,
                   268,  29,  31,  32,  37,  71,  41, 256, 254, 278,  73, 218, 204,
                    79, 120,   8,  84,  85,  87, 115, 192, 114,  93, 158,  94, 174,
                   171, 101,   4,  91, 147,   3, 195, 131, 202, 128, 127, 194, 144,
                   200,  99, 196, 197, 146]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss3 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[1, 167, 132,  21, 318, 106, 163,   2, 161, 285, 159, 145, 126,
                    19, 149, 275, 276, 151, 116, 279,  23,  35, 113, 274,  97,  98,
                    38,  81,  80, 225,  64,  74, 305, 201, 214,  67,  68, 210, 203,
                   208,  72,  83,  60, 231, 182,  42,  43, 205,  96, 251, 249, 176,
                   183, 304,  48, 244, 243,  53, 241,  90,   6]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss4 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[9, 308, 307, 310, 320, 134, 302, 135, 138, 121, 152, 104, 169,
                    88,  82,  66, 133, 235,  55,  26, 292, 293, 295,  17, 303, 162]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss5 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5
            ###############################
            
            
            
        if self.name=='pemsbay_vae_latent_corr':
            ########## pemsbay ############
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[324, 108, 199, 206, 100, 209,  98, 211, 212,  95, 213,  93,  91,
                    90, 198,  89, 216,  85, 220, 222,  81,  77, 236,  75, 147, 239,
                    71,  70, 242,  87, 245, 110, 197, 150, 144, 143, 153, 140, 139,
                   154, 137, 160, 164, 166, 131, 130, 111, 129, 127, 170, 123, 177,
                   181, 184, 119, 185, 117, 191, 194, 114, 113, 128,  65, 238, 148,
                   278, 280, 281,  39, 314, 287, 288,  10, 289,  11,  63,  34, 290,
                    13,  14,  30,  15,  28, 294, 296, 297,  24,  16,  22,  21, 312,
                     8,  18,  45,  62, 322, 258,  59,  58, 259, 261, 262, 263,  44,
                   317, 264,   5,  46,   7, 272, 271, 269, 270, 268, 266, 316]] = 1.
            target_mask = target_mask * mm
            # target_mask[:,[0,  1, 22, 23, 24, 26, 27, 28, 29, 31, 25, 30, 32, 33, 34, 35],:] = 0
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss1 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[318, 313, 161, 320, 151, 152, 167, 159, 251, 310, 249, 247, 252,
                   241, 225, 224, 223, 221, 215, 214, 273, 285, 208, 207, 205, 204,
                   291, 292, 293, 301, 308, 176, 202, 162,  42,  55,  41, 101,  88,
                    50, 104,  64,  83, 133,  66,  80,  53,  20,  72, 116,   1, 145,
                    38]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss2 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[201, 302, 200, 303, 300, 304,  61, 106, 277,  96,  37,  73, 231,
                   240, 229, 227, 299, 226, 244, 219,  68,  26, 305, 284,  82, 195,
                   192, 307,  12, 126,   9,  49, 163,  48, 169,   6,   3, 279, 274,
                   275,   2, 276, 135, 283,  17, 235, 234, 254, 256, 282,  60, 121,
                   182, 120]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss3 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[255,  47,  51, 267, 250, 260,  54, 265, 248, 246,  52, 286,  40,
                   237, 186, 118, 309, 180, 179, 178, 122, 124, 175, 173, 125, 165,
                   315, 157, 319, 155,  76, 188,  92, 190,  33,  31, 228,  78,  29,
                   189,  27,  84,  94,  25, 298, 103, 105, 112, 306,  86]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss4 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[19, 311,   4,  32, 321,  36,   0,  56, 196, 109, 107, 172, 102,
                    97, 217, 323, 230, 232,  79, 136, 156, 141, 253, 257, 142, 233,
                   193]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss5 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            target_mask = observed_mask - cond_mask
            mm = torch.zeros_like(target_mask).to(self.device)
            mm[:,[134, 183, 132, 158, 168, 171, 138, 174, 203, 115,  43,  23,  99,
                   210, 295, 218,  35, 149,  74,  69, 243,  67,  57, 187, 146]] = 1.
            target_mask = target_mask * mm
            residual = (noise - predicted) * target_mask
            num_eval = target_mask.sum()
            loss6 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            
            loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6)/6
            ###############################
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)
        else:
            if not self.use_guide:
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
                total_input = torch.cat([cond_obs, noisy_target], dim=1)
            else:
                total_input = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples, itp_info):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    if not self.use_guide:
                        cond_obs = (cond_mask * observed_data).unsqueeze(1)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    else:
                        diff_input = ((1 - cond_mask) * current_sample).unsqueeze(1)  # (B,1,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device), itp_info, cond_mask)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
            coeffs,
            cond_mask,
        ) = self.process_data(batch)

        side_info = self.get_side_info(observed_tp, cond_mask)
        itp_info = None
        if self.use_guide:
            itp_info = coeffs.unsqueeze(1)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        output = loss_func(observed_data, cond_mask, observed_mask, side_info, itp_info, is_train)
        return output

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            coeffs,
            _,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)
            itp_info = None
            if self.use_guide:
                itp_info = coeffs.unsqueeze(1)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples, itp_info)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class PriSTI_aqi36(PriSTI):
    def __init__(self, config, device, name, target_dim=36, seq_len=36):
        super(PriSTI_aqi36, self).__init__(target_dim, seq_len, config, device, name)
        self.config = config

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()
        coeffs = None
        if self.config['model']['use_guide']:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)

        if self.config['model']['use_guide']:
            coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        )



class PriSTI_MetrLA(PriSTI):
    def __init__(self, config, device, name, target_dim=207, seq_len=24):
        super(PriSTI_MetrLA, self).__init__(target_dim, seq_len, config, device, name)
        self.config = config

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        coeffs = None
        if self.config['model']['use_guide']:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
        for_pattern_mask = observed_mask

        if self.config['model']['use_guide']:
            coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        )


class PriSTI_PemsBAY(PriSTI):
    def __init__(self, config, device, name, target_dim=325, seq_len=24):
        super(PriSTI_PemsBAY, self).__init__(target_dim, seq_len, config, device, name)
        self.config = config

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        coeffs = None
        if self.config['model']['use_guide']:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
        for_pattern_mask = observed_mask

        if self.config['model']['use_guide']:
            coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        )

