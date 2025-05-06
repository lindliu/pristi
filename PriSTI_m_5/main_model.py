import numpy as np
import torch
import torch.nn as nn
from diff_models import Guide_diff_metrla, Guide_diff_pm25, Guide_diff_pemsbay


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
        if self.name == 'metrla':
            self.diffmodel = Guide_diff_metrla(config_diff, input_dim, target_dim, self.use_guide)
        if self.name == 'pemsbay':
            self.diffmodel = Guide_diff_pemsbay(config_diff, input_dim, target_dim, self.use_guide)

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
            # ########## pms25 old ############
            # target_mask = observed_mask - cond_mask
            # target_mask[:,[0,  1, 22, 23, 24, 26, 27, 28, 29, 30, 31,25, 32, 33, 34, 35],:] = 0
            # residual = (noise - predicted) * target_mask
            # num_eval = target_mask.sum()
            # loss1 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            # target_mask = observed_mask - cond_mask
            # target_mask[:,[25, 32, 33, 34, 35, 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],:] = 0
            # residual = (noise - predicted) * target_mask
            # num_eval = target_mask.sum()
            # loss2 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            # target_mask = observed_mask - cond_mask
            # target_mask[:,[2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,0,  1, 22, 23, 24, 26, 27, 28, 29, 30, 31],:] = 0
            # residual = (noise - predicted) * target_mask
            # num_eval = target_mask.sum()
            # loss3 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
            # loss = (loss1 + loss2 + loss3)/3
            # ###############################
            
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
            
            
           #  ########## metrla old ############
           #  target_mask = observed_mask - cond_mask
           #  target_mask[:,[206,  91, 174,  87,  38, 170,  41, 165, 139, 163,  47,  48,  82,
           #          81, 160,  52, 159,  56, 141, 155,  60,  76,  63, 150,  73, 148,
           #         133,  95, 144, 191,  12, 127,  96,  15, 120, 193, 187, 100, 125,
           #           4,  25, 204,  23],:] = 0
           #  residual = (noise - predicted) * target_mask
           #  num_eval = target_mask.sum()
           #  loss1 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
           
           #  target_mask = observed_mask - cond_mask
           #  target_mask[:,[75, 113, 111, 142,  74, 124, 140,  80, 119,
           # 115, 112,  94, 104, 205, 130, 138, 137, 101,  86, 131, 136,  72,
           # 134,  97,  93, 128,  89, 157,  69, 145,  26,  24, 186, 188,  19,
           # 190,  17, 181,  16, 195, 196, 197, 198,   6,   5,   3, 199, 192,
           # 180, 182,  33, 146, 149, 154,  61, 116,  57, 158,  54, 162, 161,
           #  43,  42, 166, 172, 175,  34, 164, 117],:] = 0
           #  residual = (noise - predicted) * target_mask
           #  num_eval = target_mask.sum()
           #  loss2 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
           #  target_mask = observed_mask - cond_mask
           #  target_mask[:,[176, 122, 132, 185, 151,
           #  183, 103,  64,  84,   9,  88,   8,  14,  78,  98,  77, 102,   2,
           #   28,  59,  40,  70,  29,  68,  83,  79],:] = 0
           #  residual = (noise - predicted) * target_mask
           #  num_eval = target_mask.sum()
           #  loss3 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            
           #  target_mask = observed_mask - cond_mask
           #  target_mask[:,[189, 153, 152, 114,  50,
           #   45,  44, 167,  31, 179, 178,  21, 156,  35, 105,  10,  90, 121,
           #   203,  99,  18, 106, 126, 200],:] = 0
           #  residual = (noise - predicted) * target_mask
           #  num_eval = target_mask.sum()
           #  loss4 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        
           #  target_mask = observed_mask - cond_mask
           #  target_mask[:,[177,   1,  30, 202, 201,  13,  27,
           #    22,   7,  20, 194,  32, 184,  11,  49, 173, 110, 109, 123, 108,
           #   107, 129,  92, 135,  85, 143,  71, 147,  67,  66,  65,  62,  58,
           #    55,  53,  51, 118,  46, 168, 169,  39, 171,  37,  36,   0],:] = 0
           #  residual = (noise - predicted) * target_mask
           #  num_eval = target_mask.sum()
           #  loss5 = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        
           #  loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5
            #############################
            
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

