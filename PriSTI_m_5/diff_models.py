from layers import *


class Guide_diff_metrla(nn.Module):
    def __init__(self, config, inputdim=1, target_dim=36, is_itp=False):
        super().__init__()
        self.channels = config["channels"]
        self.is_itp = is_itp
        self.itp_channels = None
        if self.is_itp:
            self.itp_channels = config["channels"]
            self.itp_projection = Conv1d_with_init(inputdim-1, self.itp_channels, 1)

            self.itp_modeling = GuidanceConstruct(channels=self.itp_channels, nheads=config["nheads"], target_dim=target_dim,
                                            order=2, include_self=True, device=config["device"], is_adp=config["is_adp"],
                                            adj_file=config["adj_file"], proj_t=config["proj_t"])
            self.cond_projection = Conv1d_with_init(config["side_dim"], self.itp_channels, 1)
            self.itp_projection2 = Conv1d_with_init(self.itp_channels, 1, 1)

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        if config["adj_file"] == 'AQI36':
            self.adj = get_adj_AQI36()
            # self.adj = np.load('adj_corr.npy')
        elif config["adj_file"] == 'metr-la':
            self.adj = get_similarity_metrla(thr=0.1)
        elif config["adj_file"] == 'pems-bay':
            self.adj = get_similarity_pemsbay(thr=0.1)
        self.device = config["device"]
        self.support = compute_support_gwn(self.adj, device=config["device"])
        self.is_adp = config["is_adp"]
        if self.is_adp:
            node_num = self.adj.shape[0]
            self.nodevec1 = nn.Parameter(torch.randn(node_num, 10).to(self.device), requires_grad=True).to(self.device)
            self.nodevec2 = nn.Parameter(torch.randn(10, node_num).to(self.device), requires_grad=True).to(self.device)
            self.support.append([self.nodevec1, self.nodevec2])

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.output_projection1_1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2_1 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2_1.weight)
        self.output_projection1_2 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2_2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2_2.weight)
        self.output_projection1_3 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2_3 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2_3.weight)
        self.output_projection1_4 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2_4 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2_4.weight)
        self.output_projection1_5 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2_5 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2_5.weight)
        
        self.residual_layers = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=target_dim,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=config["device"],
                    adj_file=config["adj_file"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                )
                for _ in range(config["layers"])
            ]
        )
        
        
        self.residual_layers1 = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=target_dim,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=config["device"],
                    adj_file=config["adj_file"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                )
                for _ in range(config["layers"])
            ]
        )
        self.residual_layers2 = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=target_dim,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=config["device"],
                    adj_file=config["adj_file"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                )
                for _ in range(config["layers"])
            ]
        )
        self.residual_layers3 = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=target_dim,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=config["device"],
                    adj_file=config["adj_file"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                )
                for _ in range(config["layers"])
            ]
        )
        
        self.residual_layers4 = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=target_dim,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=config["device"],
                    adj_file=config["adj_file"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                )
                for _ in range(config["layers"])
            ]
        )
        
        self.residual_layers5 = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=target_dim,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=config["device"],
                    adj_file=config["adj_file"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, side_info, diffusion_step, itp_x, cond_mask):
        if self.is_itp:
            x = torch.cat([x, itp_x], dim=1)
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        if self.is_itp:
            itp_x = itp_x.reshape(B, inputdim-1, K * L)
            itp_x = self.itp_projection(itp_x)
            itp_cond_info = side_info.reshape(B, -1, K * L)
            itp_cond_info = self.cond_projection(itp_cond_info)
            itp_x = itp_x + itp_cond_info
            itp_x = self.itp_modeling(itp_x, [B, self.itp_channels, K, L], self.support)
            itp_x = F.relu(itp_x)
            itp_x = itp_x.reshape(B, self.itp_channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for i in range(len(self.residual_layers)):
            x, skip_connection = self.residual_layers[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        # x = x.reshape(B, self.channels, K * L)
        # x = self.output_projection1(x)  # (B,channel,K*L)
        # x = F.relu(x)
        # x = self.output_projection2(x)  # (B,1,K*L)
        # x = x.reshape(B, K, L)
        
        
        # ############ pms25 ################
        # skip1 = []
        # for i in range(len(self.residual_layers1)):
        #     x, skip_connection = self.residual_layers1[i](x, side_info, diffusion_emb, itp_x, self.support)
        #     skip1.append(skip_connection)
        # x1 = torch.sum(torch.stack(skip1), dim=0) / math.sqrt(len(self.residual_layers1))
        # x1 = x1.reshape(B, self.channels, K * L)
        # x1 = self.output_projection1_1(x1)  # (B,channel,K*L)
        # x1 = F.relu(x1)
        # x1 = self.output_projection2_1(x1)  # (B,1,K*L)
        # x1 = x1.reshape(B, K, L)
        
        # skip2 = []
        # for i in range(len(self.residual_layers2)):
        #     x, skip_connection = self.residual_layers2[i](x, side_info, diffusion_emb, itp_x, self.support)
        #     skip2.append(skip_connection)
        # x2 = torch.sum(torch.stack(skip2), dim=0) / math.sqrt(len(self.residual_layers2))
        # x2 = x2.reshape(B, self.channels, K * L)
        # x2 = self.output_projection1_2(x2)  # (B,channel,K*L)
        # x2 = F.relu(x2)
        # x2 = self.output_projection2_2(x2)  # (B,1,K*L)
        # x2 = x2.reshape(B, K, L)
        
        # skip3 = []
        # for i in range(len(self.residual_layers3)):
        #     x, skip_connection = self.residual_layers3[i](x, side_info, diffusion_emb, itp_x, self.support)
        #     skip3.append(skip_connection)
        # x3 = torch.sum(torch.stack(skip3), dim=0) / math.sqrt(len(self.residual_layers3))
        # x3 = x3.reshape(B, self.channels, K * L)
        # x3 = self.output_projection1_3(x3)  # (B,channel,K*L)
        # x3 = F.relu(x3)
        # x3 = self.output_projection2_3(x3)  # (B,1,K*L)
        # x3 = x3.reshape(B, K, L)
        
        # ######### old ########
        # # x = torch.cat([x1[:,[2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        # #         19, 20, 21],:],x2[:,[0,  1, 22, 23, 24, 26, 27, 28, 29, 30, 31],:],x3[:,[25, 32, 33, 34, 35],:]],dim=1)
        # # x = x[:,[20, 21,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
        # #         15, 16, 17, 18, 19, 22, 23, 24, 31, 25, 26, 27, 28, 29, 30, 32, 33,
        # #         34, 35],:]
        
        # x = torch.cat([x1[:,[2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        #        19, 20, 21],:],x2[:,[0,  1, 22, 23, 24, 26, 27, 28, 29, 31],:],x3[:,[25, 30, 32, 33, 34, 35],:]],dim=1)
        # x = x[:,[20, 21,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
        #        15, 16, 17, 18, 19, 22, 23, 24, 30, 25, 26, 27, 28, 31, 29, 32, 33,
        #        34, 35],:]
        # ###################################
        
        
        
        ############ metrla ################
        skip1 = []
        for i in range(len(self.residual_layers1)):
            x, skip_connection = self.residual_layers1[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip1.append(skip_connection)
        x1 = torch.sum(torch.stack(skip1), dim=0) / math.sqrt(len(self.residual_layers1))
        x1 = x1.reshape(B, self.channels, K * L)
        x1 = self.output_projection1_1(x1)  # (B,channel,K*L)
        x1 = F.relu(x1)
        x1 = self.output_projection2_1(x1)  # (B,1,K*L)
        x1 = x1.reshape(B, K, L)
        
        skip2 = []
        for i in range(len(self.residual_layers2)):
            x, skip_connection = self.residual_layers2[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip2.append(skip_connection)
        x2 = torch.sum(torch.stack(skip2), dim=0) / math.sqrt(len(self.residual_layers2))
        x2 = x2.reshape(B, self.channels, K * L)
        x2 = self.output_projection1_2(x2)  # (B,channel,K*L)
        x2 = F.relu(x2)
        x2 = self.output_projection2_2(x2)  # (B,1,K*L)
        x2 = x2.reshape(B, K, L)
        
        skip3 = []
        for i in range(len(self.residual_layers3)):
            x, skip_connection = self.residual_layers3[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip3.append(skip_connection)
        x3 = torch.sum(torch.stack(skip3), dim=0) / math.sqrt(len(self.residual_layers3))
        x3 = x3.reshape(B, self.channels, K * L)
        x3 = self.output_projection1_3(x3)  # (B,channel,K*L)
        x3 = F.relu(x3)
        x3 = self.output_projection2_3(x3)  # (B,1,K*L)
        x3 = x3.reshape(B, K, L)
        
        skip4 = []
        for i in range(len(self.residual_layers4)):
            x, skip_connection = self.residual_layers4[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip4.append(skip_connection)
        x4 = torch.sum(torch.stack(skip4), dim=0) / math.sqrt(len(self.residual_layers4))
        x4 = x4.reshape(B, self.channels, K * L)
        x4 = self.output_projection1_4(x4)  # (B,channel,K*L)
        x4 = F.relu(x4)
        x4 = self.output_projection2_3(x4)  # (B,1,K*L)
        x4 = x4.reshape(B, K, L)
        
        skip5 = []
        for i in range(len(self.residual_layers5)):
            x, skip_connection = self.residual_layers5[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip5.append(skip_connection)
        x5 = torch.sum(torch.stack(skip5), dim=0) / math.sqrt(len(self.residual_layers5))
        x5 = x5.reshape(B, self.channels, K * L)
        x5 = self.output_projection1_5(x5)  # (B,channel,K*L)
        x5 = F.relu(x5)
        x5 = self.output_projection2_5(x5)  # (B,1,K*L)
        x5 = x5.reshape(B, K, L)
        
        x = torch.cat([x1[:,[206,  91, 174,  87,  38, 170,  41, 165, 139, 163,  47,  48,  82,
                81, 160,  52, 159,  56, 141, 155,  60,  76,  63, 150,  73, 148,
                133,  95, 144, 191,  12, 127,  96,  15, 120, 193, 187, 100, 125,
                  4,  25, 204,  23],:],
                        x2[:,[75, 113, 111, 142,  74, 124, 140,  80, 119,
                      115, 112,  94, 104, 205, 130, 138, 137, 101,  86, 131, 136,  72,
                      134,  97,  93, 128,  89, 157,  69, 145,  26,  24, 186, 188,  19,
                      190,  17, 181,  16, 195, 196, 197, 198,   6,   5,   3, 199, 192,
                      180, 182,  33, 146, 149, 154,  61, 116,  57, 158,  54, 162, 161,
                      43,  42, 166, 172, 175,  34, 164, 117],:],
                        x3[:,[176, 122, 132, 185, 151,
                      183, 103,  64,  84,   9,  88,   8,  14,  78,  98,  77, 102,   2,
                        28,  59,  40,  70,  29,  68,  83,  79],:],
                        x4[:,[189, 153, 152, 114,  50,
                        45,  44, 167,  31, 179, 178,  21, 156,  35, 105,  10,  90, 121,
                        203,  99,  18, 106, 126, 200],:],
                        x5[:,[177,   1,  30, 202, 201,  13,  27,
                        22,   7,  20, 194,  32, 184,  11,  49, 173, 110, 109, 123, 108,
                        107, 129,  92, 135,  85, 143,  71, 147,  67,  66,  65,  62,  58,
                        55,  53,  51, 118,  46, 168, 169,  39, 171,  37,  36,   0],:]
                        ],dim=1)
        x = x[:,[206, 163, 129,  88,  39,  87,  86, 170, 123, 121, 153, 175,  30,
                167, 124,  33,  81,  79, 158,  77, 171, 149, 169,  42,  74,  40,
                73, 168, 130, 134, 164, 146, 173,  93, 109, 151, 205, 204,   4,
                202, 132,   6, 105, 104, 144, 143, 199,  10,  11, 176, 142, 197,
                15, 196, 101, 195,  17,  99, 194, 131,  20,  97, 193,  22, 119,
                192, 191, 190, 135,  71, 133, 188,  64,  24,  47,  43,  21, 127,
                125, 137,  50,  13,  12, 136, 120, 186,  61,   3, 122,  69, 154,
                  1, 184,  67,  54,  27,  32,  66, 126, 157,  37,  60, 128, 118,
                55, 152, 159, 182, 181, 179, 178,  45,  53,  44, 141,  52,  98,
                111, 198,  51,  34, 155, 113, 180,  48,  38, 160,  31,  68, 183,
                57,  62, 114,  26,  65, 185,  63,  59,  58,   8,  49,  18,  46,
                187,  28,  72,  94, 189,  25,  95,  23, 116, 140, 139,  96,  19,
                150,  70, 100,  16,  14, 103, 102,   9, 110,   7, 106, 145, 200,
                201,   5, 203, 107, 177,   2, 108, 112, 162, 148, 147,  91,  80,
                92, 117, 174, 115,  75,  36,  76, 138,  78,  29,  90,  35, 172,
                82,  83,  84,  85,  89, 161, 166, 165, 156,  41,  56,   0],:]
        ################################
        
        return x



class Guide_diff_pm25(nn.Module):
    def __init__(self, config, inputdim=1, target_dim=36, is_itp=False):
        super().__init__()
        self.channels = config["channels"]
        self.is_itp = is_itp
        self.itp_channels = None
        if self.is_itp:
            self.itp_channels = config["channels"]
            self.itp_projection = Conv1d_with_init(inputdim-1, self.itp_channels, 1)

            self.itp_modeling = GuidanceConstruct(channels=self.itp_channels, nheads=config["nheads"], target_dim=target_dim,
                                            order=2, include_self=True, device=config["device"], is_adp=config["is_adp"],
                                            adj_file=config["adj_file"], proj_t=config["proj_t"])
            self.cond_projection = Conv1d_with_init(config["side_dim"], self.itp_channels, 1)
            self.itp_projection2 = Conv1d_with_init(self.itp_channels, 1, 1)

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        if config["adj_file"] == 'AQI36':
            self.adj = get_adj_AQI36()
            # self.adj = np.load('adj_corr.npy')
        elif config["adj_file"] == 'metr-la':
            self.adj = get_similarity_metrla(thr=0.1)
        elif config["adj_file"] == 'pems-bay':
            self.adj = get_similarity_pemsbay(thr=0.1)
        self.device = config["device"]
        self.support = compute_support_gwn(self.adj, device=config["device"])
        self.is_adp = config["is_adp"]
        if self.is_adp:
            node_num = self.adj.shape[0]
            self.nodevec1 = nn.Parameter(torch.randn(node_num, 10).to(self.device), requires_grad=True).to(self.device)
            self.nodevec2 = nn.Parameter(torch.randn(10, node_num).to(self.device), requires_grad=True).to(self.device)
            self.support.append([self.nodevec1, self.nodevec2])

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.output_projection1_1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2_1 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2_1.weight)
        self.output_projection1_2 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2_2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2_2.weight)
        self.output_projection1_3 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2_3 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2_3.weight)
        # self.output_projection1_4 = Conv1d_with_init(self.channels, self.channels, 1)
        # self.output_projection2_4 = Conv1d_with_init(self.channels, 1, 1)
        # nn.init.zeros_(self.output_projection2_4.weight)
        # self.output_projection1_5 = Conv1d_with_init(self.channels, self.channels, 1)
        # self.output_projection2_5 = Conv1d_with_init(self.channels, 1, 1)
        # nn.init.zeros_(self.output_projection2_5.weight)
        
        self.residual_layers = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=target_dim,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=config["device"],
                    adj_file=config["adj_file"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                )
                for _ in range(config["layers"])
            ]
        )
        
        
        self.residual_layers1 = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=target_dim,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=config["device"],
                    adj_file=config["adj_file"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                )
                for _ in range(config["layers"])
            ]
        )
        self.residual_layers2 = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=target_dim,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=config["device"],
                    adj_file=config["adj_file"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                )
                for _ in range(config["layers"])
            ]
        )
        self.residual_layers3 = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=target_dim,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=config["device"],
                    adj_file=config["adj_file"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                )
                for _ in range(config["layers"])
            ]
        )
        
        # self.residual_layers4 = nn.ModuleList(
        #     [
        #         NoiseProject(
        #             side_dim=config["side_dim"],
        #             channels=self.channels,
        #             diffusion_embedding_dim=config["diffusion_embedding_dim"],
        #             nheads=config["nheads"],
        #             target_dim=target_dim,
        #             proj_t=config["proj_t"],
        #             is_adp=config["is_adp"],
        #             device=config["device"],
        #             adj_file=config["adj_file"],
        #             is_cross_t=config["is_cross_t"],
        #             is_cross_s=config["is_cross_s"],
        #         )
        #         for _ in range(config["layers"])
        #     ]
        # )
        
        # self.residual_layers5 = nn.ModuleList(
        #     [
        #         NoiseProject(
        #             side_dim=config["side_dim"],
        #             channels=self.channels,
        #             diffusion_embedding_dim=config["diffusion_embedding_dim"],
        #             nheads=config["nheads"],
        #             target_dim=target_dim,
        #             proj_t=config["proj_t"],
        #             is_adp=config["is_adp"],
        #             device=config["device"],
        #             adj_file=config["adj_file"],
        #             is_cross_t=config["is_cross_t"],
        #             is_cross_s=config["is_cross_s"],
        #         )
        #         for _ in range(config["layers"])
        #     ]
        # )

    def forward(self, x, side_info, diffusion_step, itp_x, cond_mask):
        if self.is_itp:
            x = torch.cat([x, itp_x], dim=1)
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        if self.is_itp:
            itp_x = itp_x.reshape(B, inputdim-1, K * L)
            itp_x = self.itp_projection(itp_x)
            itp_cond_info = side_info.reshape(B, -1, K * L)
            itp_cond_info = self.cond_projection(itp_cond_info)
            itp_x = itp_x + itp_cond_info
            itp_x = self.itp_modeling(itp_x, [B, self.itp_channels, K, L], self.support)
            itp_x = F.relu(itp_x)
            itp_x = itp_x.reshape(B, self.itp_channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for i in range(len(self.residual_layers)):
            x, skip_connection = self.residual_layers[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        # x = x.reshape(B, self.channels, K * L)
        # x = self.output_projection1(x)  # (B,channel,K*L)
        # x = F.relu(x)
        # x = self.output_projection2(x)  # (B,1,K*L)
        # x = x.reshape(B, K, L)
        
        
        ############ pms25 ################
        skip1 = []
        for i in range(len(self.residual_layers1)):
            x, skip_connection = self.residual_layers1[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip1.append(skip_connection)
        x1 = torch.sum(torch.stack(skip1), dim=0) / math.sqrt(len(self.residual_layers1))
        x1 = x1.reshape(B, self.channels, K * L)
        x1 = self.output_projection1_1(x1)  # (B,channel,K*L)
        x1 = F.relu(x1)
        x1 = self.output_projection2_1(x1)  # (B,1,K*L)
        x1 = x1.reshape(B, K, L)
        
        skip2 = []
        for i in range(len(self.residual_layers2)):
            x, skip_connection = self.residual_layers2[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip2.append(skip_connection)
        x2 = torch.sum(torch.stack(skip2), dim=0) / math.sqrt(len(self.residual_layers2))
        x2 = x2.reshape(B, self.channels, K * L)
        x2 = self.output_projection1_2(x2)  # (B,channel,K*L)
        x2 = F.relu(x2)
        x2 = self.output_projection2_2(x2)  # (B,1,K*L)
        x2 = x2.reshape(B, K, L)
        
        skip3 = []
        for i in range(len(self.residual_layers3)):
            x, skip_connection = self.residual_layers3[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip3.append(skip_connection)
        x3 = torch.sum(torch.stack(skip3), dim=0) / math.sqrt(len(self.residual_layers3))
        x3 = x3.reshape(B, self.channels, K * L)
        x3 = self.output_projection1_3(x3)  # (B,channel,K*L)
        x3 = F.relu(x3)
        x3 = self.output_projection2_3(x3)  # (B,1,K*L)
        x3 = x3.reshape(B, K, L)
        
        ######### old ########
        # x = torch.cat([x1[:,[2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        #         19, 20, 21],:],x2[:,[0,  1, 22, 23, 24, 26, 27, 28, 29, 30, 31],:],x3[:,[25, 32, 33, 34, 35],:]],dim=1)
        # x = x[:,[20, 21,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
        #         15, 16, 17, 18, 19, 22, 23, 24, 31, 25, 26, 27, 28, 29, 30, 32, 33,
        #         34, 35],:]
        
        x = torch.cat([x1[:,[2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                19, 20, 21],:],x2[:,[0,  1, 22, 23, 24, 26, 27, 28, 29, 31],:],x3[:,[25, 30, 32, 33, 34, 35],:]],dim=1)
        x = x[:,[20, 21,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                15, 16, 17, 18, 19, 22, 23, 24, 30, 25, 26, 27, 28, 31, 29, 32, 33,
                34, 35],:]
        
        #by np.argsort([2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                # 19, 20, 21,0,  1, 22, 23, 24, 26, 27, 28, 29, 31,25, 30, 32, 33, 34, 35])
        ###################################
        
        
        
        # ############ metrla ################
        # skip1 = []
        # for i in range(len(self.residual_layers1)):
        #     x, skip_connection = self.residual_layers1[i](x, side_info, diffusion_emb, itp_x, self.support)
        #     skip1.append(skip_connection)
        # x1 = torch.sum(torch.stack(skip1), dim=0) / math.sqrt(len(self.residual_layers1))
        # x1 = x1.reshape(B, self.channels, K * L)
        # x1 = self.output_projection1_1(x1)  # (B,channel,K*L)
        # x1 = F.relu(x1)
        # x1 = self.output_projection2_1(x1)  # (B,1,K*L)
        # x1 = x1.reshape(B, K, L)
        
        # skip2 = []
        # for i in range(len(self.residual_layers2)):
        #     x, skip_connection = self.residual_layers2[i](x, side_info, diffusion_emb, itp_x, self.support)
        #     skip2.append(skip_connection)
        # x2 = torch.sum(torch.stack(skip2), dim=0) / math.sqrt(len(self.residual_layers2))
        # x2 = x2.reshape(B, self.channels, K * L)
        # x2 = self.output_projection1_2(x2)  # (B,channel,K*L)
        # x2 = F.relu(x2)
        # x2 = self.output_projection2_2(x2)  # (B,1,K*L)
        # x2 = x2.reshape(B, K, L)
        
        # skip3 = []
        # for i in range(len(self.residual_layers3)):
        #     x, skip_connection = self.residual_layers3[i](x, side_info, diffusion_emb, itp_x, self.support)
        #     skip3.append(skip_connection)
        # x3 = torch.sum(torch.stack(skip3), dim=0) / math.sqrt(len(self.residual_layers3))
        # x3 = x3.reshape(B, self.channels, K * L)
        # x3 = self.output_projection1_3(x3)  # (B,channel,K*L)
        # x3 = F.relu(x3)
        # x3 = self.output_projection2_3(x3)  # (B,1,K*L)
        # x3 = x3.reshape(B, K, L)
        
        # skip4 = []
        # for i in range(len(self.residual_layers4)):
        #     x, skip_connection = self.residual_layers4[i](x, side_info, diffusion_emb, itp_x, self.support)
        #     skip4.append(skip_connection)
        # x4 = torch.sum(torch.stack(skip4), dim=0) / math.sqrt(len(self.residual_layers4))
        # x4 = x4.reshape(B, self.channels, K * L)
        # x4 = self.output_projection1_4(x4)  # (B,channel,K*L)
        # x4 = F.relu(x4)
        # x4 = self.output_projection2_3(x4)  # (B,1,K*L)
        # x4 = x4.reshape(B, K, L)
        
        # skip5 = []
        # for i in range(len(self.residual_layers5)):
        #     x, skip_connection = self.residual_layers5[i](x, side_info, diffusion_emb, itp_x, self.support)
        #     skip5.append(skip_connection)
        # x5 = torch.sum(torch.stack(skip5), dim=0) / math.sqrt(len(self.residual_layers5))
        # x5 = x5.reshape(B, self.channels, K * L)
        # x5 = self.output_projection1_5(x5)  # (B,channel,K*L)
        # x5 = F.relu(x5)
        # x5 = self.output_projection2_5(x5)  # (B,1,K*L)
        # x5 = x5.reshape(B, K, L)
        
        # x = torch.cat([x1[:,[206,  91, 174,  87,  38, 170,  41, 165, 139, 163,  47,  48,  82,
        #         81, 160,  52, 159,  56, 141, 155,  60,  76,  63, 150,  73, 148,
        #         133,  95, 144, 191,  12, 127,  96,  15, 120, 193, 187, 100, 125,
        #           4,  25, 204,  23],:],
        #                 x2[:,[75, 113, 111, 142,  74, 124, 140,  80, 119,
        #               115, 112,  94, 104, 205, 130, 138, 137, 101,  86, 131, 136,  72,
        #               134,  97,  93, 128,  89, 157,  69, 145,  26,  24, 186, 188,  19,
        #               190,  17, 181,  16, 195, 196, 197, 198,   6,   5,   3, 199, 192,
        #               180, 182,  33, 146, 149, 154,  61, 116,  57, 158,  54, 162, 161,
        #               43,  42, 166, 172, 175,  34, 164, 117],:],
        #                 x3[:,[176, 122, 132, 185, 151,
        #               183, 103,  64,  84,   9,  88,   8,  14,  78,  98,  77, 102,   2,
        #                 28,  59,  40,  70,  29,  68,  83,  79],:],
        #                 x4[:,[189, 153, 152, 114,  50,
        #                 45,  44, 167,  31, 179, 178,  21, 156,  35, 105,  10,  90, 121,
        #                 203,  99,  18, 106, 126, 200],:],
        #                 x5[:,[177,   1,  30, 202, 201,  13,  27,
        #                 22,   7,  20, 194,  32, 184,  11,  49, 173, 110, 109, 123, 108,
        #                 107, 129,  92, 135,  85, 143,  71, 147,  67,  66,  65,  62,  58,
        #                 55,  53,  51, 118,  46, 168, 169,  39, 171,  37,  36,   0],:]
        #                 ],dim=1)
        # x = x[:,[206, 163, 129,  88,  39,  87,  86, 170, 123, 121, 153, 175,  30,
        #         167, 124,  33,  81,  79, 158,  77, 171, 149, 169,  42,  74,  40,
        #         73, 168, 130, 134, 164, 146, 173,  93, 109, 151, 205, 204,   4,
        #         202, 132,   6, 105, 104, 144, 143, 199,  10,  11, 176, 142, 197,
        #         15, 196, 101, 195,  17,  99, 194, 131,  20,  97, 193,  22, 119,
        #         192, 191, 190, 135,  71, 133, 188,  64,  24,  47,  43,  21, 127,
        #         125, 137,  50,  13,  12, 136, 120, 186,  61,   3, 122,  69, 154,
        #           1, 184,  67,  54,  27,  32,  66, 126, 157,  37,  60, 128, 118,
        #         55, 152, 159, 182, 181, 179, 178,  45,  53,  44, 141,  52,  98,
        #         111, 198,  51,  34, 155, 113, 180,  48,  38, 160,  31,  68, 183,
        #         57,  62, 114,  26,  65, 185,  63,  59,  58,   8,  49,  18,  46,
        #         187,  28,  72,  94, 189,  25,  95,  23, 116, 140, 139,  96,  19,
        #         150,  70, 100,  16,  14, 103, 102,   9, 110,   7, 106, 145, 200,
        #         201,   5, 203, 107, 177,   2, 108, 112, 162, 148, 147,  91,  80,
        #         92, 117, 174, 115,  75,  36,  76, 138,  78,  29,  90,  35, 172,
        #         82,  83,  84,  85,  89, 161, 166, 165, 156,  41,  56,   0],:]
        # ################################
        
        return x



import torch

class fff(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        
        self.weight = nn.Parameter(torch.randn([channels, 36, 19]).to('cuda'), requires_grad=True)
        # torch.nn.init.kaiming_uniform_(self.weight)
        
    def forward(self, x):        
        x = torch.fft.rfftn(x, norm='ortho', dim=(2,3))
        # x = x + Dense(act(temb))[:,:,None ,None]
        x = torch.einsum('bchw , chw -> bchw', x, self.weight)
        return torch.fft.irfftn(x, norm='ortho')

class NoiseProject(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, target_dim, proj_t, order=2, include_self=True,
                 device=None, is_adp=False, adj_file=None, is_cross_t=False, is_cross_s=True):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.forward_time = TemporalLearning(channels=channels, nheads=nheads, is_cross=is_cross_t)
        self.forward_feature = SpatialLearning(channels=channels, nheads=nheads, target_dim=target_dim,
                                               order=order, include_self=include_self, device=device, is_adp=is_adp,
                                               adj_file=adj_file, proj_t=proj_t, is_cross=is_cross_s)

        # self.ff = fff(channels)
        
    def forward(self, x, side_info, diffusion_emb, itp_info, support):
        B, channel, K, L = x.shape
        base_shape = x.shape
        
        # x = self.ff(x)
        # print(x.shape)
        
        x = x.reshape(B, channel, K * L)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape, itp_info)
        y = self.forward_feature(y, base_shape, support, itp_info)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, side_dim, _, _ = side_info.shape
        side_info = side_info.reshape(B, side_dim, K * L)
        side_info = self.cond_projection(side_info)  # (B,2*channel,K*L)
        y = y + side_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip






class Guide_diff_pemsbay(nn.Module):
    def __init__(self, config, inputdim=1, target_dim=36, is_itp=False):
        super().__init__()
        self.channels = config["channels"]
        self.is_itp = is_itp
        self.itp_channels = None
        if self.is_itp:
            self.itp_channels = config["channels"]
            self.itp_projection = Conv1d_with_init(inputdim-1, self.itp_channels, 1)

            self.itp_modeling = GuidanceConstruct(channels=self.itp_channels, nheads=config["nheads"], target_dim=target_dim,
                                            order=2, include_self=True, device=config["device"], is_adp=config["is_adp"],
                                            adj_file=config["adj_file"], proj_t=config["proj_t"])
            self.cond_projection = Conv1d_with_init(config["side_dim"], self.itp_channels, 1)
            self.itp_projection2 = Conv1d_with_init(self.itp_channels, 1, 1)

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        if config["adj_file"] == 'AQI36':
            self.adj = get_adj_AQI36()
            # self.adj = np.load('adj_corr.npy')
        elif config["adj_file"] == 'metr-la':
            self.adj = get_similarity_metrla(thr=0.1)
        elif config["adj_file"] == 'pems-bay':
            self.adj = get_similarity_pemsbay(thr=0.1)
        self.device = config["device"]
        self.support = compute_support_gwn(self.adj, device=config["device"])
        self.is_adp = config["is_adp"]
        if self.is_adp:
            node_num = self.adj.shape[0]
            self.nodevec1 = nn.Parameter(torch.randn(node_num, 10).to(self.device), requires_grad=True).to(self.device)
            self.nodevec2 = nn.Parameter(torch.randn(10, node_num).to(self.device), requires_grad=True).to(self.device)
            self.support.append([self.nodevec1, self.nodevec2])

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.output_projection1_1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2_1 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2_1.weight)
        self.output_projection1_2 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2_2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2_2.weight)
        self.output_projection1_3 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2_3 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2_3.weight)
        
        self.residual_layers = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=target_dim,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=config["device"],
                    adj_file=config["adj_file"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                )
                for _ in range(config["layers"])
            ]
        )
        
        
        self.residual_layers1 = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=target_dim,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=config["device"],
                    adj_file=config["adj_file"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                )
                for _ in range(config["layers"])
            ]
        )
        self.residual_layers2 = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=target_dim,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=config["device"],
                    adj_file=config["adj_file"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                )
                for _ in range(config["layers"])
            ]
        )
        self.residual_layers3 = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=target_dim,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=config["device"],
                    adj_file=config["adj_file"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                )
                for _ in range(config["layers"])
            ]
        )
        

    def forward(self, x, side_info, diffusion_step, itp_x, cond_mask):
        if self.is_itp:
            x = torch.cat([x, itp_x], dim=1)
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        if self.is_itp:
            itp_x = itp_x.reshape(B, inputdim-1, K * L)
            itp_x = self.itp_projection(itp_x)
            itp_cond_info = side_info.reshape(B, -1, K * L)
            itp_cond_info = self.cond_projection(itp_cond_info)
            itp_x = itp_x + itp_cond_info
            itp_x = self.itp_modeling(itp_x, [B, self.itp_channels, K, L], self.support)
            itp_x = F.relu(itp_x)
            itp_x = itp_x.reshape(B, self.itp_channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for i in range(len(self.residual_layers)):
            x, skip_connection = self.residual_layers[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        # x = x.reshape(B, self.channels, K * L)
        # x = self.output_projection1(x)  # (B,channel,K*L)
        # x = F.relu(x)
        # x = self.output_projection2(x)  # (B,1,K*L)
        # x = x.reshape(B, K, L)
        
        
        ############ pms25 ################
        skip1 = []
        for i in range(len(self.residual_layers1)):
            x, skip_connection = self.residual_layers1[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip1.append(skip_connection)
        x1 = torch.sum(torch.stack(skip1), dim=0) / math.sqrt(len(self.residual_layers1))
        x1 = x1.reshape(B, self.channels, K * L)
        x1 = self.output_projection1_1(x1)  # (B,channel,K*L)
        x1 = F.relu(x1)
        x1 = self.output_projection2_1(x1)  # (B,1,K*L)
        x1 = x1.reshape(B, K, L)
        
        skip2 = []
        for i in range(len(self.residual_layers2)):
            x, skip_connection = self.residual_layers2[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip2.append(skip_connection)
        x2 = torch.sum(torch.stack(skip2), dim=0) / math.sqrt(len(self.residual_layers2))
        x2 = x2.reshape(B, self.channels, K * L)
        x2 = self.output_projection1_2(x2)  # (B,channel,K*L)
        x2 = F.relu(x2)
        x2 = self.output_projection2_2(x2)  # (B,1,K*L)
        x2 = x2.reshape(B, K, L)
        
        skip3 = []
        for i in range(len(self.residual_layers3)):
            x, skip_connection = self.residual_layers3[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip3.append(skip_connection)
        x3 = torch.sum(torch.stack(skip3), dim=0) / math.sqrt(len(self.residual_layers3))
        x3 = x3.reshape(B, self.channels, K * L)
        x3 = self.output_projection1_3(x3)  # (B,channel,K*L)
        x3 = F.relu(x3)
        x3 = self.output_projection2_3(x3)  # (B,1,K*L)
        x3 = x3.reshape(B, K, L)
        
        
        x = torch.cat([x1[:,[0,   4,   5,  14,  19,  21,  23,  25,  27,  28,  29,  30,  34,
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
               321, 322, 323, 324],:],
                x2[:,[2,   6,  13,  16,  20,  38,  39,  43,  53,  63,  68,  80, 111,
                      117, 140, 146, 148, 153, 154, 164, 170, 177, 180, 183, 185, 199,
                      206, 207, 209, 210, 212, 215, 242, 251, 265, 266, 267, 313, 319],:],
                x3[:,[1,   3,   7,   8,   9,  10,  11,  12,  15,  17,  18,  22,  24,
                     26,  31,  32,  33,  37,  41,  42,  46,  48,  49,  50,  51,  55,
                     56,  60,  61,  65,  72,  73,  74,  75,  76,  79,  81,  82,  83,
                     84,  85,  91,  95,  96,  97, 101, 102, 103, 106, 108, 109, 113,
                    114, 115, 116, 119, 120, 121, 122, 123, 126, 127, 133, 134, 135,
                    136, 141, 144, 145, 147, 151, 152, 155, 163, 167, 169, 173, 175,
                    176, 178, 179, 182, 191, 192, 195, 196, 197, 200, 201, 203, 205,
                    208, 217, 219, 221, 223, 225, 226, 227, 229, 231, 232, 233, 234,
                    235, 236, 240, 244, 245, 248, 249, 252, 254, 256, 274, 275, 276,
                    277, 278, 279, 281, 282, 283, 284, 285, 299, 300, 301, 302, 303,
                    304, 305, 306, 307, 308, 309, 311, 317, 318],:]],dim=1)
        x = x[:,[0, 186, 147, 187,   1,   2, 148, 188, 189, 190, 191, 192, 193,
               149,   3, 194, 150, 195, 196,   4, 151,   5, 197,   6, 198,   7,
               199,   8,   9,  10,  11, 200, 201, 202,  12,  13,  14, 203, 152,
               153,  15, 204, 205, 154,  16,  17, 206,  18, 207, 208, 209, 210,
                19, 155,  20, 211, 212,  21,  22,  23, 213, 214,  24, 156,  25,
               215,  26,  27, 157,  28,  29,  30, 216, 217, 218, 219, 220,  31,
                32, 221, 158, 222, 223, 224, 225, 226,  33,  34,  35,  36,  37,
               227,  38,  39,  40, 228, 229, 230,  41,  42,  43, 231, 232, 233,
                44,  45, 234,  46, 235, 236,  47, 159,  48, 237, 238, 239, 240,
               160,  49, 241, 242, 243, 244, 245,  50,  51, 246, 247,  52,  53,
                54,  55,  56, 248, 249, 250, 251,  57,  58,  59, 161, 252,  60,
                61, 253, 254, 162, 255, 163,  62,  63, 256, 257, 164, 165, 258,
                64,  65,  66,  67,  68,  69,  70, 259, 166,  71,  72, 260,  73,
               261, 167,  74,  75, 262,  76, 263, 264, 168, 265, 266, 169,  77,
               267, 170,  78, 171,  79,  80,  81,  82,  83, 268, 269,  84,  85,
               270, 271, 272,  86, 172, 273, 274,  87, 275,  88, 276, 173, 174,
               277, 175, 176,  89, 177,  90,  91, 178,  92, 278,  93, 279,  94,
               280,  95, 281,  96, 282, 283, 284,  97, 285,  98, 286, 287, 288,
               289, 290, 291,  99, 100, 101, 292, 102, 179, 103, 293, 294, 104,
               105, 295, 296, 106, 180, 297, 107, 298, 108, 299, 109, 110, 111,
               112, 113, 114, 115, 116, 181, 182, 183, 117, 118, 119, 120, 121,
               122, 300, 301, 302, 303, 304, 305, 123, 306, 307, 308, 309, 310,
               124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
               311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 137, 322,
               138, 184, 139, 140, 141, 323, 324, 185, 142, 143, 144, 145, 146],:]
        ###################################
        
        
        return x