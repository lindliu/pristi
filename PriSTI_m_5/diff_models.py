from layers import *




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








class Guide_diff_metrla_vae_error(nn.Module):
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
        
        x = torch.cat([x1[:,[103,  59, 135,  57, 136,  55,  54, 142, 145, 152, 153,  93,  44,
               154, 156, 157, 123,  40, 122,  68,  89,  94,  98,  85,  84,  83,
                99, 102,  80,  79,  78,  74, 110,  70, 111, 119, 161,  45,  19,
               178,  20, 162, 182, 184,  10,   9, 201,   7,   6, 202, 203, 204,
                 2,   1, 177, 176,  91, 175,  27,  28, 173,  31, 169, 171, 167,
               168,  24],:],
                        x2[:,[107,  95, 166, 160, 196, 105, 106, 194, 109, 138, 139, 132, 129,
                               128, 125, 108, 140, 120, 181, 174, 143, 185, 187, 180, 163,   0,
                                46,  25,  26,  16,  66,  30,  56,  32,  51,  12,  77,  71,  11,
                                 8,  36,  62],:],
                        x3[:,[37, 127,  38,  22, 114,  39,  96],:],
                        x4[:,[23,  42,  41,  29,  35, 172, 165, 170, 164,  33,  34, 158, 159,
                                90, 179,   3,   4,   5, 200, 199, 198, 197, 195, 193, 192, 191,
                               190, 189, 188,  13, 186,  14,  15, 183,  17, 155,  21,  18, 149,
                                47,  67, 118, 117, 116, 115, 113, 112,  69,  72,  73, 121,  75,
                               104, 205,  81, 101, 100,  82,  86,  97,  87,  88,  76,  65,  64,
                               124, 151, 150,  92, 148, 147, 146,  48, 144,  49,  50, 141,  52,
                                53, 137,  58, 134, 133,  60, 131, 130,  61, 126,  63,  43, 206],:]
                        ],dim=1)
        x = x[:,[92,  53,  52, 131, 132, 133,  48,  47, 106,  45,  44, 105, 102,
               145, 147, 148,  96, 150, 153,  38,  40, 152, 112, 116,  66,  94,
                95,  58,  59, 119,  98,  61, 100, 125, 126, 120, 107, 109, 111,
               114,  17, 118, 117, 205,  12,  37,  93, 155, 188, 190, 191, 101,
               193, 194,   6,   5,  99,   3, 196,   1, 199, 202, 108, 204, 180,
               179,  97, 156,  19, 163,  33, 104, 164, 165,  31, 167, 178, 103,
                30,  29,  28, 170, 173,  25,  24,  23, 174, 176, 177,  20, 129,
                56, 184,  11,  21,  68, 115, 175,  22,  26, 172, 171,  27,   0,
               168,  72,  73,  67,  82,  75,  32,  34, 162, 161, 113, 160, 159,
               158, 157,  35,  84, 166,  18,  16, 181,  81, 203, 110,  80,  79,
               201, 200,  78, 198, 197,   2,   4, 195,  76,  77,  83, 192,   7,
                87, 189,   8, 187, 186, 185, 154, 183, 182,   9,  10,  13, 151,
                14,  15, 127, 128,  70,  36,  41,  91, 124, 122,  69,  64,  65,
                62, 123,  63, 121,  60,  86,  57,  55,  54,  39, 130,  90,  85,
                42, 149,  43,  88, 146,  89, 144, 143, 142, 141, 140, 139,  74,
               138,  71, 137, 136, 135, 134,  46,  49,  50,  51, 169, 206],:]
        
        
        
        # np.argsort([103,  59, 135,  57, 136,  55,  54, 142, 145, 152, 153,  93,  44,
        #         154, 156, 157, 123,  40, 122,  68,  89,  94,  98,  85,  84,  83,
        #         99, 102,  80,  79,  78,  74, 110,  70, 111, 119, 161,  45,  19,
        #         178,  20, 162, 182, 184,  10,   9, 201,   7,   6, 202, 203, 204,
        #           2,   1, 177, 176,  91, 175,  27,  28, 173,  31, 169, 171, 167,
        #         168,  24, 
        #         107,  95, 166, 160, 196, 105, 106, 194, 109, 138, 139, 132, 129,
        #         128, 125, 108, 140, 120, 181, 174, 143, 185, 187, 180, 163,   0,
        #         46,  25,  26,  16,  66,  30,  56,  32,  51,  12,  77,  71,  11,
        #           8,  36,  62,
        #           37, 127,  38,  22, 114,  39,  96,
        #             23,  42,  41,  29,  35, 172, 165, 170, 164,  33,  34, 158, 159,
        #         90, 179,   3,   4,   5, 200, 199, 198, 197, 195, 193, 192, 191,
        #         190, 189, 188,  13, 186,  14,  15, 183,  17, 155,  21,  18, 149,
        #         47,  67, 118, 117, 116, 115, 113, 112,  69,  72,  73, 121,  75,
        #         104, 205,  81, 101, 100,  82,  86,  97,  87,  88,  76,  65,  64,
        #         124, 151, 150,  92, 148, 147, 146,  48, 144,  49,  50, 141,  52,
        #         53, 137,  58, 134, 133,  60, 131, 130,  61, 126,  63,  43, 206])
        ################################
        
        return x





class Guide_diff_metrla_vae_latent_mean(nn.Module):
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
        
        x = torch.cat([x1[:,[67,  79,  78,  74, 161, 162,  68, 136,  66,  65,  64, 167, 168,
               169, 171, 173, 176,  45,  80,  44,  85,  90, 132, 123, 122, 119,
               118, 117, 116, 145, 110, 108, 107, 104, 205,  99,  97,  92, 153,
                89, 177, 135,  40,  20, 188,  13,  24, 195, 198, 185,  28, 184,
               183, 200, 201, 203,  31, 178, 182,  34,   2,  32,   1],:],
                        x2[:,[3, 102, 101, 100,  98, 194, 150,  95, 133,  14, 131, 148, 139,
                               199, 109, 127, 140, 125, 124, 151, 112, 146,   5, 137,  18, 154,
                               164,  63,  41,  61,  60, 165,  57,  53, 170, 181, 172, 180,  47,
                               179,  39,  43, 163,  70, 206,  25,  84,  26, 156,  23, 158,  82,
                                76, 159,  73, 155,  77],:],
                        x3[:,[192, 138, 175, 190, 152, 141, 186, 142, 204, 166, 202, 143, 147,
                               149, 157, 189,   0, 130,  58,  55,  54,  52, 134,  50,  49,  46,
                                42,  37,  36,  59,  35,  30,  27,  21,  19,  17,  11,  10,   9,
                                 8,   7,   6,  33,  62, 103,  71, 129, 128, 126, 121, 120, 115,
                               114, 113,  69, 106, 105,  96,  94, 111,  88,  72,  83,  87,  86,
                                93],:],
                        x4[:,[38,   4, 160,  75,  81, 197,  12,  15, 193, 191, 144, 187,  22,
                               174,  29,  48,  51],:],
                        x5[:,[196,  56,  91,  16],:]
                        ],dim=1)
        x = x[:,[136,  62,  60,  63, 187,  85, 161, 160, 159, 158, 157, 156, 192,
                45,  72, 193, 206, 155,  87, 154,  43, 153, 198, 112,  46, 108,
               110, 152,  50, 200, 151,  56,  61, 162,  59, 150, 148, 147, 186,
               103,  42,  91, 146, 104,  19,  17, 145, 101, 201, 144, 143, 202,
               141,  96, 140, 139, 204,  95, 138, 149,  93,  92, 163,  90,  10,
                 9,   8,   0,   6, 174, 106, 165, 181, 117,   3, 189, 115, 119,
                 2,   1,  18, 190, 114, 182, 109,  20, 184, 183, 180,  39,  21,
               205,  37, 185, 178,  70, 177,  36,  67,  35,  66,  65,  64, 164,
                33, 176, 175,  32,  31,  77,  30, 179,  83, 173, 172, 171,  28,
                27,  26,  25, 170, 169,  24,  23,  81,  80, 168,  78, 167, 166,
               137,  73,  22,  71, 142,  41,   7,  86, 121,  75,  79, 125, 127,
               131, 196,  29,  84, 132,  74, 133,  69,  82, 124,  38,  88, 118,
               111, 134, 113, 116, 188,   4,   5, 105,  89,  94, 129,  11,  12,
                13,  97,  14,  99,  15, 199, 122,  16,  40,  57, 102, 100,  98,
                58,  52,  51,  49, 126, 197,  44, 135, 123, 195, 120, 194,  68,
                47, 203, 191,  48,  76,  53,  54, 130,  55, 128,  34, 107],:]
        
        
        
        # np.argsort([67,  79,  78,  74, 161, 162,  68, 136,  66,  65,  64, 167, 168,
               # 169, 171, 173, 176,  45,  80,  44,  85,  90, 132, 123, 122, 119,
               # 118, 117, 116, 145, 110, 108, 107, 104, 205,  99,  97,  92, 153,
               #  89, 177, 135,  40,  20, 188,  13,  24, 195, 198, 185,  28, 184,
               # 183, 200, 201, 203,  31, 178, 182,  34,   2,  32,   1, 
               # 3, 102, 101, 100,  98, 194, 150,  95, 133,  14, 131, 148, 139,
               # 199, 109, 127, 140, 125, 124, 151, 112, 146,   5, 137,  18, 154,
               # 164,  63,  41,  61,  60, 165,  57,  53, 170, 181, 172, 180,  47,
               # 179,  39,  43, 163,  70, 206,  25,  84,  26, 156,  23, 158,  82,
               #  76, 159,  73, 155,  77, 
               #  192, 138, 175, 190, 152, 141, 186, 142, 204, 166, 202, 143, 147,
               # 149, 157, 189,   0, 130,  58,  55,  54,  52, 134,  50,  49,  46,
               #  42,  37,  36,  59,  35,  30,  27,  21,  19,  17,  11,  10,   9,
               #   8,   7,   6,  33,  62, 103,  71, 129, 128, 126, 121, 120, 115,
               # 114, 113,  69, 106, 105,  96,  94, 111,  88,  72,  83,  87,  86,
               #  93, 
               #  38,   4, 160,  75,  81, 197,  12,  15, 193, 191, 144, 187,  22,
               # 174,  29,  48,  51, 
               # 196,  56,  91,  16])
        ################################
        
        return x




class Guide_diff_metrla_vae_latent_corr(nn.Module):
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
        
        x = torch.cat([x1[:,[0, 108, 110, 111, 114, 115, 116, 107, 117, 119, 122, 123, 128,
               129, 132, 118, 135, 106, 205,  79,  80,  83,  85,  86,  87, 104,
                88,  90,  92,  93,  94,  97,  99,  89, 136, 138, 142, 183, 184,
               185, 186, 188, 189, 182, 190, 198, 200, 201, 202, 203, 204, 195,
               178, 177, 176, 145, 147, 149, 152, 153, 154, 157, 161, 162, 167,
               168, 169, 171, 173, 175,  78,  74, 103,  49,  13,  55,  54,  19,
                20,  46,  45,  44,  42,  24,  40,  39,  37,  36,  35,  27,  34,
                28,  30,  33,  32,  11,  10,  31,   9,  66,  57,  65,  64,   7,
                67,  68,  70,   6,  59,  71,   8,   2,   1,  58],:],
                        x2[:,[151, 163,  25,  26, 155, 160, 159, 199,   5,   4, 156,   3, 158,
                                23, 165, 187,  12, 191,  14,  15, 181, 180, 179,  17, 150,  18,
                               192, 174, 193, 194, 172,  21, 170, 166, 164,  73, 139, 146, 148,
                               109,  51,  52,  53, 105, 102, 101, 100,  98,  50,  96,  60,  61,
                                62,  63,  84,  82,  81,  69,  77,  76,  95, 112, 206,  48, 143,
                               141, 140,  72, 137, 134, 133, 113, 131, 130,  38, 126, 125, 124,
                                41, 127, 121, 120,  43,  47],:],
                        x3[:,[56,  75, 144,  29, 197,  22,  91, 196,  16],:]
                        ],dim=1)
        x = x[:,[ 0, 112, 111, 125, 123, 122, 107, 103, 110,  98,  96,  95, 130,
                74, 132, 133, 206, 137, 139,  77,  78, 145, 203, 127,  83, 116,
               117,  89,  91, 201,  92,  97,  94,  93,  90,  88,  87,  86, 188,
                85,  84, 192,  82, 196,  81,  80,  79, 197, 177,  73, 162, 154,
               155, 156,  76,  75, 198, 100, 113, 108, 164, 165, 166, 167, 102,
               101,  99, 104, 105, 171, 106, 109, 181, 149,  71, 199, 173, 172,
                70,  19,  20, 170, 169,  21, 168,  22,  23,  24,  26,  33,  27,
               204,  28,  29,  30, 174, 163,  31, 161,  32, 160, 159, 158,  72,
                25, 157,  17,   7,   1, 153,   2,   3, 175, 185,   4,   5,   6,
                 8,  15,   9, 195, 194,  10,  11, 191, 190, 189, 193,  12,  13,
               187, 186,  14, 184, 183,  16,  34, 182,  35, 150, 180, 179,  36,
               178, 200,  55, 151,  56, 152,  57, 138, 114,  58,  59,  60, 118,
               124,  61, 126, 120, 119,  62,  63, 115, 148, 128, 147,  64,  65,
                66, 146,  67, 144,  68, 141,  69,  54,  53,  52, 136, 135, 134,
                43,  37,  38,  39,  40, 129,  41,  42,  44, 131, 140, 142, 143,
                51, 205, 202,  45, 121,  46,  47,  48,  49,  50,  18, 176],:]
        
        
        
        # np.argsort([0, 108, 110, 111, 114, 115, 116, 107, 117, 119, 122, 123, 128,
               # 129, 132, 118, 135, 106, 205,  79,  80,  83,  85,  86,  87, 104,
               #  88,  90,  92,  93,  94,  97,  99,  89, 136, 138, 142, 183, 184,
               # 185, 186, 188, 189, 182, 190, 198, 200, 201, 202, 203, 204, 195,
               # 178, 177, 176, 145, 147, 149, 152, 153, 154, 157, 161, 162, 167,
               # 168, 169, 171, 173, 175,  78,  74, 103,  49,  13,  55,  54,  19,
               #  20,  46,  45,  44,  42,  24,  40,  39,  37,  36,  35,  27,  34,
               #  28,  30,  33,  32,  11,  10,  31,   9,  66,  57,  65,  64,   7,
               #  67,  68,  70,   6,  59,  71,   8,   2,   1,  58, 
               #  151, 163,  25,  26, 155, 160, 159, 199,   5,   4, 156,   3, 158,
               #  23, 165, 187,  12, 191,  14,  15, 181, 180, 179,  17, 150,  18,
               # 192, 174, 193, 194, 172,  21, 170, 166, 164,  73, 139, 146, 148,
               # 109,  51,  52,  53, 105, 102, 101, 100,  98,  50,  96,  60,  61,
               #  62,  63,  84,  82,  81,  69,  77,  76,  95, 112, 206,  48, 143,
               # 141, 140,  72, 137, 134, 133, 113, 131, 130,  38, 126, 125, 124,
               #  41, 127, 121, 120,  43,  47, 
               #  56,  75, 144,  29, 197,  22,  91, 196,  16])
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
        
        x = torch.cat([x1[:,[2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],:],
                       x2[:,[0,  1, 22, 23, 24, 26, 27, 28, 29, 31],:],
                       x3[:,[25, 30, 32, 33, 34, 35],:]],dim=1)
        x = x[:,[20, 21,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                15, 16, 17, 18, 19, 22, 23, 24, 30, 25, 26, 27, 28, 31, 29, 32, 33,
                34, 35],:]
        
        #by np.argsort([2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                # 19, 20, 21,0,  1, 22, 23, 24, 26, 27, 28, 29, 31,25, 30, 32, 33, 34, 35])
        ###################################
        
        return x


    
    
    
    
    
class Guide_diff_pm25_vae_error(nn.Module):
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
        
        
        
        #########################
        x = torch.cat([x1[:,[17, 21, 19, 16, 15, 14, 13, 12, 10, 11,  8,  7,  5,  4,  3,  9],:],
                       x2[:,[26, 30, 29, 28, 31, 27, 32, 25,  0, 23, 22,  6,  2,  1, 24],:],
                       x3[:,[20, 18, 34, 33, 35],:]],dim=1)
        x = x[:,[24, 29, 28, 14, 13, 12, 27, 11, 10, 15,  8,  9,  7,  6,  5,  4,  3,
                0, 32,  2, 31,  1, 26, 25, 30, 23, 16, 21, 19, 18, 17, 20, 22, 34,
               33, 35],:]
        
        # np.argsort([17, 21, 19, 16, 15, 14, 13, 12, 10, 11,  8,  7,  5,  4,  3,  9,26, 30, 29, 28, 31, 27, 32, 25,  0, 23, 22,  6,  2,  1, 24,20, 18, 34, 33, 35])
        #########################
        
        return x






    
class Guide_diff_pm25_vae_latent_mean(nn.Module):
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
        
        
        
        #########################
        x = torch.cat([x1[:,[0, 22, 21, 20, 19, 15, 13, 12, 11, 14,  8,  7,  6,  5,  9,  3,  2],:],
                       x2[:,[32, 26, 25, 24, 23, 28, 29, 30, 31, 27,  1],:],
                       x3[:,[33, 17, 34, 16, 10,  4, 18, 35],:]],dim=1)
        x = x[:,[0, 27, 16, 15, 33, 13, 12, 11, 10, 14, 32,  8,  7,  6,  9,  5, 31,
               29, 34,  4,  3,  2,  1, 21, 20, 19, 18, 26, 22, 23, 24, 25, 17, 28,
               30, 35],:]
        
        # np.argsort([0, 22, 21, 20, 19, 15, 13, 12, 11, 14,  8,  7,  6,  5,  9,  3,  2,32, 26, 25, 24, 23, 28, 29, 30, 31, 27,  1,33, 17, 34, 16, 10,  4, 18, 35])
        #########################
        
        return x






class Guide_diff_pm25_vae_latent_corr(nn.Module):
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
        
        skip4 = []
        for i in range(len(self.residual_layers4)):
            x, skip_connection = self.residual_layers4[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip4.append(skip_connection)
        x4 = torch.sum(torch.stack(skip4), dim=0) / math.sqrt(len(self.residual_layers4))
        x4 = x4.reshape(B, self.channels, K * L)
        x4 = self.output_projection1_4(x4)  # (B,channel,K*L)
        x4 = F.relu(x4)
        x4 = self.output_projection2_4(x4)  # (B,1,K*L)
        x4 = x4.reshape(B, K, L)
        
        #########################
        x = torch.cat([x1[:,[13, 15, 14, 21, 12, 11, 10,  9,  8,  7,  6,  4,  3, 20, 19],:],
                       x2[:,[22, 23, 27,  0, 29, 30, 25, 31, 32, 28, 26],:],
                       x3[:,[33, 17, 18, 34, 16, 35],:],
                       x4[:,[5,  2,  1, 24],:],
                       ],dim=1)
        x = x[:,[18, 34, 33, 12, 11, 32, 10,  9,  8,  7,  6,  5,  4,  0,  2,  1, 30,
               27, 28, 14, 13,  3, 15, 16, 35, 21, 25, 17, 24, 19, 20, 22, 23, 26,
               29, 31],:]
        
        # np.argsort([13, 15, 14, 21, 12, 11, 10,  9,  8,  7,  6,  4,  3, 20, 19,22, 23, 27,  0, 29, 30, 25, 31, 32, 28, 26,33, 17, 18, 34, 16, 35,5,  2,  1, 24])
        #########################
        
        return x
    
    
    
    
    
    
    
    





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
    
    
    
    
    
    

class Guide_diff_pemsbay_vae_error(nn.Module):
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
        
        
        ############################
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
        x4 = self.output_projection2_4(x4)  # (B,1,K*L)
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
        
        x = torch.cat([x1[:,[0, 139, 142, 143, 149, 157, 159, 161, 171, 172, 183, 184, 186,
               187, 188, 189, 137, 190, 132, 129,  87,  90,  92,  93,  94,  98,
                99, 104, 105, 107, 110, 112, 118, 124, 128, 131,  78, 193, 214,
               271, 272, 280, 286, 288, 289, 290, 294, 296, 297, 298, 312, 314,
               315, 316, 270, 194, 269, 262, 216, 218, 220, 228, 230, 237, 238,
               239, 241, 246, 253, 257, 258, 259, 260, 264,  77, 162, 324,  67,
                66,  40,  28,  27,  44,  52,  29,  62,  45,  47,  59,  58,  57,
                21,  54,   5,  36,  25,  69,  34,  35,  30,  14,  71,  70],:],
                x2[:,[160, 212, 255, 158, 156, 213,  23, 202, 153, 152, 215, 150,  50,
                        39, 265, 267, 211, 154, 323, 165, 204, 185, 224,  26, 181, 242,
                       206, 247, 207, 209, 250, 222,  46, 251, 168,  24, 198, 163, 166,
                        19,  63,  64, 140,  95,  68,   4, 115,  89, 291,  10,  88, 310,
                        86, 121, 282,  13, 313, 100, 320,  53, 136, 273, 138, 322, 134,
                        82, 130,   2, 318, 321],:],
                x3[:,[236, 235, 311, 233,   3, 229, 223, 227, 226, 225, 317,   1, 232,
                        31,  15, 306,  17, 221,  20, 274, 276, 277, 278, 256, 279, 281,
                       285, 254,  11, 309,   9,   8,   7,   6, 249, 248, 299, 300, 244,
                       243, 301, 302, 240, 303, 252,  32, 200,  33, 114, 116,  60, 119,
                       120, 122, 123, 113, 126,  56,  55, 135, 219,  51, 144, 145, 127,
                       147,  61, 109,  73,  79,  80,  81,  83,  84,  85, 111,  72,  96,
                        97, 101, 102, 103, 106, 108,  91, 151, 141, 196, 179, 178,  43,
                        42,  41,  38, 191, 192, 176, 175,  37, 173, 195, 182, 180, 169,
                       167, 199,  75,  48, 201,  49, 203, 205, 208, 210, 155, 217, 197],:],
                x4[:,[234, 148, 292,  76, 146, 133, 174],:],
                x5[:,[74, 261, 263, 319,  18, 266, 268,  16, 164, 275, 231, 170, 293,
                       307, 305, 304, 125,  65, 177, 283, 284, 245,  12, 287, 295, 117,
                        22, 308],:]],dim=1)
        x = x[:,[0, 184, 170, 177, 148,  93, 206, 205, 204, 203, 152, 201, 319,
               158, 100, 187, 304, 189, 301, 142, 191,  91, 323, 109, 138,  95,
               126,  81,  80,  84,  99, 186, 218, 220,  97,  98,  94, 271, 266,
               116,  79, 265, 264, 263,  82,  86, 135,  87, 280, 282, 115, 234,
                83, 162,  92, 231, 230,  90,  89,  88, 223, 239,  85, 143, 144,
               314,  78,  77, 147,  96, 102, 101, 249, 241, 297, 279, 293,  74,
                36, 242, 243, 244, 168, 245, 246, 247, 155,  20, 153, 150,  21,
               257,  22,  23,  24, 146, 250, 251,  25,  26, 160, 252, 253, 254,
                27,  28, 255,  29, 256, 240,  30, 248,  31, 228, 221, 149, 222,
               322,  32, 224, 225, 156, 226, 227,  33, 313, 229, 237,  34,  19,
               169,  35,  18, 295, 167, 232, 163,  16, 165,   1, 145, 259,   2,
                 3, 235, 236, 294, 238, 291,   4, 114, 258, 112, 111, 120, 287,
               107,   5, 106,   6, 103,   7,  75, 140, 305, 122, 141, 277, 137,
               276, 308,   8,   9, 272, 296, 270, 269, 315, 262, 261, 275, 127,
               274,  10,  11, 124,  12,  13,  14,  15,  17, 267, 268,  37,  55,
               273, 260, 289, 139, 278, 219, 281, 110, 283, 123, 284, 129, 131,
               285, 132, 286, 119, 104, 108,  38, 113,  58, 288,  59, 233,  60,
               190, 134, 179, 125, 182, 181, 180,  61, 178,  62, 307, 185, 176,
               290, 174, 173,  63,  64,  65, 215,  66, 128, 212, 211, 318,  67,
               130, 208, 207, 133, 136, 217,  68, 200, 105, 196,  69,  70,  71,
                72, 298,  57, 299,  73, 117, 302, 118, 303,  56,  54,  39,  40,
               164, 192, 306, 193, 194, 195, 197,  41, 198, 157, 316, 317, 199,
                42, 320,  43,  44,  45, 151, 292, 309,  46, 321,  47,  48,  49,
               209, 210, 213, 214, 216, 312, 311, 188, 310, 324, 202, 154, 175,
                50, 159,  51,  52,  53, 183, 171, 300, 161, 172, 166, 121,  76],:]
        ###################################
        
        # np.argsort([0, 139, 142, 143, 149, 157, 159, 161, 171, 172, 183, 184, 186,
        #        187, 188, 189, 137, 190, 132, 129,  87,  90,  92,  93,  94,  98,
        #         99, 104, 105, 107, 110, 112, 118, 124, 128, 131,  78, 193, 214,
        #        271, 272, 280, 286, 288, 289, 290, 294, 296, 297, 298, 312, 314,
        #        315, 316, 270, 194, 269, 262, 216, 218, 220, 228, 230, 237, 238,
        #        239, 241, 246, 253, 257, 258, 259, 260, 264,  77, 162, 324,  67,
        #         66,  40,  28,  27,  44,  52,  29,  62,  45,  47,  59,  58,  57,
        #         21,  54,   5,  36,  25,  69,  34,  35,  30,  14,  71,  70, 
        #         160, 212, 255, 158, 156, 213,  23, 202, 153, 152, 215, 150,  50,
        #         39, 265, 267, 211, 154, 323, 165, 204, 185, 224,  26, 181, 242,
        #        206, 247, 207, 209, 250, 222,  46, 251, 168,  24, 198, 163, 166,
        #         19,  63,  64, 140,  95,  68,   4, 115,  89, 291,  10,  88, 310,
        #         86, 121, 282,  13, 313, 100, 320,  53, 136, 273, 138, 322, 134,
        #         82, 130,   2, 318, 321,
        #         236, 235, 311, 233,   3, 229, 223, 227, 226, 225, 317,   1, 232,
        #         31,  15, 306,  17, 221,  20, 274, 276, 277, 278, 256, 279, 281,
        #        285, 254,  11, 309,   9,   8,   7,   6, 249, 248, 299, 300, 244,
        #        243, 301, 302, 240, 303, 252,  32, 200,  33, 114, 116,  60, 119,
        #        120, 122, 123, 113, 126,  56,  55, 135, 219,  51, 144, 145, 127,
        #        147,  61, 109,  73,  79,  80,  81,  83,  84,  85, 111,  72,  96,
        #         97, 101, 102, 103, 106, 108,  91, 151, 141, 196, 179, 178,  43,
        #         42,  41,  38, 191, 192, 176, 175,  37, 173, 195, 182, 180, 169,
        #        167, 199,  75,  48, 201,  49, 203, 205, 208, 210, 155, 217, 197,
        #        234, 148, 292,  76, 146, 133, 174,
        #             74, 261, 263, 319,  18, 266, 268,  16, 164, 275, 231, 170, 293,
        #        307, 305, 304, 125,  65, 177, 283, 284, 245,  12, 287, 295, 117,
        #         22, 308])
        return x
    








    

class Guide_diff_pemsbay_vae_latent_mean(nn.Module):
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
        
        
        ############################
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
        x4 = self.output_projection2_4(x4)  # (B,1,K*L)
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
        
        x = torch.cat([x1[:,[324,  86, 185, 186, 187, 188, 191,  77, 198,  75, 199, 206, 207,
               209,  70, 211, 213, 137, 220, 222,  63, 234,  58,  56, 184,  89,
               180, 178, 130, 129, 139, 141, 124, 123, 122, 118, 153, 155, 157,
               239, 112, 110, 109, 108, 105, 103, 102, 166, 100, 168, 170,  95,
               160, 240, 216,  52, 269, 270, 271,  25, 272, 280, 286, 287, 288,
               291, 296, 311,  13,  12,  10, 314,   7, 316,   5, 317, 322, 265,
               264, 136,  51,  34, 248, 246,  45,  36,  50, 255, 257,  46,  40],:],
                x2[:,[190, 312, 156, 228, 154, 315, 150, 148, 143, 233, 245, 142, 237,
                       140, 319, 321, 230, 309, 164, 323, 189, 263, 193, 262, 266, 267,
                       181, 261, 260, 306, 179, 242, 175, 173, 172, 258, 212, 253, 165,
                       298, 259, 177,   0,  39,  92,  14,  33,  30, 107, 111,  27,  24,
                        44, 117,  62, 119,  78,  47,  65, 125,  76,  18,  22,  16],:],
                x3[:,[226, 247,  69,  49,  54, 215, 217, 238,  57,  11, 219,  59, 232,
                        61, 221, 223, 227, 224, 236, 313, 229, 252, 301, 300, 299, 297,
                       294,  15, 290, 289,  20, 284, 283, 282, 250, 281, 277, 273,  28,
                       268,  29,  31,  32,  37,  71,  41, 256, 254, 278,  73, 218, 204,
                        79, 120,   8,  84,  85,  87, 115, 192, 114,  93, 158,  94, 174,
                       171, 101,   4,  91, 147,   3, 195, 131, 202, 128, 127, 194, 144,
                       200,  99, 196, 197, 146],:],
                x4[:,[1, 167, 132,  21, 318, 106, 163,   2, 161, 285, 159, 145, 126,
                        19, 149, 275, 276, 151, 116, 279,  23,  35, 113, 274,  97,  98,
                        38,  81,  80, 225,  64,  74, 305, 201, 214,  67,  68, 210, 203,
                       208,  72,  83,  60, 231, 182,  42,  43, 205,  96, 251, 249, 176,
                       183, 304,  48, 244, 243,  53, 241,  90,   6],:],
                x5[:,[9, 308, 307, 310, 320, 134, 302, 135, 138, 121, 152, 104, 169,
                        88,  82,  66, 133, 235,  55,  26, 292, 293, 295,  17, 303, 162],:]],dim=1)
        
        x = x[:,[133, 238, 245, 225, 222,  74, 298,  72, 209, 299,  70, 164,  69,
        68, 136, 182, 154, 322, 152, 251, 185, 241, 153, 258, 142,  59,
       318, 141, 193, 195, 138, 196, 197, 137,  81, 259,  85, 198, 264,
       134,  90, 200, 283, 284, 143,  84,  89, 148, 292, 158,  86,  80,
        55, 295, 159, 317,  23, 163,  22, 166, 280, 168, 145,  20, 268,
       149, 314, 273, 274, 157,  14, 199, 278, 204, 269,   9, 151,   7,
       147, 207, 266, 265, 313, 279, 210, 211,   1, 212, 312,  25, 297,
       223, 135, 216, 218,  51, 286, 262, 263, 234,  48, 221,  46,  45,
       310,  44, 243, 139,  43,  42,  41, 140,  40, 260, 215, 213, 256,
       144,  35, 146, 208, 308,  34,  33,  32, 150, 250, 230, 229,  29,
        28, 227, 240, 315, 304, 306,  79,  17, 307,  30, 104,  31, 102,
        99, 232, 249, 237, 224,  98, 252,  97, 255, 309,  36,  95,  37,
        93,  38, 217, 248,  52, 246, 324, 244, 109, 129,  47, 239,  49,
       311,  50, 220, 125, 124, 219, 123, 289, 132,  27, 121,  26, 117,
       282, 290,  24,   2,   3,   4,   5, 111,  91,   6, 214, 113, 231,
       226, 235, 236,   8,  10, 233, 271, 228, 276, 206, 285,  11,  12,
       277,  13, 275,  15, 127,  16, 272, 160,  54, 161, 205, 165,  18,
       169,  19, 170, 172, 267, 155, 171,  94, 175, 107, 281, 167, 100,
        21, 316, 173, 103, 162,  39,  53, 296, 122, 294, 293, 101,  83,
       156,  82, 288, 189, 287, 176, 128, 202,  87, 201,  88, 126, 131,
       119, 118, 114, 112,  78,  77, 115, 116, 194,  56,  57,  58,  60,
       192, 261, 253, 254, 191, 203, 257,  61, 190, 188, 187, 186, 247,
        62,  63,  64, 184, 183,  65, 319, 320, 181, 321,  66, 180, 130,
       179, 178, 177, 305, 323, 291, 270, 120, 301, 300, 108, 302,  67,
        92, 174,  71,  96,  73,  75, 242, 105, 303, 106,  76, 110,   0],:]
        ###################################
        
        # np.argsort([324,  86, 185, 186, 187, 188, 191,  77, 198,  75, 199, 206, 207,
        #        209,  70, 211, 213, 137, 220, 222,  63, 234,  58,  56, 184,  89,
        #        180, 178, 130, 129, 139, 141, 124, 123, 122, 118, 153, 155, 157,
        #        239, 112, 110, 109, 108, 105, 103, 102, 166, 100, 168, 170,  95,
        #        160, 240, 216,  52, 269, 270, 271,  25, 272, 280, 286, 287, 288,
        #        291, 296, 311,  13,  12,  10, 314,   7, 316,   5, 317, 322, 265,
        #        264, 136,  51,  34, 248, 246,  45,  36,  50, 255, 257,  46,  40,
        #        190, 312, 156, 228, 154, 315, 150, 148, 143, 233, 245, 142, 237,
        #        140, 319, 321, 230, 309, 164, 323, 189, 263, 193, 262, 266, 267,
        #        181, 261, 260, 306, 179, 242, 175, 173, 172, 258, 212, 253, 165,
        #        298, 259, 177,   0,  39,  92,  14,  33,  30, 107, 111,  27,  24,
        #         44, 117,  62, 119,  78,  47,  65, 125,  76,  18,  22,  16, 
        #         226, 247,  69,  49,  54, 215, 217, 238,  57,  11, 219,  59, 232,
        #         61, 221, 223, 227, 224, 236, 313, 229, 252, 301, 300, 299, 297,
        #        294,  15, 290, 289,  20, 284, 283, 282, 250, 281, 277, 273,  28,
        #        268,  29,  31,  32,  37,  71,  41, 256, 254, 278,  73, 218, 204,
        #         79, 120,   8,  84,  85,  87, 115, 192, 114,  93, 158,  94, 174,
        #        171, 101,   4,  91, 147,   3, 195, 131, 202, 128, 127, 194, 144,
        #        200,  99, 196, 197, 146, 
        #        1, 167, 132,  21, 318, 106, 163,   2, 161, 285, 159, 145, 126,
        #         19, 149, 275, 276, 151, 116, 279,  23,  35, 113, 274,  97,  98,
        #         38,  81,  80, 225,  64,  74, 305, 201, 214,  67,  68, 210, 203,
        #        208,  72,  83,  60, 231, 182,  42,  43, 205,  96, 251, 249, 176,
        #        183, 304,  48, 244, 243,  53, 241,  90,   6, 
        #        9, 308, 307, 310, 320, 134, 302, 135, 138, 121, 152, 104, 169,
        #         88,  82,  66, 133, 235,  55,  26, 292, 293, 295,  17, 303, 162])
        return x
    









    

class Guide_diff_pemsbay_vae_latent_corr(nn.Module):
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
        
        
        ############################
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
        x4 = self.output_projection2_4(x4)  # (B,1,K*L)
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
        
        skip6 = []
        for i in range(len(self.residual_layers6)):
            x, skip_connection = self.residual_layers6[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip6.append(skip_connection)
        x6 = torch.sum(torch.stack(skip6), dim=0) / math.sqrt(len(self.residual_layers6))
        x6 = x6.reshape(B, self.channels, K * L)
        x6 = self.output_projection1_5(x6)  # (B,channel,K*L)
        x6 = F.relu(x6)
        x6 = self.output_projection2_5(x6)  # (B,1,K*L)
        x6 = x6.reshape(B, K, L)
        
        x = torch.cat([x1[:,[324, 108, 199, 206, 100, 209,  98, 211, 212,  95, 213,  93,  91,
                            90, 198,  89, 216,  85, 220, 222,  81,  77, 236,  75, 147, 239,
                            71,  70, 242,  87, 245, 110, 197, 150, 144, 143, 153, 140, 139,
                            154, 137, 160, 164, 166, 131, 130, 111, 129, 127, 170, 123, 177,
                            181, 184, 119, 185, 117, 191, 194, 114, 113, 128,  65, 238, 148,
                            278, 280, 281,  39, 314, 287, 288,  10, 289,  11,  63,  34, 290,
                            13,  14,  30,  15,  28, 294, 296, 297,  24,  16,  22,  21, 312,
                            8,  18,  45,  62, 322, 258,  59,  58, 259, 261, 262, 263,  44,
                            317, 264,   5,  46,   7, 272, 271, 269, 270, 268, 266, 316],:],
                x2[:,[318, 313, 161, 320, 151, 152, 167, 159, 251, 310, 249, 247, 252,
                        241, 225, 224, 223, 221, 215, 214, 273, 285, 208, 207, 205, 204,
                        291, 292, 293, 301, 308, 176, 202, 162,  42,  55,  41, 101,  88,
                        50, 104,  64,  83, 133,  66,  80,  53,  20,  72, 116,   1, 145, 38],:],
                x3[:,[201, 302, 200, 303, 300, 304,  61, 106, 277,  96,  37,  73, 231,
                    240, 229, 227, 299, 226, 244, 219,  68,  26, 305, 284,  82, 195,
                    192, 307,  12, 126,   9,  49, 163,  48, 169,   6,   3, 279, 274,
                    275,   2, 276, 135, 283,  17, 235, 234, 254, 256, 282,  60, 121,
                    182, 120],:],
                x4[:,[255,  47,  51, 267, 250, 260,  54, 265, 248, 246,  52, 286,  40,
                    237, 186, 118, 309, 180, 179, 178, 122, 124, 175, 173, 125, 165,
                    315, 157, 319, 155,  76, 188,  92, 190,  33,  31, 228,  78,  29,
                    189,  27,  84,  94,  25, 298, 103, 105, 112, 306,  86],:],
                x5[:,[19, 311,   4,  32, 321,  36,   0,  56, 196, 109, 107, 172, 102,
                    97, 217, 323, 230, 232,  79, 136, 156, 141, 253, 257, 142, 233, 193],:],
                x6[:,[134, 183, 132, 158, 168, 171, 138, 174, 203, 115,  43,  23,  99,
                    210, 295, 218,  35, 149,  74,  69, 243,  67,  57, 187, 146],:]],dim=1)
        
        x = x[:,[279, 166, 209, 205, 275, 106, 204, 108,  91, 199,  72,  74, 197,
        78,  79,  81,  87, 213,  92, 273, 163,  89,  88, 311,  86, 266,
       190, 263,  82, 261,  80, 258, 276, 257,  76, 316, 278, 179, 168,
        68, 235, 152, 150, 310, 103,  93, 107, 224, 202, 200, 155, 225,
       233, 162, 229, 151, 280, 322,  98,  97, 219, 175,  94,  75, 157,
        62, 160, 321, 189, 319,  27,  26, 164, 180, 318,  23, 253,  21,
       260, 291, 161,  20, 193, 158, 264,  17, 272,  29, 154,  15,  13,
        12, 255,  11, 265,   9, 178, 286,   6, 312,   4, 153, 285, 268,
       156, 269, 176, 283,   1, 282,  31,  46, 270,  60,  59, 309, 165,
        56, 238,  54, 222, 220, 243,  50, 244, 247, 198,  48,  61,  47,
        45,  44, 302, 159, 300, 211, 292,  40, 306,  38,  37, 294, 297,
        35,  34, 167, 324,  24,  64, 317,  33, 120, 121,  36,  39, 252,
       293, 250, 303, 123,  41, 118, 149, 201,  42, 248,  43, 122, 304,
       203,  49, 305, 284, 246, 307, 245, 147,  51, 242, 241, 240,  52,
       221, 301,  53,  55, 237, 323, 254, 262, 256,  57, 195, 299,  58,
       194, 281,  32,  14,   2, 171, 169, 148, 308, 141, 140,   3, 139,
       138,   5, 313,   7,   8,  10, 135, 134,  16, 287, 315, 188,  18,
       133,  19, 132, 131, 130, 186, 184, 259, 183, 289, 181, 290, 298,
       215, 214,  22, 236,  63,  25, 182, 129,  28, 320, 187,  30, 232,
       127, 231, 126, 227, 124, 128, 295, 216, 223, 217, 296,  96,  99,
       228, 100, 101, 102, 105, 230, 114, 226, 113, 111, 112, 110, 109,
       136, 207, 208, 210, 177,  65, 206,  66,  67, 218, 212, 192, 137,
       234,  70,  71,  73,  77, 142, 143, 144,  83, 314,  84,  85, 267,
       185, 173, 145, 170, 172, 174, 191, 271, 196, 146, 239, 125, 274,
        90, 117,  69, 249, 115, 104, 116, 251, 119, 277,  95, 288,   0],:]
        ###################################
        
    #     np.argsort([324, 108, 199, 206, 100, 209,  98, 211, 212,  95, 213,  93,  91,
    #     90, 198,  89, 216,  85, 220, 222,  81,  77, 236,  75, 147, 239,
    #     71,  70, 242,  87, 245, 110, 197, 150, 144, 143, 153, 140, 139,
    #    154, 137, 160, 164, 166, 131, 130, 111, 129, 127, 170, 123, 177,
    #    181, 184, 119, 185, 117, 191, 194, 114, 113, 128,  65, 238, 148,
    #    278, 280, 281,  39, 314, 287, 288,  10, 289,  11,  63,  34, 290,
    #     13,  14,  30,  15,  28, 294, 296, 297,  24,  16,  22,  21, 312,
    #      8,  18,  45,  62, 322, 258,  59,  58, 259, 261, 262, 263,  44,
    #    317, 264,   5,  46,   7, 272, 271, 269, 270, 268, 266, 316,
    # 318, 313, 161, 320, 151, 152, 167, 159, 251, 310, 249, 247, 252,
    #    241, 225, 224, 223, 221, 215, 214, 273, 285, 208, 207, 205, 204,
    #    291, 292, 293, 301, 308, 176, 202, 162,  42,  55,  41, 101,  88,
    #     50, 104,  64,  83, 133,  66,  80,  53,  20,  72, 116,   1, 145,
    #     38,
    #     201, 302, 200, 303, 300, 304,  61, 106, 277,  96,  37,  73, 231,
    #    240, 229, 227, 299, 226, 244, 219,  68,  26, 305, 284,  82, 195,
    #    192, 307,  12, 126,   9,  49, 163,  48, 169,   6,   3, 279, 274,
    #    275,   2, 276, 135, 283,  17, 235, 234, 254, 256, 282,  60, 121,
    #    182, 120,
    #    255,  47,  51, 267, 250, 260,  54, 265, 248, 246,  52, 286,  40,
    #    237, 186, 118, 309, 180, 179, 178, 122, 124, 175, 173, 125, 165,
    #    315, 157, 319, 155,  76, 188,  92, 190,  33,  31, 228,  78,  29,
    #    189,  27,  84,  94,  25, 298, 103, 105, 112, 306,  86,
    #    19, 311,   4,  32, 321,  36,   0,  56, 196, 109, 107, 172, 102,
    #     97, 217, 323, 230, 232,  79, 136, 156, 141, 253, 257, 142, 233,
    #    193,134, 183, 132, 158, 168, 171, 138, 174, 203, 115,  43,  23,  99,
    #    210, 295, 218,  35, 149,  74,  69, 243,  67,  57, 187, 146])
        return x
    
