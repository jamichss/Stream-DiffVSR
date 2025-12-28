import torch.nn as nn

class VGGPerceptualLoss(nn.Module):
    def __init__(self, vgg, feature_layers=[2, 7, 12, 21]):  
        """
        提取 VGG-19 的不同層來計算感知損失
        預設提取：
        - relu1_2 (第2層)
        - relu2_2 (第7層)
        - relu3_4 (第12層)
        - relu4_4 (第21層)
        """
        super(VGGPerceptualLoss, self).__init__()
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:max(feature_layers)+1])
        self.feature_layers = feature_layers
        for param in self.vgg_layers.parameters():
            param.requires_grad = False  # 凍結 VGG，避免影響訓練
    
    def forward(self, pred, target):
        loss = 0
        for i, layer in enumerate(self.vgg_layers):
            pred = layer(pred)
            target = layer(target)
            if i in self.feature_layers:
                loss += nn.functional.l1_loss(pred, target)  # L1 loss 更適合高層特徵
        return loss
