# 提取图片中的水印

# 引入所需函数

from PIL import Image
import torch
import torchvision.transforms as transforms
from scipy.stats import binomtest

def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])


def str2msg(str):
    return [True if el == '1' else False for el in str]


msg_extractor = torch.jit.load("./input/msg/decoder/dec_48b_whit.torchscript.pt").to("cuda")
transform_imnet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 待测模型key
key = '111010110101000001010111010011010100010000100111'  # modelA key
# key = '110110001011100011001000001101000000001011001010'# modelB key
# 使用两个图片进行测试：可以与密钥匹配的modelA生成的水印图片与由其他模型modelB生成的水印图片
img_a = "output/imgs/000_val_w.png"
img_b = "output/imgs/000_val_w_difmodel.png"

# 加载图像并提取key
img = Image.open(img_b)
img = transform_imnet(img).unsqueeze(0).to("cuda")
msg = msg_extractor(img)  # b c h w -> b k
bool_msg = (msg > 0).squeeze().cpu().numpy().tolist()
print("提取key: ", msg2str(bool_msg))

# 对结果进行比较
bool_key = str2msg(key)

# 计算图像水印（key）与待测模型key的差值,
diff = [bool_msg[i] != bool_key[i] for i in range(len(bool_msg))]
bit_acc = 1 - sum(diff) / len(diff)
print("准确率（按比特位查看）: ", bit_acc)

# 对得到的结果进行二项式检验，假设该水印图片有90%的概率不为以该key为水印的模型生成的
print("二项式检验假设：该水印图片有90%的概率不为以该key为水印的模型生成的")
pval = binomtest(len(diff) - sum(diff), len(diff), 0.90, alternative='greater')
print("p值: ", pval.pvalue)
# 显著性水平
alpha = 0.05
if alpha < pval.pvalue:
    print("结果高于alpha，说明该水印图片不为以该key为水印的模型生成的。")
else:
    print("结果低于alpha，说明该水印图片是以该key为水印的模型生成的。")
