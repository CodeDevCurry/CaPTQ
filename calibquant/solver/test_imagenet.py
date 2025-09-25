import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from calibquant.quantization.util_log import *
import argparse
import imagenet_utils
import torch

# python ./brecq/solver/test_imagenet.py --config ./exp/w4a4/rs101/config.yaml --quantized-model-path resnet101_w4_a4_13027.pth

parser = argparse.ArgumentParser(description='Calibquant configuration',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', default='config.yaml', type=str)
parser.add_argument('--quantized-model-path', default='resnet101_w4_a4_13027.pth', type=str,
                    help='quantized model path')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 获取 config 配置
config = imagenet_utils.parse_config(args.config)

output_path = get_ckpt_path_test(config)
set_util_logging(output_path + "/calibquant.log")
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(output_path + "/calibquant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(output_path)
logger.info(args)
logger.info(config)

# Model
logger.info('==> Building quantized model..')
path = "model_quantized"
pathname = "{}_{}_w{}_a{}".format(
    config.model.type, config.data.type, config.quant.w_qconfig.bit, config.quant.a_qconfig.bit)
path = os.path.join(path, pathname)
quantized_model_path = os.path.join(path, args.quantized_model_path)
logger.info("Pretranied model ckpt: {}".format(quantized_model_path))
if not os.path.isdir(path):
    logger.info("It is not the file: {}".format(quantized_model_path))
# Load validation data
logger.info('==> Preparing data..')
imagenet_utils.set_seed(config.process.seed)
train_loader, val_loader = imagenet_utils.load_data(**config.data)
quantized_model = torch.load(quantized_model_path)

quantized_model.to(device)

logger.info('==> Validate Accuracy..')
imagenet_utils.validate_model(val_loader, quantized_model, device)
