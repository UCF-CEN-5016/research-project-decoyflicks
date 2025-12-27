import logging
import torch
import torch.distributed as dist

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

class Logger():
    def __init__(self, cuda=False):
        self.logger = logging.getLogger(__name__)
        self.cuda = cuda

    def info(self, message, *args, **kwargs):
        if (self.cuda and dist.get_rank() == 0) or not self.cuda:
            self.logger.info(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)

def main():
    dist.init_process_group(backend='nccl', init_method='env://')
    logger = Logger(cuda=True)
    logger.info("Starting training...")
    
    # Simulating a potential bug with GPU mapping
    if dist.get_rank() == 0:
        tensor = torch.cuda.FloatTensor([1.0, 2.0, 3.0])
    else:
        tensor = None

    # This will raise an AttributeError if tensor is None
    tensor.all_reduce()

if __name__ == "__main__":
    main()