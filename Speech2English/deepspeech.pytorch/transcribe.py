import hydra
from hydra.core.config_store import ConfigStore

from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.inference import transcribe

cs = ConfigStore.instance()
cs.store(name="config", node=TranscribeConfig)

import ntpath
import os

@hydra.main(config_name="config")
def hydra_main(cfg: TranscribeConfig):
    
    results = transcribe(cfg=cfg)
    transcription = results['output'][0]['transcription']
    print(transcription)
    
    input_filename = ntpath.basename(cfg.audio_path)
    output_filename = input_filename + '.txt'
    output_dir = '../../../../output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(os.path.join(output_dir, output_filename), "w") as f:
        f.write(transcription)
        
    
if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    hydra_main()
