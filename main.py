import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.config_loader import load_config

def main():
    parser = argparse.ArgumentParser(description="GraphEval-GT: End-to-End Pipeline")
    parser.add_argument('action', choices=['preprocess', 'train', 'evaluate'], 
                        help="The action to perform.")
    parser.add_argument('--dataset', type=str, default='iclr_papers', 
                        choices=['iclr_papers', 'ai_researcher'],
                        help="The dataset to process or use.")
    
    args = parser.parse_args()
    
    print(f"Executing action: '{args.action}' on dataset: '{args.dataset}'")
    
    config = load_config()
    
    if args.action == 'preprocess':
        print("This action is now manual. Please run `scripts/01_extract_all_viewpoints.py` for API calls,")
        print("and then `python -m src.data_preprocessor` to build graphs from intermediate files.")

    elif args.action == 'train':
        from src.trainer import run_training
        config['processed_data_paths']['iclr_papers'] = config['processed_data_paths'][args.dataset]
        run_training(config)
        
    elif args.action == 'evaluate':
        from src.evaluator import run_validation
        config['processed_data_paths']['iclr_papers'] = config['processed_data_paths'][args.dataset]
        run_validation(config)

if __name__ == '__main__':
    main()