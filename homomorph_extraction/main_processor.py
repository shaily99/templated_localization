import argparse
import sys
from pathlib import Path
import itertools
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings import EmbeddingGenerator
from homomorphs import HomomorphismExtractor
from utils import load_responses

logging.basicConfig(level=logging.INFO)


def process_specific_nationality_pair(input_file, nat1, nat2, tokenization_path, output_dir="homomorphisms_csv", 
                                     n_columns=5, k_homomorphs=10, batch_size=32, 
                                     model_name="bert-base-uncased", include_self_matches=False):
    """
    Process a specific pair of nationalities and save homomorphisms as CSV files.
    Uses pre-computed tokenization for NPMI words.
    
    Args:
        input_file (str): Path to input dataset
        nat1 (str): First nationality (e.g., "Afghan")
        nat2 (str): Second nationality (e.g., "American") 
        tokenization_path (str): Path to pre-computed tokenization JSON file
        output_dir (str): Output directory for CSV files
        n_columns (int): Number of response columns to process
        k_homomorphs (int): Number of homomorphs per word
        batch_size (int): Batch size for processing
        model_name (str): HuggingFace model name
        include_self_matches (bool): Whether to include self-matches
    """
    # Initialize components with tokenization path
    embedding_gen = EmbeddingGenerator(model_name, tokenization_path=tokenization_path)
    homomorph_extractor = HomomorphismExtractor(k_homomorphs, include_self_matches)
    
    # Load and prepare data
    logging.info(f"Loading dataset from {input_file}...")
    dataset = load_responses(input_file)
    story_dict = embedding_gen.prepare_story_data(dataset, n_columns)
    
    # Check that both nationalities exist in the data
    if nat1 not in story_dict:
        raise ValueError(f"Nationality '{nat1}' not found in dataset. Available: {list(story_dict.keys())}")
    if nat2 not in story_dict:
        raise ValueError(f"Nationality '{nat2}' not found in dataset. Available: {list(story_dict.keys())}")
    
    # Check that both nationalities have tokenization data
    if nat1 not in embedding_gen.tokenization_data:
        raise ValueError(f"Nationality '{nat1}' not found in tokenization data. Available: {list(embedding_gen.tokenization_data.keys())}")
    if nat2 not in embedding_gen.tokenization_data:
        raise ValueError(f"Nationality '{nat2}' not found in tokenization data. Available: {list(embedding_gen.tokenization_data.keys())}")
    
    logging.info(f"Processing pair: {nat1} - {nat2}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Extract homomorphisms for this specific pair
    homomorphism_rows = homomorph_extractor.process_nationality_pair(
        embedding_gen, story_dict, nat1, nat2, batch_size)
    
    # Separate bidirectional homomorphisms
    nat1_to_nat2 = [row for row in homomorphism_rows if row['source_nationality'] == nat1]
    nat2_to_nat1 = [row for row in homomorphism_rows if row['source_nationality'] == nat2]
    
    # Save separate files for each direction
    file1 = f"{output_dir}/{nat1}_to_{nat2}_homomorphisms.csv"
    file2 = f"{output_dir}/{nat2}_to_{nat1}_homomorphisms.csv"
    
    homomorph_extractor.save_homomorphisms_to_csv(nat1_to_nat2, file1)
    homomorph_extractor.save_homomorphisms_to_csv(nat2_to_nat1, file2)
    
    return file1, file2


def process_all_nationality_pairs(input_file, tokenization_path, output_dir="homomorphisms_csv", n_columns=5,
                                 nationality_filter=None, k_homomorphs=10,
                                 batch_size=32, model_name="bert-base-uncased",
                                 include_self_matches=False):
    """
    Process all nationality pairs and save homomorphisms as CSV files.
    Uses pre-computed tokenization for NPMI words.
    
    Args:
        input_file (str): Path to input dataset
        tokenization_path (str): Path to pre-computed tokenization JSON file
        output_dir (str): Output directory for CSV files
        n_columns (int): Number of response columns to process
        nationality_filter (list): List of nationalities to filter (None for all)
        k_homomorphs (int): Number of homomorphs per word
        batch_size (int): Batch size for processing
        model_name (str): HuggingFace model name
        include_self_matches (bool): Whether to include self-matches
    """
    # Initialize components with tokenization path
    embedding_gen = EmbeddingGenerator(model_name, tokenization_path=tokenization_path)
    homomorph_extractor = HomomorphismExtractor(k_homomorphs, include_self_matches)
    
    # Load and prepare data
    logging.info(f"Loading dataset from {input_file}...")
    dataset = load_responses(input_file)
    story_dict = embedding_gen.prepare_story_data(dataset, n_columns)
    
    # Filter nationalities based on both data availability and tokenization availability
    available_nationalities = set(story_dict.keys()) & set(embedding_gen.tokenization_data.keys())
    
    if nationality_filter:
        available_nationalities = available_nationalities & set(nationality_filter)
    
    nationalities = list(available_nationalities)
    
    if len(nationalities) < 2:
        raise ValueError("Need at least 2 nationalities with both story data and tokenization data")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Process all pairs
    total_pairs = len(list(itertools.combinations(nationalities, 2)))
    logging.info(f"Processing {total_pairs} nationality pairs...")
    
    for pair_idx, (nat1, nat2) in enumerate(itertools.combinations(nationalities, 2)):
        logging.info(f"[{pair_idx+1}/{total_pairs}] Processing pair: {nat1} - {nat2}")
        
        try:
            # Extract homomorphisms for this pair
            homomorphism_rows = homomorph_extractor.process_nationality_pair(
                embedding_gen, story_dict, nat1, nat2, batch_size)
            
            # Save individual pair results
            pair_output_file = f"{output_dir}/homomorphisms_{nat1}_{nat2}.csv"
            homomorph_extractor.save_homomorphisms_to_csv(homomorphism_rows, pair_output_file)
            
        except Exception as e:
            logging.error(f"Error processing {nat1}-{nat2}: {str(e)}")
            continue
    
    logging.info("Processing complete!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract homomorphisms from NPMI words and save as CSV")
    parser.add_argument("--input_file", required=True, help="Path to input Arrow dataset")
    parser.add_argument("--tokenization_path", required=True, 
                       help="Path to pre-computed tokenization JSON file")
    parser.add_argument("--output_dir", default="homomorphisms_csv", 
                       help="Output directory for CSV files")
    parser.add_argument("--n_columns", type=int, default=5,
                       help="Number of response columns to process")
    parser.add_argument("--nat1", default=None,
                       help="First nationality for specific pair processing (e.g., 'Afghan')")
    parser.add_argument("--nat2", default=None, 
                       help="Second nationality for specific pair processing (e.g., 'American')")
    parser.add_argument("--nationalities", nargs="+", default=None,
                       help="List of nationalities to process all pairs (default: all available)")
    parser.add_argument("--k_homomorphs", type=int, default=10,
                       help="Number of top homomorphs to extract per word")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for embedding computation")
    parser.add_argument("--model_name", default="bert-base-uncased",
                       help="HuggingFace model name")
    parser.add_argument("--include_self_matches", action="store_true",
                       help="Include self-matches in results")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Validate tokenization file exists
    if not Path(args.tokenization_path).exists():
        logging.error(f"Tokenization file not found: {args.tokenization_path}")
        return
    
    # Check if specific pair processing is requested
    if args.nat1 and args.nat2:
        # Process specific nationality pair
        logging.info(f"Processing specific pair: {args.nat1} - {args.nat2}")
        
        try:
            process_specific_nationality_pair(
                input_file=args.input_file,
                nat1=args.nat1,
                nat2=args.nat2,
                tokenization_path=args.tokenization_path,
                output_dir=args.output_dir,
                n_columns=args.n_columns,
                k_homomorphs=args.k_homomorphs,
                batch_size=args.batch_size,
                model_name=args.model_name,
                include_self_matches=args.include_self_matches
            )
                
        except Exception as e:
            logging.error(f"Error in specific pair processing: {e}")
            return
            
    elif args.nat1 or args.nat2:
        logging.error("Both --nat1 and --nat2 must be specified for specific pair processing")
        return
        
    else:
        # Process all pairs or filtered nationalities
        try:
            nationality_filter = set(args.nationalities) if args.nationalities else None
            
            process_all_nationality_pairs(
                input_file=args.input_file,
                tokenization_path=args.tokenization_path,
                output_dir=args.output_dir,
                n_columns=args.n_columns,
                nationality_filter=nationality_filter,
                k_homomorphs=args.k_homomorphs,
                batch_size=args.batch_size,
                model_name=args.model_name,
                include_self_matches=args.include_self_matches
            )
                    
        except Exception as e:
            logging.error(f"Error in all pairs processing: {e}")
            return
    
    logging.info("Processing complete!")


if __name__ == "__main__":
    main()