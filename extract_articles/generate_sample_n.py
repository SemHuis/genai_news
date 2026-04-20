import argparse
import random
import sys

def generate_sample_indices(total_x, sample_y):
    random.seed(42) 

    if sample_y > total_x:
        print(f"Error: Sample size ({sample_y}) cannot be larger than total articles ({total_x}).")
        sys.exit(1)

    # Calculate the exact interval
    interval = total_x / sample_y
    
    # Pick a random starting point within the first interval
    start = random.uniform(1, interval)
    
    # Generate the list of indices
    indices = [int(start + i * interval) for i in range(sample_y)]
    
    return indices

def main():
    parser = argparse.ArgumentParser(description="Generate systematic sampling indices.")
    parser.add_argument("-n", type=int, help="Total number of articles available")
    parser.add_argument("d", type=int, help="Number of articles to download")
    args = parser.parse_args()
    
    sample_list = generate_sample_indices(args.n, args.d)
    
    print(f"Generated {len(sample_list)} indices:")
    print(sample_list)

if __name__ == "__main__":
    main()