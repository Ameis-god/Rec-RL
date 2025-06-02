# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import argparse
import numpy as np
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Compare large and small datasets for inclusion and sequence similarity")
    parser.add_argument(
        "--large_datamaps_file", type=str, help="Path to large dataset datamaps.json"
    )
    parser.add_argument(
        "--small_datamaps_file", type=str, help="Path to small dataset datamaps.json"
    )
    parser.add_argument(
        "--large_sequential_file", type=str, help="Path to large dataset sequential_data.txt"
    )
    parser.add_argument(
        "--small_sequential_file", type=str, help="Path to small dataset sequential_data.txt"
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to save analysis results"
    )
    args = parser.parse_args()
    return args

def load_data(args):
    # Load datamaps
    with open(args.large_datamaps_file, 'r') as f:
        large_datamaps = json.load(f)
    
    with open(args.small_datamaps_file, 'r') as f:
        small_datamaps = json.load(f)
    
    # Load sequential data
    large_user_seqs = {}
    with open(args.large_sequential_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            large_uid = parts[0]
            large_items = parts[1:]
            large_user_seqs[large_uid] = large_items
    
    small_user_seqs = {}
    with open(args.small_sequential_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            small_uid = parts[0]
            small_items = parts[1:]
            small_user_seqs[small_uid] = small_items
    
    return large_datamaps, small_datamaps, large_user_seqs, small_user_seqs

def check_user_inclusion(large_datamaps, small_datamaps):
    # Extract user mappings - use string representation for consistent comparison
    large_users_raw = set(str(user_raw) for user_raw in large_datamaps['user2id'].keys())
    
    small_users_raw = set(str(user_raw) for user_raw in small_datamaps['user2id'].keys())
    small_user_raw_to_id = {}
    for user_raw, user_id in small_datamaps['user2id'].items():
        small_user_raw_to_id[str(user_raw)] = user_id
    
    # Check inclusion
    missing_users = set()
    for user in small_users_raw:
        if user not in large_users_raw:
            missing_users.add(user)
    
    included_users = small_users_raw - missing_users
    
    # For debugging
    print(f"Small users count (unique): {len(small_users_raw)}")
    print(f"Large users count (unique): {len(large_users_raw)}")
    print(f"Included users count: {len(included_users)}")
    
    return missing_users, included_users, small_user_raw_to_id

def check_item_inclusion(large_datamaps, small_datamaps):
    # Extract item mappings - use string representation for consistent comparison
    large_items_raw = set(str(item_raw) for item_raw in large_datamaps['item2id'].keys())
    
    small_items_raw = set(str(item_raw) for item_raw in small_datamaps['item2id'].keys())
    small_item_raw_to_id = {}
    for item_raw, item_id in small_datamaps['item2id'].items():
        small_item_raw_to_id[str(item_raw)] = item_id
    
    # Check inclusion
    missing_items = set()
    for item in small_items_raw:
        if item not in large_items_raw:
            missing_items.add(item)
    
    included_items = small_items_raw - missing_items
    
    # For debugging
    print(f"Small items count (unique): {len(small_items_raw)}")
    print(f"Large items count (unique): {len(large_items_raw)}")
    print(f"Included items count: {len(included_items)}")
    
    return missing_items, included_items, small_item_raw_to_id

def analyze_sequence_similarity(large_datamaps, small_datamaps, large_user_seqs, small_user_seqs, included_users, small_user_raw_to_id):
    # Build mapping from original user ID to new ID in both datasets
    large_raw_user_to_id = {}
    for raw_user, new_id in large_datamaps['user2id'].items():
        large_raw_user_to_id[str(raw_user)] = new_id
    
    small_raw_user_to_id = {}
    for raw_user, new_id in small_datamaps['user2id'].items():
        small_raw_user_to_id[str(raw_user)] = new_id
    
    # Create item ID mapping (small dataset item ID -> original item ID -> large dataset item ID)
    small_item_to_raw = {}
    for item_id, raw_item in small_datamaps['id2item'].items():
        small_item_to_raw[item_id] = str(raw_item)
    
    large_raw_item_to_id = {}
    for raw_item, item_id in large_datamaps['item2id'].items():
        large_raw_item_to_id[str(raw_item)] = item_id
    
    # Calculate sequence similarity for each user that appears in both datasets
    similarity_scores = []
    sequence_match_details = []
    
    for i, raw_user in enumerate(included_users):
        # Get user IDs in both datasets
        small_uid = None
        large_uid = None
        
        # Try different type combinations to find the user ID
        if raw_user in small_raw_user_to_id:
            small_uid = small_raw_user_to_id[raw_user]
        elif str(raw_user) in small_raw_user_to_id:
            small_uid = small_raw_user_to_id[str(raw_user)]
        
        if raw_user in large_raw_user_to_id:
            large_uid = large_raw_user_to_id[raw_user]
        elif str(raw_user) in large_raw_user_to_id:
            large_uid = large_raw_user_to_id[str(raw_user)]
        
        if small_uid is None or large_uid is None or small_uid not in small_user_seqs or large_uid not in large_user_seqs:
            print(f"Comparing sequences for user {raw_user} (small UID: {small_uid}, large UID: {large_uid})")
            continue
        
        # Get item sequences
        small_items = small_user_seqs[small_uid]
        large_items = large_user_seqs[large_uid]

        
        # Convert small dataset item IDs to large dataset item IDs for comparison
        small_items_mapped = []
        mapping_success = True
        
        for small_item_id in small_items:
            if small_item_id in small_item_to_raw:
                raw_item = small_item_to_raw[small_item_id]
                
                # Try to find raw_item in large dataset
                large_item_id = None
                if raw_item in large_raw_item_to_id:
                    large_item_id = large_raw_item_to_id[raw_item]
                elif str(raw_item) in large_raw_item_to_id:
                    large_item_id = large_raw_item_to_id[str(raw_item)]
                
                if large_item_id is not None:
                    small_items_mapped.append(large_item_id)
                else:
                    mapping_success = False
                    break
            else:
                mapping_success = False
                break
        
        if not mapping_success:
            continue
        
        # Calculate sequence similarity
        # We'll focus on three key aspects:
        # 1. Length similarity
        # 2. Item overlap (how many items appear in both sequences)
        # 3. Order similarity (how well the order matches)

        if i == 1:
            print(f"Comparing sequences for user {raw_user} (small UID: {small_uid}, large UID: {large_uid})")
            print(f"Small items: {small_items_mapped}")
            print(f"Large items: {large_items}")
        
        # Length similarity
        len_small = len(small_items_mapped)
        len_large = len(large_items)
        length_ratio = min(len_small, len_large) / max(len_small, len_large) if max(len_small, len_large) > 0 else 0
        
        # Item overlap (using sets)
        small_items_set = set(small_items_mapped)
        large_items_set = set(large_items)
        intersection = small_items_set.intersection(large_items_set)
        union = small_items_set.union(large_items_set)
        jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0
        
        # Check if the small sequence is a subset of the large sequence
        is_subset = all(item in large_items_set for item in small_items_mapped)
        
        # Order similarity (simplified - check if small sequence appears in large sequence)
        # Convert to strings for easier substring check
        small_str = " ".join(small_items_mapped)
        large_str = " ".join(large_items)
        contains_sequence = small_str in large_str
        
        # Compute a combined similarity score (weighted average)
        combined_score = 0.3 * length_ratio + 0.5 * jaccard_similarity + 0.2 * (1 if contains_sequence else 0)
        
        similarity_scores.append(combined_score)
        sequence_match_details.append({
            "raw_user_id": raw_user,
            "small_user_id": small_uid,
            "large_user_id": large_uid,
            "small_sequence": small_items,
            "large_sequence": large_items,
            "small_seq_length": len_small,
            "large_seq_length": len_large,
            "length_ratio": length_ratio,
            "jaccard_similarity": jaccard_similarity,
            "is_subset": is_subset,
            "contains_exact_sequence": contains_sequence,
            "combined_score": combined_score
        })
    
    return similarity_scores, sequence_match_details

def extract_train_test_sets(user_seqs):
    """
    Extract training and test sets from sequence data following unirec_split.py logic
    - Test set = last item in each user's sequence
    - Training set = all items except first and last in each user's sequence
    
    Args:
        user_seqs: Dictionary mapping user IDs to their item sequences
        
    Returns:
        train_pairs: Set of (user_id, item_id) pairs for training
        test_pairs: Set of (user_id, item_id) pairs for testing
    """
    train_pairs = set()
    test_pairs = set()
    
    for user_id, items in user_seqs.items():
        if len(items) >= 3:  # Ensure we have at least 3 items (first, middle, last)
            # Last item goes to test set
            test_pairs.add((user_id, items[-1]))
            
            # All items except first and last go to training set
            for item in items[1:-1]:
                train_pairs.add((user_id, item))
    
    return train_pairs, test_pairs

def map_item_ids(item_id, dataset_id2item, other_dataset_item2id):
    """
    Map an item ID from one dataset to the corresponding ID in another dataset
    
    Args:
        item_id: Item ID to map
        dataset_id2item: Mapping from ID to raw item in source dataset
        other_dataset_item2id: Mapping from raw item to ID in target dataset
        
    Returns:
        Mapped item ID or None if mapping fails
    """
    if item_id not in dataset_id2item:
        return None
    
    raw_item = dataset_id2item[item_id]
    raw_item_str = str(raw_item)
    
    if raw_item_str in other_dataset_item2id:
        return other_dataset_item2id[raw_item_str]
    
    return None

def check_test_train_overlap(large_datamaps, small_datamaps, large_user_seqs, small_user_seqs):
    """
    Check if (user_id, item_id) pairs from small dataset test set appear in large dataset training set
    
    Args:
        large_datamaps: Datamaps for large dataset
        small_datamaps: Datamaps for small dataset
        large_user_seqs: User sequences for large dataset
        small_user_seqs: User sequences for small dataset
        
    Returns:
        overlap_results: Dictionary with overlap statistics and examples
    """
    # Extract train and test sets from both datasets
    large_train_pairs, large_test_pairs = extract_train_test_sets(large_user_seqs)
    small_train_pairs, small_test_pairs = extract_train_test_sets(small_user_seqs)
    
    print(f"Large dataset - Training pairs: {len(large_train_pairs)}, Test pairs: {len(large_test_pairs)}")
    print(f"Small dataset - Training pairs: {len(small_train_pairs)}, Test pairs: {len(small_test_pairs)}")
    
    # Create mappings for users and items
    # User mapping: small dataset user ID -> raw user ID -> large dataset user ID
    small_user_to_raw = {user_id: raw_user for raw_user, user_id in small_datamaps['user2id'].items()}
    large_raw_to_user = {str(raw_user): user_id for raw_user, user_id in large_datamaps['user2id'].items()}
    
    # Item mapping: small dataset item ID -> raw item ID -> large dataset item ID
    small_item_to_raw = {item_id: raw_item for item_id, raw_item in small_datamaps['id2item'].items()}
    large_raw_to_item = {str(raw_item): item_id for raw_item, item_id in large_datamaps['item2id'].items()}
    
    # Check for overlap
    overlap_pairs = []
    mapped_pairs_count = 0
    
    for small_user_id, small_item_id in small_test_pairs:
        # Try to map small user ID to large user ID
        small_raw_user = small_user_to_raw.get(small_user_id)
        if small_raw_user is None:
            continue
            
        large_user_id = large_raw_to_user.get(str(small_raw_user))
        if large_user_id is None:
            continue
        
        # Try to map small item ID to large item ID
        small_raw_item = small_item_to_raw.get(small_item_id)
        if small_raw_item is None:
            continue
            
        large_item_id = large_raw_to_item.get(str(small_raw_item))
        if large_item_id is None:
            continue
        
        mapped_pairs_count += 1
        
        # Check if this mapped pair exists in large training set
        if (large_user_id, large_item_id) in large_train_pairs:
            overlap_pairs.append({
                "small_user_id": small_user_id,
                "small_item_id": small_item_id,
                "large_user_id": large_user_id,
                "large_item_id": large_item_id,
                "raw_user_id": str(small_raw_user),
                "raw_item_id": str(small_raw_item)
            })
    
    # Prepare results
    overlap_results = {
        "total_small_test_pairs": len(small_test_pairs),
        "mappable_pairs": mapped_pairs_count,
        "overlap_pairs_count": len(overlap_pairs),
        "overlap_percentage_of_mappable": 100 * len(overlap_pairs) / mapped_pairs_count if mapped_pairs_count > 0 else 0,
        "overlap_percentage_of_total": 100 * len(overlap_pairs) / len(small_test_pairs) if len(small_test_pairs) > 0 else 0,
        "overlap_examples": overlap_pairs[:10]  # Show first 10 examples
    }
    
    return overlap_results

def main():
    args = parse_args()
    
    # Load data
    large_datamaps, small_datamaps, large_user_seqs, small_user_seqs = load_data(args)
    
    # Check user inclusion
    missing_users, included_users, small_user_raw_to_id = check_user_inclusion(large_datamaps, small_datamaps)
    
    # Check item inclusion
    missing_items, included_items, small_item_raw_to_id = check_item_inclusion(large_datamaps, small_datamaps)
    
    # Analyze sequence similarity
    similarity_scores, sequence_match_details = analyze_sequence_similarity(
        large_datamaps, small_datamaps, large_user_seqs, small_user_seqs, included_users, small_user_raw_to_id
    )
    
    # Check for test-train overlap - NEW FUNCTIONALITY
    overlap_results = check_test_train_overlap(
        large_datamaps, small_datamaps, large_user_seqs, small_user_seqs
    )
    
    # Get unique counts
    unique_small_users = len(set(str(u) for u in small_datamaps['user2id'].keys()))
    unique_small_items = len(set(str(i) for i in small_datamaps['item2id'].keys()))
    
    # Prepare results
    results = {
        "user_inclusion": {
            "total_small_users": unique_small_users,
            "total_large_users": len(large_datamaps['user2id']),
            "included_users_count": len(included_users),
            "missing_users_count": len(missing_users),
            "inclusion_rate": len(included_users) / unique_small_users if unique_small_users > 0 else 0,
            "missing_users_sample": list(missing_users)[:10] if missing_users else []
        },
        "item_inclusion": {
            "total_small_items": unique_small_items,
            "total_large_items": len(large_datamaps['item2id']),
            "included_items_count": len(included_items),
            "missing_items_count": len(missing_items),
            "inclusion_rate": len(included_items) / unique_small_items if unique_small_items > 0 else 0,
            "missing_items_sample": list(missing_items)[:10] if missing_items else []
        },
        "sequence_similarity": {
            "users_analyzed": len(similarity_scores),
            "average_similarity": np.mean(similarity_scores) if similarity_scores else 0,
            "median_similarity": np.median(similarity_scores) if similarity_scores else 0,
            "min_similarity": min(similarity_scores) if similarity_scores else 0,
            "max_similarity": max(similarity_scores) if similarity_scores else 0,
            "similarity_distribution": {
                "0.0-0.2": sum(1 for s in similarity_scores if 0.0 <= s < 0.2),
                "0.2-0.4": sum(1 for s in similarity_scores if 0.2 <= s < 0.4),
                "0.4-0.6": sum(1 for s in similarity_scores if 0.4 <= s < 0.6),
                "0.6-0.8": sum(1 for s in similarity_scores if 0.6 <= s < 0.8),
                "0.8-1.0": sum(1 for s in similarity_scores if 0.8 <= s <= 1.0)
            },
            "sequence_details_sample": sequence_match_details[:10] if sequence_match_details else []
        },
        # Add test-train overlap results - NEW SECTION
        "test_train_overlap": overlap_results
    }
    
    # Print summary
    print("=== Dataset Inclusion Analysis ===")
    print(f"User inclusion rate: {results['user_inclusion']['inclusion_rate']:.2%}")
    print(f"Item inclusion rate: {results['item_inclusion']['inclusion_rate']:.2%}")
    print(f"Average sequence similarity: {results['sequence_similarity']['average_similarity']:.4f}")
    print(f"Users with high similarity (>0.8): {results['sequence_similarity']['similarity_distribution']['0.8-1.0']}")
    
    # Print test-train overlap summary - NEW OUTPUT
    print("\n=== Test-Train Overlap Analysis ===")
    print(f"Small test pairs that could be mapped to large dataset: {results['test_train_overlap']['mappable_pairs']} out of {results['test_train_overlap']['total_small_test_pairs']}")
    print(f"Small test pairs found in large training set: {results['test_train_overlap']['overlap_pairs_count']}")
    print(f"Overlap percentage (of mappable pairs): {results['test_train_overlap']['overlap_percentage_of_mappable']:.2f}%")
    print(f"Overlap percentage (of total test pairs): {results['test_train_overlap']['overlap_percentage_of_total']:.2f}%")
    
    if results['test_train_overlap']['overlap_pairs_count'] > 0:
        print("\nOverlap Examples (Small Test → Large Train):")
        for i, example in enumerate(results['test_train_overlap']['overlap_examples'][:5]):
            print(f"{i+1}. Small: (User {example['small_user_id']}, Item {example['small_item_id']}) → Large: (User {example['large_user_id']}, Item {example['large_item_id']})")
    else:
        print("\nNo overlap found between small dataset test set and large dataset training set.")
    
    # Save detailed results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to {args.output_file}")

if __name__ == "__main__":
    args = parse_args()
    main()