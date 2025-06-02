# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Convert small dataset test IDs to match large dataset IDs")
    parser.add_argument(
        "--large_datamaps_file", type=str, help="Path to large dataset datamaps.json"
    )
    parser.add_argument(
        "--small_datamaps_file", type=str, help="Path to small dataset datamaps.json"
    )
    parser.add_argument(
        "--small_test_file", type=str, help="Path to small dataset test_ids.csv"
    )
    parser.add_argument(
        "--output_test_file", type=str, help="Path to save converted test_ids.csv"
    )
    args = parser.parse_args()
    return args

def convert_test_ids(args):
    # Load datamaps
    with open(args.large_datamaps_file, 'r') as f:
        large_datamaps = json.load(f)
    
    with open(args.small_datamaps_file, 'r') as f:
        small_datamaps = json.load(f)
    
    # Print some examples to verify data types
    print("=== Verifying Data Types ===")
    print("Small datamaps - First user ID in id2user:", next(iter(small_datamaps['id2user'].items())))
    print("Large datamaps - First user ID in user2id:", next(iter(large_datamaps['user2id'].items())))
    
    # Extract mappings
    small_id2user = small_datamaps['id2user']  # small_uid -> raw_user
    small_id2item = small_datamaps['id2item']  # small_iid -> raw_item
    
    large_user2id = {}
    # Normalize keys to ensure consistent types (both string and int versions)
    for k, v in large_datamaps['user2id'].items():
        large_user2id[k] = v  # Original key (could be string)
        large_user2id[int(k)] = v  # Integer version
        large_user2id[str(k)] = v  # String version
    
    large_item2id = {}
    # Normalize keys to ensure consistent types (both string and int versions)
    for k, v in large_datamaps['item2id'].items():
        large_item2id[k] = v  # Original key (could be string)
        large_item2id[int(k)] = v  # Integer version
        large_item2id[str(k)] = v  # String version
    
    # Create direct mapping from small ID to large ID
    small2large_user = {}
    missing_users_in_large = 0
    for small_uid, raw_user in small_id2user.items():
        # Try different type combinations
        if raw_user in large_user2id:
            small2large_user[small_uid] = large_user2id[raw_user]
        elif str(raw_user) in large_user2id:
            small2large_user[small_uid] = large_user2id[str(raw_user)]
        elif int(raw_user) in large_user2id:
            small2large_user[small_uid] = large_user2id[int(raw_user)]
        else:
            missing_users_in_large += 1
            if missing_users_in_large <= 5:  # Print only first 5 missing users
                print(f"Raw user {raw_user} (type: {type(raw_user)}) from small dataset not found in large dataset")
    
    print(f"Total missing users in large dataset: {missing_users_in_large}")
    print(f"Successfully mapped users: {len(small2large_user)}")
    
    small2large_item = {}
    missing_items_in_large = 0
    for small_iid, raw_item in small_id2item.items():
        # Try different type combinations
        if raw_item in large_item2id:
            small2large_item[small_iid] = large_item2id[raw_item]
        elif str(raw_item) in large_item2id:
            small2large_item[small_iid] = large_item2id[str(raw_item)]
        elif int(raw_item) in large_item2id:
            small2large_item[small_iid] = large_item2id[int(raw_item)]
        else:
            missing_items_in_large += 1
            if missing_items_in_large <= 5:  # Print only first 5 missing items
                print(f"Raw item {raw_item} (type: {type(raw_item)}) from small dataset not found in large dataset")
    
    print(f"Total missing items in large dataset: {missing_items_in_large}")
    print(f"Successfully mapped items: {len(small2large_item)}")
    
    # Print sample of successful mappings
    print("\nSample of successful user mappings (small_uid -> raw_user -> large_uid):")
    sample_count = 0
    for small_uid, raw_user in small_id2user.items():
        if small_uid in small2large_user:
            print(f"  {small_uid} -> {raw_user} -> {small2large_user[small_uid]}")
            sample_count += 1
            if sample_count >= 5:
                break
    
    # Load small test file
    small_test_df = pd.read_csv(args.small_test_file, sep='\t')
    
    # Convert IDs
    converted_test = []
    for _, row in small_test_df.iterrows():
        small_uid = str(row['user_id'])
        small_iid = str(row['item_id'])
        
        if small_uid in small2large_user and small_iid in small2large_item:
            large_uid = int(small2large_user[small_uid])
            large_iid = int(small2large_item[small_iid])
            converted_test.append([large_uid, large_iid])
    
    # Create and save converted dataframe
    converted_df = pd.DataFrame(converted_test, columns=['user_id', 'item_id'])
    converted_df.to_csv(args.output_test_file, index=False, sep='\t')
    
    print(f"\nOriginal small test set size: {len(small_test_df)}")
    print(f"Converted test set size: {len(converted_df)}")
    print(f"Mapping rate: {len(converted_df)/len(small_test_df)*100:.2f}%")
    
    # Check for missing mappings in test set
    missing_users = 0
    missing_items = 0
    for _, row in small_test_df.iterrows():
        small_uid = str(row['user_id'])
        small_iid = str(row['item_id'])
        
        if small_uid not in small2large_user:
            missing_users += 1
        if small_iid not in small2large_item:
            missing_items += 1
    
    print(f"Missing user mappings in test set: {missing_users}")
    print(f"Missing item mappings in test set: {missing_items}")

if __name__ == '__main__':
    args = parse_args()
    convert_test_ids(args)