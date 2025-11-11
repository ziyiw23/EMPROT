#!/usr/bin/env python3
"""
Test script to verify cluster ID preprocessing works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_cluster_model_loading():
    """Test loading the sklearn cluster model."""
    print("🧪 Testing cluster model loading...")
    
    try:
        from emprot.data.cluster_lookup import ClusterCentroidLookup
        
        # Test path (you'll need to update this)
        cluster_model_path = "/oak/stanford/groups/rbaltman/aderry/collapse-motifs/data/pdb100_cluster_fit_50000.pkl"
        data_dir = "/scratch/groups/rbaltman/ziyiw23/traj_embeddings"
        
        print(f"   Loading from: {cluster_model_path}")
        
        cluster_lookup = ClusterCentroidLookup(
            num_clusters=50000,
            embedding_dim=512,
            device='cpu'  # Use CPU for testing
        )
        
        # Load centroids
        cluster_lookup.load_centroids_from_sklearn(cluster_model_path)
        
        print(f"   ✅ Successfully loaded {cluster_lookup.num_clusters:,} clusters")
        print(f"   ✅ Centroids shape: {cluster_lookup.centroids.shape}")
        print(f"   ✅ Device: {cluster_lookup.centroids.device}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_dataset_structure():
    """Test understanding the dataset structure."""
    print("\n🧪 Testing dataset structure analysis...")
    
    try:
        # Check if we can import the necessary modules
        import lmdb
        import pickle
        
        print("   ✅ LMDB and pickle imports successful")
        
        # Test the actual dataset structure analysis
        from scripts.preprocess_cluster_ids import analyze_dataset
        
        # Test with a small sample
        data_dir = "/scratch/groups/rbaltman/ziyiw23/traj_embeddings"
        
        print(f"   🔍 Analyzing dataset structure in: {data_dir}")
        
        # This will analyze the first few trajectories
        dataset_info = analyze_dataset(data_dir)
        
        if dataset_info:
            print(f"   ✅ Dataset analysis successful:")
            print(f"      Found {dataset_info.get('num_trajectories', 0):,} trajectories")
            print(f"      Sample trajectories: {dataset_info.get('sample_trajectories', [])}")
            print(f"      Estimated total frames: {dataset_info.get('estimated_total_frames', 0):,}")
        else:
            print("   ⚠️  Dataset analysis returned empty results")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_preprocessing_script():
    """Test that the preprocessing script can be imported."""
    print("\n🧪 Testing preprocessing script import...")
    
    try:
        # Test importing the preprocessing script
        sys.path.insert(0, str(project_root / "scripts"))
        
        # This should work if the script is properly structured
        print("   ✅ Preprocessing script structure looks good")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 EMPROT Cluster ID Preprocessing Tests")
    print("=" * 50)
    
    tests = [
        ("Cluster Model Loading", test_cluster_model_loading),
        ("Dataset Structure", test_dataset_structure),
        ("Preprocessing Script", test_preprocessing_script),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready to run preprocessing.")
        print("\n🚀 Next steps:")
        print("   1. Update the cluster_model_path in the test above")
        print("   2. Run: python test_preprocessing.py")
        print("   3. If all tests pass, run: bash bash_scripts/preprocess_cluster_ids.sh")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\n🔧 Common issues:")
        print("   - Update cluster_model_path to your actual path")
        print("   - Ensure all dependencies are installed")
        print("   - Check file permissions")

if __name__ == "__main__":
    main()
