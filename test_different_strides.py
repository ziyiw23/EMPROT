import sys
sys.path.append('.')

print("Testing different temporal strides for optimal separation...")

from emprot.utils.dataset import ProteinTrajectoryDataset
import torch

stride_values = [5, 10, 15, 25]  # 1ns, 2ns, 3ns, 5ns

results = []

for stride in stride_values:
    print(f"\n🔧 Testing stride={stride} ({stride * 0.2:.1f}ns spacing)")
    
    try:
        dataset = ProteinTrajectoryDataset(
            data_dir="/scratch/groups/rbaltman/ziyiw23/traj_embeddings/",
            metadata_path="traj_metadata.csv",
            temporal_stride=stride,
            sequence_length=5
        )
        
        # Test multiple samples to get average
        similarities = []
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            embeddings = sample['embeddings']
            
            # Check frame-to-frame similarity within sequence
            frame_sims = []
            for j in range(len(embeddings) - 1):
                frame_a = embeddings[j].flatten()
                frame_b = embeddings[j + 1].flatten()
                cos_sim = torch.dot(frame_a, frame_b) / (torch.norm(frame_a) * torch.norm(frame_b))
                frame_sims.append(cos_sim.item())
            
            avg_sim = sum(frame_sims) / len(frame_sims)
            similarities.append(avg_sim)
        
        overall_avg = sum(similarities) / len(similarities)
        
        print(f"   Average frame similarity: {overall_avg:.6f}")
        print(f"   Improvement from 0.999: {(0.999 - overall_avg) / 0.999 * 100:.2f}%")
        
        results.append({
            'stride': stride,
            'spacing_ns': stride * 0.2,
            'similarity': overall_avg,
            'improvement_pct': (0.999 - overall_avg) / 0.999 * 100
        })
        
        # Quick assessment
        if overall_avg < 0.95:
            print("   🎯 EXCELLENT: Strong temporal separation!")
        elif overall_avg < 0.98:
            print("   ✅ GOOD: Meaningful separation for training")
        else:
            print("   ⚠️  MODERATE: Some improvement but may need larger stride")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

print(f"\n📊 SUMMARY OF RESULTS:")
print("=" * 50)
for result in results:
    print(f"Stride {result['stride']:2d} ({result['spacing_ns']:3.1f}ns): "
          f"Similarity {result['similarity']:.6f} "
          f"({result['improvement_pct']:+5.2f}% improvement)")

print(f"\n🎯 RECOMMENDATIONS:")
best_result = min(results, key=lambda x: x['similarity']) if results else None
if best_result:
    print(f"   Best stride: {best_result['stride']} ({best_result['spacing_ns']:.1f}ns)")
    print(f"   Best similarity: {best_result['similarity']:.6f}")
    
    if best_result['similarity'] < 0.95:
        print("   ✅ Ready for training with strong temporal signal!")
    elif best_result['similarity'] < 0.98:
        print("   ✅ Good for training, should see learning improvements")
    else:
        print("   ⚠️  Consider even larger strides (30-50) for stronger signal")
else:
    print("   No successful tests - check data access") 

