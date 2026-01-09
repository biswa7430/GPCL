#!/usr/bin/env python3
"""
Quick test script to verify class-wise metrics implementation.
This script checks if the class-wise metrics functions are working correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_class_wise_metrics():
    """Test the class-wise metrics functionality."""
    print("="*70)
    print("Testing Class-wise Metrics Implementation")
    print("="*70)
    
    # Test 1: Check if functions exist
    print("\n1. Checking if functions exist...")
    try:
        from src.evaluation.evaluator import get_class_wise_metrics, save_class_wise_csv
        print("   ✓ get_class_wise_metrics found")
        print("   ✓ save_class_wise_csv found")
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        return False
    
    # Test 2: Test CSV export with dummy data
    print("\n2. Testing CSV export with dummy data...")
    try:
        import tempfile
        
        dummy_metrics = {
            'car': {
                'category_id': 1,
                'AP': 0.5234,
                'AP50': 0.7891,
                'AP75': 0.5678,
                'Recall': 0.6234
            },
            'pedestrian': {
                'category_id': 2,
                'AP': 0.4123,
                'AP50': 0.6789,
                'AP75': 0.4456,
                'Recall': 0.5456
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_csv = f.name
        
        save_class_wise_csv(dummy_metrics, temp_csv)
        
        # Read and verify
        with open(temp_csv, 'r') as f:
            content = f.read()
            if 'Class Name' in content and 'car' in content and 'pedestrian' in content:
                print("   ✓ CSV export successful")
                print(f"   ✓ CSV saved to: {temp_csv}")
                print("\n   CSV Content Preview:")
                print("   " + "\n   ".join(content.split('\n')[:4]))
            else:
                print("   ✗ CSV content invalid")
                return False
        
        # Cleanup
        os.remove(temp_csv)
        
    except Exception as e:
        print(f"   ✗ CSV export failed: {e}")
        return False
    
    # Test 3: Check evaluation script modifications
    print("\n3. Checking evaluation script modifications...")
    try:
        eval_script = os.path.join(os.path.dirname(__file__), 'evaluate.py')
        with open(eval_script, 'r') as f:
            content = f.read()
            
        if 'class_wise' in content:
            print("   ✓ Class-wise handling added to evaluate.py")
        else:
            print("   ✗ Class-wise handling not found in evaluate.py")
            return False
            
        if 'Class-wise Evaluation Results' in content:
            print("   ✓ Class-wise output formatting added")
        else:
            print("   ✗ Class-wise output formatting not found")
            return False
            
    except Exception as e:
        print(f"   ✗ Script check failed: {e}")
        return False
    
    # Test 4: Check documentation
    print("\n4. Checking documentation...")
    doc_file = os.path.join(os.path.dirname(__file__), '..', 'docs', 
                            'CLASS_WISE_METRICS_GUIDE.md')
    if os.path.exists(doc_file):
        print(f"   ✓ Documentation created: {doc_file}")
    else:
        print("   ✗ Documentation not found")
        return False
    
    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
    print("\nNext steps:")
    print("1. Run evaluation with: python scripts/evaluate.py --config configs/eval_config.yaml")
    print("2. Check output directory for class_wise_results.json and .csv files")
    print("3. Review docs/CLASS_WISE_METRICS_GUIDE.md for usage examples")
    print("="*70)
    
    return True


if __name__ == '__main__':
    success = test_class_wise_metrics()
    sys.exit(0 if success else 1)
