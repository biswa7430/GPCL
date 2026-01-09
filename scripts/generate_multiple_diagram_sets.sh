#!/bin/bash
# Generate multiple sets of model diagram images and save them with timestamps
# Usage: ./generate_multiple_diagram_sets.sh [number_of_sets]

NUM_SETS=${1:-3}  # Default: generate 3 sets
OUTPUT_BASE="/media/voyager/ssd2tb/ICPR/SHM_torch_vision/outputs/model_diagram_images"

echo "================================================================"
echo "Generating $NUM_SETS different sets of model diagram images"
echo "================================================================"
echo ""

for i in $(seq 1 $NUM_SETS); do
    echo "▶ Generating Set $i/$NUM_SETS..."
    echo "----------------------------------------"
    
    # Generate images
    python scripts/generate_model_diagram_images.py
    
    # Create timestamped directory for this set
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    SET_DIR="${OUTPUT_BASE}_set${i}_${TIMESTAMP}"
    
    # Copy generated images to set directory
    mkdir -p "$SET_DIR"
    cp ${OUTPUT_BASE}/*.png "$SET_DIR/"
    cp ${OUTPUT_BASE}/*.tex "$SET_DIR/"
    
    echo ""
    echo "✓ Set $i saved to: ${SET_DIR}"
    echo ""
    echo "Images in this set:"
    echo "  - IndraEye: $(basename $(ls ${SET_DIR}/1_indra_original.png))"
    echo "  - VisDrone: $(basename $(ls ${SET_DIR}/1_visdrone_original.png))"
    echo ""
    
    # Wait a second to ensure different timestamps
    sleep 1
done

echo "================================================================"
echo "✓ Generated $NUM_SETS different sets of images!"
echo "================================================================"
echo ""
echo "All sets saved in: $OUTPUT_BASE"
echo ""
echo "Review each set and choose the one with:"
echo "  - Clear visual differences between domains"
echo "  - Obvious style mixing effects"
echo "  - Good representative examples"
echo ""
echo "To generate more sets, run:"
echo "  ./generate_multiple_diagram_sets.sh 5"
echo ""
