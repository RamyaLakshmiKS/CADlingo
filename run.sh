#!/bin/bash
# CADlingo Quick Start Script
# This script helps you run the complete pipeline

set -e  # Exit on error

echo "======================================"
echo "CADlingo - Quick Start Script"
echo "======================================"
echo ""

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "Virtual environment not activated!"
    echo "Please activate it first:"
    echo "  source cad-env/bin/activate"
    exit 1
fi

# Function to show menu
show_menu() {
    echo ""
    echo "What would you like to do?"
    echo "1. Create training dataset"
    echo "2. Train the model"
    echo "3. Run the Streamlit UI"
    echo "4. Generate sample predictions"
    echo "5. Run training notebook"
    echo "6. View results"
    echo "0. Exit"
    echo ""
    read -p "Enter your choice: " choice
}

# Function to create dataset
create_dataset() {
    echo ""
    echo "======================================"
    echo "Creating Training Dataset"
    echo "======================================"
    cd src/data
    python dataset_creator.py
    cd ../..
    echo "Dataset created!"
}

# Function to train model
train_model() {
    echo ""
    echo "======================================"
    echo "Training Model"
    echo "======================================"
    echo "This will take 1-3 hours depending on your hardware"
    read -p "Continue? (y/n): " confirm
    
    if [[ $confirm == "y" || $confirm == "Y" ]]; then
        cd src/models
        python train.py
        cd ../..
        echo "Training complete!"
    else
        echo "Training cancelled"
    fi
}

# Function to run UI
run_ui() {
    echo ""
    echo "======================================"
    echo "Starting Streamlit UI"
    echo "======================================"
    echo "Opening in browser at http://localhost:8501"
    echo "Press Ctrl+C to stop"
    streamlit run ui/app.py
}

# Function to generate samples
generate_samples() {
    echo ""
    echo "======================================"
    echo "Generating Sample Predictions"
    echo "======================================"
    cd src/models
    python inference.py
    cd ../..
    echo "Samples generated in data/outputs/"
}

# Function to run notebook
run_notebook() {
    echo ""
    echo "======================================"
    echo "Opening Training Notebook"
    echo "======================================"
    jupyter notebook notebooks/train_and_evaluate.ipynb
}

# Function to view results
view_results() {
    echo ""
    echo "======================================"
    echo "Results Summary"
    echo "======================================"
    
    if [ -d "results/models/best_model" ]; then
        echo "Best model: results/models/best_model/"
    else
        echo "No trained model found"
    fi
    
    if [ -d "results/plots" ]; then
        echo "Training plots:"
        ls -1 results/plots/
    else
        echo "No plots found"
    fi
    
    if [ -d "results/samples" ]; then
        echo "Sample outputs:"
        ls -1 results/samples/
    else
        echo "No samples found"
    fi
    
    if [ -d "data/processed" ]; then
        echo "Processed datasets:"
        ls -1 data/processed/
    else
        echo "No processed data found"
    fi
}

# Main loop
while true; do
    show_menu
    
    case $choice in
        1)
            create_dataset
            ;;
        2)
            train_model
            ;;
        3)
            run_ui
            ;;
        4)
            generate_samples
            ;;
        5)
            run_notebook
            ;;
        6)
            view_results
            ;;
        0)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            ;;
    esac
done
