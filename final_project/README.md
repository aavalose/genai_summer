# Fashion Outfit Recommendation System

A generative AI system that provides personalized fashion outfit recommendations based on natural language prompts describing user plans and occasions.

## 🎯 Project Overview

This project addresses the common challenge of outfit selection by creating an AI system that generates personalized fashion recommendations based on natural language descriptions of events or occasions.

### Key Features

- **Natural Language Processing**: Understands user prompts like "I'm going to a beach wedding"
- **Semantic Matching**: Uses advanced text embeddings to find relevant fashion items
- **Occasion-Aware**: Applies different rules for casual, formal, business, party, and beach occasions
- **Multi-Modal Approach**: Combines text descriptions with structured fashion attributes
- **🎨 IMAGE GENERATION**: Creates visual representations of recommended outfits using Stable Diffusion
- **Performance Metrics**: Comprehensive evaluation framework with relevance, completeness, and diversity scores

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- At least 8GB RAM recommended
- **GPU with 4GB+ VRAM recommended for image generation** (CPU fallback available)

### Installation

1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: For GPU acceleration (image generation)**
   ```bash
   # Install PyTorch with CUDA support if you have a compatible GPU
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Fashion_Outfit_Recommendation_System.ipynb
   ```

5. Run all cells to initialize the system

**Note**: The system will automatically detect if GPU and diffusers library are available. If not, it will fall back to simulated image generation.

## 📁 Project Structure

```
final_project/
├── Fashion_Outfit_Recommendation_System.ipynb  # Main project notebook
├── requirements.txt                            # Python dependencies
├── README.md                                  # This file
├── description.txt                            # Project requirements
└── proposal.txt                              # Project proposal
```

## 🔧 System Components

### 1. Data Preprocessing (5%)
- Fashion dataset simulation
- Text processing and feature engineering
- Data exploration and visualization

### 2. Model Implementation (10%)
- Outfit recommendation system architecture
- Text encoder and item matcher
- Rule-based outfit generation
- **Image generation system using Stable Diffusion**

### 3. Methods (5%)
- Sentence transformers for text embeddings
- Cosine similarity for item matching
- Occasion detection algorithms

### 4. Experiments and Results (10%)
- System demonstration with test prompts
- Performance analysis and visualization
- Comprehensive evaluation metrics

## 🎭 Usage Examples

The system can handle various types of prompts and generates both text recommendations and visual outfit images:

- **Formal Events**: "I'm going to a beach wedding" → Elegant formal wear with matching accessories
- **Professional**: "Job interview at a tech company" → Professional business attire with modern styling
- **Casual**: "Casual weekend brunch with friends" → Comfortable yet stylish everyday wear
- **Business**: "Formal business meeting" → Sophisticated business outfits with polished details
- **Social**: "Date night at a fancy restaurant" → Trendy evening wear perfect for upscale dining

### 🎨 Visual Output Types

1. **Real AI Images**: When Stable Diffusion is available, generates photorealistic fashion photos
2. **Enhanced Placeholders**: Professional-looking outfit cards when AI generation isn't available
3. **Fallback Options**: Multiple backup methods ensure you always get quality visualizations

## 📊 Performance Metrics

The system is evaluated using:

- **Relevance Score**: Semantic similarity between prompts and recommendations
- **Completeness Score**: Percentage of outfits meeting occasion requirements
- **Diversity Score**: Variety in recommended categories
- **Price Consistency**: Appropriateness of outfit pricing

## 🔮 Future Enhancements

- Computer vision integration for visual style matching
- Advanced personalization with user preference learning
- Real-time fashion trend integration
- Budget-aware recommendations
- Social features and community ratings

## 🎓 Academic Context

This project was developed as a final project for a Generative AI course, demonstrating:

- Multi-modal AI system design
- Natural language processing applications
- Recommendation system implementation
- Performance evaluation frameworks

## 📝 Technical Details

### Technologies Used

- **Deep Learning**: PyTorch for neural network components
- **NLP**: Transformers library with BERT-based embeddings
- **Image Generation**: Stable Diffusion via Diffusers library
- **Data Science**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, PIL

### Architecture Highlights

- Sentence transformer embeddings for semantic understanding
- Rule-based filtering for fashion domain knowledge
- Cosine similarity for efficient item matching
- Modular design for easy extension and customization

## 🤝 Contributing

This project is part of academic coursework. For educational purposes, feel free to:

- Experiment with different embedding models
- Add new occasion types and rules
- Implement visual similarity features
- Extend the evaluation framework

## 📄 License

This project is for educational purposes as part of coursework requirements.

---

**Built with ❤️ for Generative AI Course Final Project** 