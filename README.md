# Aurora: AI Powered Data Analysis Tool

## Overview

Aurora AI represents a breakthrough in accessible, AI-driven data analytics. Built on a robust Streamlit framework, this web application democratizes complex data analysis techniques, making them available to users across all skill levels. At its core, Aurora AI leverages cutting-edge artificial intelligence, particularly the Gemini API, to transform raw data into actionable insights.
The platform offers a suite of six powerful features, each designed to address specific data analysis needs:

1. CleanStats automates data preprocessing and provides deep statistical insights, ensuring data quality and reliability.
2. AutoViz harnesses AI to generate intuitive, customizable visualizations, facilitating exploratory data analysis with ease.
3. InsightGen produces comprehensive reports, translating complex data patterns into clear, actionable narratives.
4. SmartQuery employs natural language processing, allowing users to interact with their data through simple, conversational queries.
5. VisionFusion introduces advanced image analysis capabilities, from object detection to visual content moderation.
6. FutureCast AI offers predictive analytics and recommendations based on sophisticated data analysis.

What sets Aurora AI apart is its seamless integration of these diverse functionalities within a user-friendly interface. The platform's modular architecture ensures scalability and easy feature expansion, while robust error handling and user feedback mechanisms guarantee a smooth user experience.
Aurora AI is not just a tool; it's a comprehensive solution for modern data challenges. Whether you're a data scientist looking to streamline your workflow, a business analyst seeking deeper insights, or a newcomer to data analysis, Aurora AI provides the capabilities you need to unlock the full potential of your data. With its combination of accessibility, advanced AI integration, and powerful analytical features, Aurora AI is poised to revolutionize how we approach and understand data in the digital age.

## Core Features

1. **CleanStats:** Automated data preprocessing and in-depth statistical analysis
2. **AutoViz:** AI-driven data visualization and Exploratory Data Analysis (EDA). 
3. **InsightGen:** Intelligent automated report generation
4. **SmartQuery:** NLP-powered data querying system
5. **VisionFusion:** AI-powered image analysis (New!)
6. **FutureCast AI:** AI-driven recommendations based on dataset analysis (New!)

## Detailed Feature Descriptions

### CleanStats
- **Data Cleaning:** 
  - Automatic detection and removal of duplicate entries
  - Advanced missing value imputation using multiple strategies
  - Outlier detection and handling
- **Statistical Analysis:** 
  - Comprehensive descriptive statistics
  - Correlation analysis with interactive heatmaps
  - Distribution analysis with automatic plot generation

### AutoViz
- Dynamic visualization generation leveraging Matplotlib and Seaborn
- AI-assisted plot selection based on data characteristics
- Interactive customization options for generated visualizations
- Support for various chart types: scatter plots, line charts, bar charts, histograms, box plots, etc.

### InsightGen
- Generates detailed text reports using Gemini AI, providing data-driven insights
- Produces interactive HTML reports with ydata-profiling for comprehensive data understanding
- Customizable report templates to focus on specific aspects of the data

### SmartQuery
- Utilizes Gemini API for advanced natural language understanding
- Performs complex data analysis based on natural language user queries
- Supports a wide range of analytical questions, from simple summaries to complex correlations

### VisionFusion
- AI-powered image analysis capabilities
- Features include:
  - Text extraction from images (OCR)
  - Image based data analysis
  - Visual content moderation

### FutureCast AI
- AI-driven recommendation system based on dataset analysis
- Provides actionable insights and future predictions
- Features include:
  - Trend analysis and forecasting
  - Anomaly detection in time-series data
  - Customer segmentation and personalized recommendations
  - Risk assessment and mitigation strategies

## Technology Stack

- **Backend:** Python 3.12
- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **AI Integration:** Google Generative AI (Gemini API)
- **Authentication:** Streamlit-Authenticator
- **Configuration:** YAML, python-dotenv

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/AnubhavYadavBCA25/Aurora-IA-Project.git
   ```

2. Navigate to the project directory:
   ```
   cd aurora-ai
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up the environment variables:
   - Create a `.env` file in the root directory
   - Add your Gemini API key:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

5. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Aurora Web App Link:
Link: https://aurora-ai.streamlit.app/

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to Anthropic for providing the Gemini API
- Special thanks to all users and supporters of Aurora AI

For any questions or support, please contact the development team or open an issue on the GitHub repository.


